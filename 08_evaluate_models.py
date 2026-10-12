"""
Evaluate the proposed DTBAD architecture.

Component A: Behavioral reconstruction using trained autoencoders
Component B: Temporal consistency using rolling reconstruction errors
Component C: Storage-memory integration for paired RanSMAP executions

Baseline: Isolation Forest
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import torch
from torch import nn

# Configuration
SPLITS_DIR = Path("splits")
MODELS_DIR = Path("models")
EVALUATION_DIR = Path("evaluation")

DATA_TYPES = ["storage", "memory"]
SPLIT_NAMES = ["train", "validation", "test"]

# Initial temporal analysis configuration
ROLLING_WINDOW = 5

# Initial fusion configuration
STORAGE_WEIGHT = 0.50
MEMORY_WEIGHT = 0.50

# Validation-based novelty calibration
NOVELTY_PERCENTILE = 95

BATCH_SIZE = 4096

METADATA_COLUMNS = [
    "dataset",
    "execution_id",
    "window_start_ns",
    "window_end_ns"
]

EXECUTION_COLUMNS = ["dataset", "execution_id"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder architecture
class BehavioralAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)

        return self.decoder(latent)

# Load trained models
def load_models(data_type):
    ae_path = (MODELS_DIR / f"{data_type}_autoencoder.pt")
    if_path = (MODELS_DIR / f"{data_type}_isolation_forest.joblib")

    if not ae_path.exists():
        raise FileNotFoundError(ae_path)

    if not if_path.exists():
        raise FileNotFoundError(if_path)

    # Only load model files created by our own trusted training script
    ae_bundle = torch.load(ae_path, map_location="cpu", weights_only=False)
    if_bundle = joblib.load(if_path)

    architecture = ae_bundle["architecture"]

    autoencoder = BehavioralAutoencoder(
        architecture["input_size"],
        architecture["hidden_size"],
        architecture["latent_size"]
    )
    autoencoder.load_state_dict(ae_bundle["model_state_dict"])
    autoencoder = autoencoder.to(DEVICE)
    autoencoder.eval()

    # Both models must use the same feature order
    if (ae_bundle["feature_columns"] != if_bundle["feature_columns"]):
        raise ValueError(f"{data_type}: inconsistent feature columns.")

    return (autoencoder, ae_bundle, if_bundle)

# Rejoin operation-specific columns by the exact execution/window key
SPLIT_FILE_GROUPS = {
    "storage": ("storage_read", "storage_write"),
    "memory": ("memory_read", "memory_write", "memory_readwrite", "memory_execute"),
}

EXPECTED_FEATURE_COUNTS = {"storage": 26, "memory": 60}

KEY_COLUMNS = METADATA_COLUMNS
NON_FEATURE_COLUMNS = METADATA_COLUMNS + ["label"]

# Load every row from the six Step 06 outputs; join same-window columns
def load_split_data(data_type, split_name):
    result = None

    for group in SPLIT_FILE_GROUPS[data_type]:
        path = SPLITS_DIR / f"{group}_{split_name}.csv"

        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")

        current = pd.read_csv(path, dtype={"dataset": str, "execution_id": str})

        if current.empty:
            raise ValueError(f"Input file is empty: {path}")

        if current.columns.duplicated().any():
            raise ValueError(f"Duplicate column names in {path}")

        missing = set(KEY_COLUMNS + ["label"]) - set(current.columns)

        if missing:
            raise ValueError(f"Missing window keys in {path}: {sorted(missing)}")

        if current[KEY_COLUMNS + ["label"]].isna().any().any():
            raise ValueError(f"Missing window identifiers in {path}")

        if not current["label"].isin([0, 1]).all():
            raise ValueError(f"Invalid binary labels in {path}")

        if current.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"Duplicate execution windows in {path}")

        if result is None:
            result = current

            continue

        # Check overlapping feature values before dropping duplicate columns
        shared = [c for c in current.columns
                  if c not in KEY_COLUMNS and c in result.columns]

        check = result[KEY_COLUMNS + shared].merge(
            current[KEY_COLUMNS + shared],
            on=KEY_COLUMNS,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_left", "_right"),
            sort=False,
        )

        if (len(check) != len(result) or len(current) != len(result) or not
        check["_merge"].eq("both").all()):
            raise RuntimeError(f"Window mismatch while joining {path}")

        for column in shared:
            left = pd.to_numeric(check[f"{column}_left"], errors="raise").to_numpy()
            right = pd.to_numeric(check[f"{column}_right"], errors="raise").to_numpy()

            if not np.allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=False):
                raise RuntimeError(f"Shared feature {column} differs in {path}")

        new_columns = [c for c in current.columns
                       if c not in KEY_COLUMNS and c not in result.columns]

        result = result.merge(current[KEY_COLUMNS + new_columns],
                              on=KEY_COLUMNS,
                              how="left",
                              validate="one_to_one",
                              sort=False)

        if len(result) != len(check):
            raise RuntimeError(f"Rows changed while joining {path}")

    actual = len([c for c in result.columns if c not in NON_FEATURE_COLUMNS])

    if actual != EXPECTED_FEATURE_COUNTS[data_type]:
        raise ValueError(f"{data_type}: expected {EXPECTED_FEATURE_COUNTS[data_type]} unique features, found {actual}")

    return result


# Validate execution-level splitting
def validate_split_integrity(all_data):
    """Reject leakage within a modality and between storage and memory."""
    assignments = {}
    labels = {}
    for data_type in DATA_TYPES:
        execution_sets = {}
        for split_name in SPLIT_NAMES:
            dataframe = all_data[data_type][split_name]
            current = set(zip(dataframe["dataset"], dataframe["execution_id"]))
            for (dataset, execution_id), group in dataframe.groupby(EXECUTION_COLUMNS):
                key = (dataset, execution_id)
                if group["label"].nunique() != 1:
                    raise RuntimeError(f"Conflicting labels within execution {key}")
                label = int(group["label"].iloc[0])
                if key in labels and labels[key] != label:
                    raise RuntimeError(f"Conflicting labels across modalities for {key}")
                labels[key] = label
            execution_sets[split_name] = current
            for key in current:
                previous = assignments.get(key)
                if previous is not None and previous != split_name:
                    raise RuntimeError(
                        f"Execution {key} appears in {previous} and {split_name} "
                        "across modalities or within a modality."
                    )
                assignments[key] = split_name
        for i, first in enumerate(SPLIT_NAMES):
            for second in SPLIT_NAMES[i + 1:]:
                if execution_sets[first] & execution_sets[second]:
                    raise RuntimeError(
                        f"{data_type}: execution leakage between {first} and {second}"
                    )


# Prepare model input
def prepare_model_input(dataframe, ae_bundle):
    feature_columns = ae_bundle["feature_columns"]
    missing = (set(feature_columns) - set(dataframe.columns))

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    features = dataframe[feature_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity.")

    # Reuse the exact StandardScaler parameters fitted on training data
    mean = np.asarray(ae_bundle["scaler_mean"])
    scale = np.asarray(ae_bundle["scaler_scale"])

    scaled = ((features - mean) / scale).astype(np.float32)

    if not np.isfinite(scaled).all():
        raise ValueError("Invalid values after scaling.")

    return scaled

# Component A: reconstruction errors
def calculate_reconstruction_errors(model, features):
    errors = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            batch = torch.from_numpy(
                features[start:start + BATCH_SIZE]).to(DEVICE)

            reconstruction = model(batch)

            batch_errors = torch.mean((batch - reconstruction) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)

# Isolation forest baseline
def calculate_isolation_scores(model, features):
    # sklearn returns lower scores for more abnormal observations
    return -model.score_samples(features)

# Calculate window scores
def calculate_window_scores(dataframe, autoencoder, ae_bundle, if_bundle):
    features = prepare_model_input(dataframe, ae_bundle)

    result = dataframe[METADATA_COLUMNS + ["label"]].copy()

    # Component A
    result["ae_error"] = calculate_reconstruction_errors(autoencoder, features)
    # Baseline
    result["if_score"] = calculate_isolation_scores(if_bundle["model"], features)

    return result

# Component B: Temporal Consistency
def add_temporal_scores(dataframe):
    result = dataframe.copy()

    # Calculate rolling averages separately for each execution
    result["temporal_score"] = (
        result.groupby(EXECUTION_COLUMNS, sort=False)["ae_error"].transform(
            lambda values: values.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
    )

    return result

# Calibrate thresholds
def percentile_threshold(values):
    return float(np.percentile(values, NOVELTY_PERCENTILE))


def calibrate_thresholds(validation_scores):
    # Reference novelty thresholds, not ransomware detection thresholds
    return {
        "ae_error": percentile_threshold(validation_scores["ae_error"]),
        "temporal_score": percentile_threshold(validation_scores["temporal_score"]),
        "if_score": percentile_threshold(validation_scores["if_score"])
    }

# Normalize scores for fusion
def fit_score_normalizer(validation_scores):
    # Estimate normalization parameters using validation data only
    values = validation_scores["temporal_score"].to_numpy(dtype=np.float64)

    median = float(np.median(values))

    q25, q75 = np.percentile(values, [25, 75])

    scale = float(q75 - q25)

    if scale <= 0:
        scale = float(np.std(values))

    if scale <= 0:
        scale = 1.0

    return {"median": median, "scale": scale}

def normalize_scores(values, normalizer):
    return (values - normalizer["median"]) / normalizer["scale"]

# Apply calibrated thresholds
def add_novelty_flags(dataframe, thresholds):
    result = dataframe.copy()
    result["ae_novel"] = (result["ae_error"] > thresholds["ae_error"]).astype(int)
    result["temporal_novel"] = (result["temporal_score"] > thresholds["temporal_score"]).astype(int)
    result["if_novel"] = (result["if_score"] > thresholds["if_score"]).astype(int)

    return result

# Calculate maximum consecutive novel windows
def maximum_consecutive_ones(values):
    longest = 0
    current = 0

    for value in values:
        if value == 1:
            current += 1
            longest = max(longest, current)

        else:
            current = 0

    return longest

# Summarize complete executions
def summarize_executions(dataframe, normalizer):
    records = []

    for (dataset, execution_id), group in dataframe.groupby(EXECUTION_COLUMNS, sort=False):
        if group["label"].nunique() != 1:
            raise RuntimeError(f"Inconsistent labels for execution {(dataset, execution_id)}")

        temporal_values = group["temporal_score"].to_numpy()

        normalized_temporal = normalize_scores(temporal_values, normalizer)

        records.append({
            "dataset": dataset,
            "execution_id": execution_id,
            "label": int(group["label"].iloc[0]),
            "window_count": len(group),

            # Component A
            "ae_mean_error": group["ae_error"].mean(),
            "ae_novelty_rate": group["ae_novel"].mean(),

            # Component B
            "temporal_mean_score": (temporal_values.mean()),
            "temporal_max_score": (temporal_values.max()),
            "temporal_novelty_rate": group["temporal_novel"].mean(),

            "max_consecutive_novel": (
                maximum_consecutive_ones(group["temporal_novel"].to_numpy())
            ),

            # Normalized temporal score for fusion
            "normalized_temporal_score": (normalized_temporal.mean()),

            # Isolation Forest baseline
            "if_mean_score": group["if_score"].mean(),
            "if_novelty_rate": group["if_novel"].mean(),
        })

    return pd.DataFrame(records)

# Check storage-memory split consistency
def check_fusion_split_consistency(all_data):
    storage_assignments = {}
    memory_assignments = {}

    for split_name in SPLIT_NAMES:
        for data_type, assignments in [
            ("storage", storage_assignments),
            ("memory", memory_assignments)
        ]:
            dataframe = all_data[data_type][split_name]

            executions = dataframe[EXECUTION_COLUMNS].drop_duplicates()

            for dataset, execution_id in (
                executions.itertuples(index=False, name=None)
            ):
                if dataset not in ("RanSMAP", "Benign"):
                    continue

                assignments[(dataset, execution_id)] = split_name

    shared = (set(storage_assignments) & set(memory_assignments))

    if not shared:
        return False

    # Reject fusion if a shared execution was assigned to different splits
    for execution in shared:
        if (storage_assignments[execution] != memory_assignments[execution]):
            return False

    return True

# Component C: storage-memory integration
def combine_execution_scores(storage_summary, memory_summary):
    # Fuse matched RanSMAP ransomware and Benign benign executions
    storage = storage_summary[storage_summary["dataset"].isin(["RanSMAP", "Benign"])].copy()
    memory = memory_summary[memory_summary["dataset"].isin(["RanSMAP", "Benign"])].copy()
    combined = storage.merge(
        memory,
        on=EXECUTION_COLUMNS,
        how="inner",
        suffixes=("_storage", "_memory"),
        validate="one_to_one"
    )

    if combined.empty:
        return combined

    if not combined["label_storage"].eq(combined["label_memory"]).all():
        raise RuntimeError("Storage and memory labels differ for matched executions.")
    combined["label"] = combined["label_storage"].astype(int)
    combined = combined.drop(columns=["label_storage", "label_memory"])

    # Component C: weighted integration of calibrated temporal behavioral scores
    combined["combined_score"] = (
        STORAGE_WEIGHT * combined[
            "normalized_temporal_score_storage"
        ] + MEMORY_WEIGHT * combined[
            "normalized_temporal_score_memory"
        ]
    )

    return combined

# Summarize evaluation results
def create_evaluation_summary(execution_results, combined_results):
    records = []

    for data_type in DATA_TYPES:
        dataframe = execution_results[data_type]

        records.append({
            "method": (f"{data_type}_autoencoder"),
            "executions": len(dataframe),
            "mean_score": dataframe["ae_mean_error"].mean(),
            "mean_novelty_rate": dataframe["ae_novelty_rate"].mean()
        })

        records.append({
            "method": (f"{data_type}_temporal"),
            "executions": len(dataframe),
            "mean_score": dataframe["temporal_mean_score"].mean(),
            "mean_novelty_rate": dataframe["temporal_novelty_rate"].mean()
        })

        records.append({
            "method": (f"{data_type}_isolation_forest"),
            "executions": len(dataframe),
            "mean_score": dataframe["if_mean_score"].mean(),
            "mean_novelty_rate": dataframe["if_novelty_rate"].mean()
        })

    if not combined_results.empty:
        records.append({
            "method": "DTBAD_combined",
            "executions": len(combined_results),
            "mean_score": combined_results["combined_score"].mean(),

            # No combined decision threshold has been calibrated
            "mean_novelty_rate": np.nan
        })

    return pd.DataFrame(records)

# Main
if __name__ == "__main__":
    if ROLLING_WINDOW < 1:
        raise ValueError("ROLLING_WINDOW must be positive.")

    if (STORAGE_WEIGHT < 0 or MEMORY_WEIGHT < 0 or not np.isclose(STORAGE_WEIGHT + MEMORY_WEIGHT, 1.0)):
        raise ValueError("Fusion weights must be nonnegative and sum to 1.")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # File title
    print("\nDTBAD Model Evaluation")

    # Validate execution splits across both modalities before scoring
    all_data = {}

    for data_type in DATA_TYPES:
        all_data[data_type] = {}

        for split_name in SPLIT_NAMES:
            all_data[data_type][split_name] = load_split_data(data_type, split_name)

    validate_split_integrity(all_data)
    fusion_allowed = check_fusion_split_consistency(all_data)

    execution_results = {}
    validation_execution_results = {}

    # Evaluate storage and memory separately
    for data_type in DATA_TYPES:
        print(f"\nEvaluating {data_type}")

        (autoencoder, ae_bundle, if_bundle) = load_models(data_type)

        # Validation data is used to calibrate temporal scoring and normalization
        validation_data = all_data[data_type]["validation"]
        validation_scores = calculate_window_scores(
                validation_data,
                autoencoder,
                ae_bundle,
                if_bundle
        )
        validation_scores = add_temporal_scores(validation_scores)

        # Fit thresholds and normalizers on normal behavior only
        benign_validation_scores = validation_scores.loc[validation_scores["label"] == 0]
        if benign_validation_scores.empty:
            raise ValueError(f"{data_type}: no benign validation windows for calibration.")
        thresholds = calibrate_thresholds(benign_validation_scores)
        normalizer = fit_score_normalizer(benign_validation_scores)

        # Save validation execution scores for Step 10 threshold calibration
        validation_scores = add_novelty_flags(validation_scores, thresholds)
        validation_scores["normalized_temporal_score"] = normalize_scores(
            validation_scores["temporal_score"], normalizer
        )
        validation_execution_summary = summarize_executions(
            validation_scores, normalizer
        )
        validation_execution_results[data_type] = validation_execution_summary
        validation_execution_summary.to_csv(
            EVALUATION_DIR / f"{data_type}_validation_execution_scores.csv",
            index=False
        )

        # Evaluate the untouched test executions
        test_data = all_data[data_type]["test"]
        test_scores = calculate_window_scores(
                test_data,
                autoencoder,
                ae_bundle,
                if_bundle
        )
        test_scores = add_temporal_scores(test_scores)
        test_scores = add_novelty_flags(test_scores, thresholds)

        # Store the normalized temporal score alongside the original window scores
        test_scores["normalized_temporal_score"] = normalize_scores(
            test_scores["temporal_score"],
            normalizer
        )

        execution_summary = summarize_executions(test_scores, normalizer)
        execution_results[data_type] = execution_summary

        # Save window-level and execution-level evaluation results
        test_scores.to_csv(EVALUATION_DIR / f"{data_type}_window_scores.csv", index=False)
        execution_summary.to_csv(EVALUATION_DIR / f"{data_type}_execution_scores.csv", index=False)

        # Save calibration settings so future datasets use the same parameters
        joblib.dump({
                "thresholds": thresholds,
                "normalizer": normalizer,
                "rolling_window": ROLLING_WINDOW,
                "novelty_percentile": NOVELTY_PERCENTILE,
            },
            EVALUATION_DIR / f"{data_type}_calibration.joblib"
        )

        print(f"Test executions: {len(execution_summary)}")
        print("Evaluation complete")

    # Component C: combined evaluation
    combined_results = pd.DataFrame()

    if fusion_allowed:
        combined_results = combine_execution_scores(
                execution_results["storage"],
                execution_results["memory"]
        )

        print("\nStorage-memory integration complete")

    else:
        print("\nStorage-memory integration skipped: matching executions do not "
            "have consistent split assignments."
        )

    """
     Save validation fusion scores so Step 10 can calibrate a fusion
     decision threshold without inspecting the test executions
    """
    combined_validation_results = pd.DataFrame()
    if fusion_allowed:
        combined_validation_results = combine_execution_scores(
            validation_execution_results["storage"],
            validation_execution_results["memory"]
        )
    combined_validation_results.to_csv(
        EVALUATION_DIR / "combined_validation_execution_scores.csv", index=False
    )

    # Save the combined file even when fusion cannot be performed
    combined_results.to_csv(EVALUATION_DIR / "combined_execution_scores.csv", index=False)

    # Final Summary
    summary = create_evaluation_summary(execution_results, combined_results)
    summary.to_csv(EVALUATION_DIR / "evaluation_summary.csv", index=False)

    # Closure
    print("\nModel Evaluation Complete")