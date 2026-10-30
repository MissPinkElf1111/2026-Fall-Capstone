"""
Compare the original paper's 23 features:
    Storage: 5 features
    Memory: 18 features

Against our expanded 86 features:
    Storage: 26 features
    Memory: 60 features

Training: Benign training windows only.
Calibration: Benign validation windows and executions only.
Evaluation: Held-out test executions.

Additional experiment - Compare storage/memory fusion weights:
        40/60
        50/50
        60/40
"""

from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Configuration
# Paper feature splits
PAPER_SPLITS_DIR = Path("splits_paper")
# Original execution assignments from Step 06
ASSIGNMENTS_FILE = Path("splits/execution_split_assignments.csv")
# Original 86-feature evaluation results
BASE_EVALUATION_DIR = Path("evaluation")
# Separate output directory for the ablation study
OUTPUT_DIR = Path("ablation")
# Store the paper's trained models separately
MODEL_DIR = OUTPUT_DIR / "models_paper"

# Reproducibility
RANDOM_SEED = 42

# Autoencoder training configuration
BATCH_SIZE = 256
SCORE_BATCH_SIZE = 4096
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
MIN_DELTA = 0.00001

# Isolation Forest configuration
IF_ESTIMATORS = 200

# Temporal consistency configuration
ROLLING_WINDOW = 5

# Benign validation percentile for detection thresholds
THRESHOLD_PERCENTILE = 95

# Storage and memory fusion configurations
FUSION_WEIGHTS = [(0.40, 0.60), (0.50, 0.50), (0.60, 0.40)]

# Two behavioral modalities
MODALITIES = ("storage", "memory")

# Execution-level data partitions
SPLITS = ("train", "validation", "test")

# Execution identifiers
KEYS = ["dataset", "execution_id"]

# Execution identifiers and window boundaries
WINDOW_KEYS = KEYS + ["window_start_ns", "window_end_ns"]

# Metadata excluded from model input
META = WINDOW_KEYS + ["label"]

# Expected paper feature counts
PAPER_FEATURE_COUNTS = {"storage": 5, "memory": 18}

# Expected expanded feature counts
EXPANDED_FEATURE_COUNTS = {"storage": 26, "memory": 60}

# Evaluation methods from the original DTBAD architecture
SCORE_METHODS = {
    "autoencoder": "ae_mean_error",
    "temporal": "temporal_mean_score",
    "isolation_forest": "if_mean_score"
}

# Use GPU when available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
def set_seed():
    # Set random seeds for Python, NumPy, and PyTorch
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Request deterministic operations when supported
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load Paper Feature Splits
def read_paper_split(modality, split):
    path = (PAPER_SPLITS_DIR / f"{modality}_paper_{split}.csv")

    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 06 paper output: {path}")

    dataframe = pd.read_csv(
        path,
        dtype={"dataset": str, "execution_id": str}
    )

    if dataframe.empty:
        raise ValueError(f"Empty paper split: {path}")

    # Reject duplicate column names
    if dataframe.columns.duplicated().any():
        raise ValueError(f"Duplicate columns: {path}")

    # Ensure all required metadata exists
    missing = set(META) - set(dataframe.columns)

    if missing:
        raise ValueError(f"Missing metadata in {path}: {sorted(missing)}")

    # Reject missing identifiers or duplicate windows
    if (
        dataframe[META].isna().any().any()
        or dataframe.duplicated(WINDOW_KEYS).any()
    ):
        raise ValueError(f"Missing metadata or duplicate windows in {path}")

    # Labels must be binary
    if not dataframe["label"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {path}")

    # Identify model input features
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in META
    ]

    expected = PAPER_FEATURE_COUNTS[modality]

    if len(feature_columns) != expected:
        raise ValueError(f"{path}: expected {expected} features, found {len(feature_columns)}")

    # Check for nonnumeric or infinite feature values
    values = dataframe[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite features in {path}")

    # Window ending time must follow starting time
    if (dataframe["window_end_ns"] <= dataframe["window_start_ns"]).any():
        raise ValueError(f"Invalid time windows in {path}")

    # Sort windows chronologically for temporal analysis
    dataframe = dataframe.sort_values(WINDOW_KEYS, kind="stable").reset_index(drop=True)

    return dataframe, feature_columns

# Validate Shared Execution Assignments
def validate_assignments(paper):
    if not ASSIGNMENTS_FILE.is_file():
        raise FileNotFoundError(f"Run Step 06 first: {ASSIGNMENTS_FILE}")

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        dtype={
            "dataset": str,
            "execution_id": str
        }
    )

    required = KEYS + ["label", "split"]

    if not set(required).issubset(assignments.columns):
        raise ValueError("Original split assignments have missing columns")

    if assignments.duplicated(KEYS).any():
        raise ValueError("Duplicate execution assignments")

    for modality in MODALITIES:
        for split in SPLITS:

            dataframe = paper[modality][split]

            # Match paper executions to original assignments
            matched = dataframe[KEYS + ["label"]].merge(
                assignments[required],
                on=KEYS,
                how="left",
                validate="many_to_one",
                suffixes=("", "_original")
            )

            # Every paper execution must have an assignment
            if matched["split"].isna().any():
                raise ValueError(f"{modality}/{split}: execution missing from original assignments")

            # Ensure the same execution is in the same partition
            if not matched["split"].eq(split).all():
                raise ValueError(f"{modality}/{split}: execution leakage or incorrect split")

            # Ground-truth labels must agree
            if not matched["label"].eq(matched["label_original"]).all():
                raise ValueError(f"{modality}/{split}: labels disagree with original assignments")

# Autoencoder Architecture
class BehavioralAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        # Compress the behavioral feature vector
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

        # Reconstruct the original feature vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Create batches for model training
def make_loader(features, shuffle):
    dataset = TensorDataset(torch.from_numpy(features))

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

# Train Paper Autoencoder
def train_autoencoder(x_train, x_validation):
    input_size = x_train.shape[1]

    # Match the architecture selection used in Step 07
    hidden_size = max(8, input_size // 2)

    latent_size = min(
        max(2, input_size // 4),
        hidden_size - 1,
        input_size - 1
    )

    model = BehavioralAutoencoder(
        input_size,
        hidden_size,
        latent_size
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    train_loader = make_loader(x_train, shuffle=True)

    validation_loader = make_loader(x_validation, shuffle=False)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    waiting = 0

    # Train until the maximum epochs or early stopping
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for (batch,) in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(reconstruction, batch)

            loss.backward()
            optimizer.step()

        # Evaluate on benign validation windows
        model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():

            for (batch,) in validation_loader:
                batch = batch.to(DEVICE)

                reconstruction = model(batch)

                loss = criterion(reconstruction, batch)
                total_loss += loss.item() * len(batch)
                total_samples += len(batch)

        validation_loss = (total_loss / total_samples)

        if not np.isfinite(validation_loss):
            raise RuntimeError("Non-finite autoencoder validation loss")

        # Keep the model with the lowest validation loss
        if validation_loss < best_loss - MIN_DELTA:

            best_loss = validation_loss
            best_epoch = epoch
            waiting = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:
            waiting += 1

            if waiting >= PATIENCE:
                break

    # Restore the best-performing model
    model.load_state_dict(best_state)
    model.eval()

    information = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "latent_size": latent_size,
        "best_epoch": best_epoch,
        "validation_loss": best_loss
    }

    return model, information

# Train Paper Models
def train_paper_models(
    modality,
    data,
    feature_columns
):
    # Train on benign windows only
    train = data["train"].loc[data["train"]["label"] == 0]

    validation = data["validation"].loc[data["validation"]["label"] == 0]

    if train.empty or validation.empty:
        raise ValueError(f"{modality}: benign training and validation data are required")

    # Fit StandardScaler on benign training windows
    scaler = StandardScaler()

    x_train = scaler.fit_transform(train[feature_columns]).astype(np.float32)

    x_validation = scaler.transform(validation[feature_columns]).astype(np.float32)

    if not (
        np.isfinite(x_train).all()
        and np.isfinite(x_validation).all()
    ):
        raise ValueError(f"{modality}: invalid scaled values")

    # Train behavioral autoencoder
    model, information = train_autoencoder(x_train, x_validation)

    # Train Isolation Forest baseline
    forest = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    forest.fit(x_train)

    # Save paper autoencoder separately from Step 07
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": information,
            "feature_columns": feature_columns,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_
        },
        MODEL_DIR / f"{modality}_paper_autoencoder.pt"
    )

    # Save paper Isolation Forest separately
    joblib.dump(
        {
            "model": forest,
            "scaler": scaler,
            "feature_columns": feature_columns
        },
        MODEL_DIR / f"{modality}_paper_isolation_forest.joblib"
    )

    return model, forest, scaler, information

# Window-Level Anomaly Scoring
def score_windows(
    dataframe,
    feature_columns,
    model,
    forest,
    scaler
):
    # Apply the scaler fitted on benign training data
    features = scaler.transform(dataframe[feature_columns]).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Non-finite model input after scaling")

    errors = []

    model.eval()

    # Calculate autoencoder reconstruction errors
    with torch.no_grad():
        for start in range(
            0,
            len(features),
            SCORE_BATCH_SIZE
        ):
            batch = torch.from_numpy(
                features[start:start + SCORE_BATCH_SIZE]
            ).to(DEVICE)

            reconstruction = model(batch)
            batch_errors = torch.mean((batch - reconstruction) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())

    # Preserve execution and window metadata
    scores = dataframe[META].copy()

    # Component A: reconstruction error
    scores["ae_error"] = np.concatenate(errors)

    # Isolation Forest baseline
    # Higher scores indicate greater abnormality
    scores["if_score"] = -forest.score_samples(features)

    # Component B: temporal consistency
    # Rolling average is calculated separately per execution
    scores["temporal_score"] = (
        scores.groupby(KEYS, sort=False)["ae_error"].transform(
            lambda values: values.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
    )

    return scores

# Validation Calibration
def fit_calibration(validation_scores):
    benign = validation_scores.loc[validation_scores["label"] == 0]

    if benign.empty:
        raise ValueError("No benign validation windows for calibration")

    # Novelty thresholds for each scoring method
    thresholds = {
        name: float(
            np.percentile(
                benign[name],
                THRESHOLD_PERCENTILE
            )
        )
        for name in ("ae_error", "if_score", "temporal_score")
    }

    # Robust normalization for temporal fusion scores
    values = benign["temporal_score"].to_numpy(dtype=float)

    median = float(np.median(values))

    q25, q75 = np.percentile(values,[25, 75])

    scale = float(q75 - q25)

    # Fall back to standard deviation when IQR is zero
    if scale <= 0:
        scale = float(np.std(values))

    if scale <= 0:
        scale = 1.0

    normalizer = {"median": median, "scale": scale}

    return thresholds, normalizer

# Execution-Level Score Aggregation
def summarize_executions(
    scores,
    thresholds,
    normalizer
):
    scores = scores.copy()

    # Normalize temporal scores using benign validation statistics
    scores["normalized_temporal_score"] = (
        (scores["temporal_score"] - normalizer["median"]) / normalizer["scale"]
    )

    # Identify unfamiliar behavior in each window
    for name in ("ae_error", "if_score", "temporal_score"):
        scores[f"{name}_novel"] = (
            scores[name] > thresholds[name]
        ).astype(int)

    # Reject inconsistent labels within an execution
    if (scores.groupby(KEYS)["label"].nunique() > 1).any():
        raise ValueError("Conflicting labels within an execution")

    # Summarize complete executions
    result = scores.groupby(KEYS, sort=False).agg(
        label=("label", "first"),
        window_count=("ae_error", "size"),
        ae_mean_error=("ae_error", "mean"),
        temporal_mean_score=("temporal_score", "mean"),
        if_mean_score=("if_score", "mean"),
        normalized_temporal_score=("normalized_temporal_score", "mean")
    ).reset_index()

    return result

# Run Paper Feature Branch
def run_paper_branch(paper, feature_columns):
    results = {}
    training_information = []

    for modality in MODALITIES:
        print(f"\nTraining paper {modality} model ({len(feature_columns[modality])} features)")

        # Train paper Autoencoder and Isolation Forest
        model, forest, scaler, information = (
            train_paper_models(
                modality,
                paper[modality],
                feature_columns[modality]
            )
        )

        # Calculate validation window scores
        validation_scores = score_windows(
            paper[modality]["validation"],
            feature_columns[modality],
            model,
            forest,
            scaler
        )

        # Calibrate on benign validation windows
        thresholds, normalizer = fit_calibration(validation_scores)

        results[modality] = {}

        # Evaluate validation and test executions
        for split, window_scores in (
            ("validation", validation_scores),
            (
                "test",
                score_windows(
                    paper[modality]["test"],
                    feature_columns[modality],
                    model,
                    forest,
                    scaler
                )
            )
        ):
            summary = summarize_executions(
                window_scores,
                thresholds,
                normalizer
            )

            output = OUTPUT_DIR / f"{modality}_paper_{split}_execution_scores.csv"
            summary.to_csv(output, index=False)
            results[modality][split] = summary

        # Save paper calibration parameters
        joblib.dump(
            {
                "thresholds": thresholds,
                "normalizer": normalizer,
                "rolling_window": ROLLING_WINDOW
            },
            MODEL_DIR / f"{modality}_paper_calibration.joblib"
        )

        training_information.append({
            "modality": modality,
            "features": len(feature_columns[modality]),
            **information
        })

        print(f"\tBest epoch: {information['best_epoch']}; validation MSE: {information['validation_loss']:.6f}")

    # Save model training summary
    pd.DataFrame(training_information).to_csv(
        OUTPUT_DIR / "paper_training_summary.csv",
        index=False
    )

    return results

# Load Original 86-Feature Results
def load_expanded_scores():
    results = {}

    required = set(
        KEYS + ["label", "window_count", "normalized_temporal_score"] + list(SCORE_METHODS.values())
    )

    for modality in MODALITIES:
        results[modality] = {}

        for split in ("validation", "test"):
            if split == "validation":
                filename = (f"{modality}_validation_execution_scores.csv")
            else:
                filename = (f"{modality}_execution_scores.csv")

            path = BASE_EVALUATION_DIR / filename

            if not path.is_file():
                raise FileNotFoundError(f"Run Step 08 first: {path}")

            dataframe = pd.read_csv(
                path,
                dtype={
                    "dataset": str,
                    "execution_id": str
                }
            )

            missing = required - set(dataframe.columns)

            if dataframe.empty or missing:
                raise ValueError(f"Missing execution scores in {path}: {sorted(missing)}")

            if dataframe.duplicated(KEYS).any():
                raise ValueError(f"Duplicate executions in {path}")

            # Verify numeric execution scores
            for column in required - set(KEYS):
                values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)

                if not np.isfinite(values).all():
                    raise ValueError(f"Invalid {column} in {path}")

            results[modality][split] = dataframe

    return results

# Match Executions Across Feature Configurations
def align_executions(
    expanded,
    paper,
    modality,
    split
):
    original = expanded[modality][split]
    paper_data = paper[modality][split]

    paired = original.merge(
        paper_data,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_86", "_23")
    )

    if paired.empty:
        raise ValueError(f"{modality}/{split}: no common executions between 86 and 23 features")

    # Verify labels agree
    if not paired["label_86"].eq(paired["label_23"]).all():
        raise ValueError(f"{modality}/{split}: labels differ between feature sets")

    # Verify execution window counts agree
    if not paired["window_count_86"].eq(paired["window_count_23"]).all():
        raise ValueError(f"{modality}/{split}: window counts differ; fix Step 05A alignment before ablation")

    # Report when only a common subset can be compared
    if (
        len(paired) < len(original)
        or len(paired) < len(paper_data)
    ):
        print(f"\tNote: {modality}/{split}: evaluating {len(paired)} common executions "
            f"(86: {len(original)}, 23: {len(paper_data)})"
        )

    return paired

# Detection Performance Metrics
def detection_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    # Scores above the threshold indicate ransomware
    predictions = (scores > threshold).astype(int)

    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    # Precision
    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    # Recall / True positive rate
    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    # Specificity / True negative rate
    specificity = (
        tn / (tn + fp)
        if tn + fp
        else float("nan")
    )

    # F1-score
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    # ROC-AUC requires both classes
    if len(np.unique(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = float("nan")

    return {
        "executions": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": (
            fp / (fp + tn)
            if fp + tn
            else float("nan")
        ),
        "specificity": specificity,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

# Evaluate One Feature Configuration
def evaluate_variant(
    validation,
    test,
    score_column,
    tag
):
    benign = validation.loc[validation["label_86"] == 0, score_column]

    if benign.empty:
        raise ValueError(f"{tag}: no benign validation executions")

    # Detection threshold uses benign validation only
    threshold = float(
        np.percentile(benign.to_numpy(dtype=float), THRESHOLD_PERCENTILE)
    )

    # Validation metrics are used for weight selection
    validation_metrics = detection_metrics(
        validation["label_86"],
        validation[score_column],
        threshold
    )

    # Test metrics measure held-out performance
    test_metrics = detection_metrics(
        test["label_86"],
        test[score_column],
        threshold
    )

    return {
        "threshold": threshold,
        "validation_f1": validation_metrics["f1_score"],
        "benign_validation_executions": len(benign),
        **test_metrics
    }

# Storage-Memory Fusion
def fuse(modality_scores, split):
    storage = modality_scores["storage"][split]
    memory = modality_scores["memory"][split]

    allowed = ["RanSMAP", "Benign", "Begin"]

    storage = storage.loc[storage["dataset"].isin(allowed)]
    memory = memory.loc[memory["dataset"].isin(allowed)]

    # Match executions across modalities
    merged = storage[KEYS + ["label", "normalized_temporal_score"]].merge(
        memory[KEYS + ["label", "normalized_temporal_score"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_storage", "_memory")
    )

    if merged.empty:
        raise ValueError(f"No matched storage-memory executions for {split}")

    if not merged["label_storage"].eq(merged["label_memory"]).all():
        raise ValueError("Storage-memory labels disagree")

    merged["label"] = merged["label_storage"]

    return merged[
        KEYS + ["label", "normalized_temporal_score_storage", "normalized_temporal_score_memory"]
    ]

# Match Fused Executions Across Feature Configurations
def align_fused(expanded, paper, split):
    original = fuse(expanded, split)

    paper_data = fuse(paper, split)

    # Match identical executions across both feature sets
    paired = original.merge(
        paper_data,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_86", "_23")
    )

    if paired.empty:
        raise ValueError(f"Fusion/{split}: no shared executions")

    if not paired["label_86"].eq(paired["label_23"]).all():
        raise ValueError(f"Fusion/{split}: inconsistent labels")

    if (
        len(paired) < len(original)
        or len(paired) < len(paper_data)
    ):
        print(f"\tNote: fusion/{split}: {len(paired)} common executions "
            f"(86: {len(original)}, 23: {len(paper_data)})"
        )

    return paired

# Feature and Fusion Weight Comparisons
def run_comparisons(expanded, paper):
    feature_rows = []
    weight_rows = []

    # Compare individual storage and memory methods
    for modality in MODALITIES:

        validation = align_executions(
            expanded,
            paper,
            modality,
            "validation"
        )

        test = align_executions(
            expanded,
            paper,
            modality,
            "test"
        )

        for method, score in SCORE_METHODS.items():
            for feature_set in ("86", "23"):
                score_column = (f"{score}_{feature_set}")

                result = evaluate_variant(
                    validation,
                    test,
                    score_column,
                    f"{modality}_{method}_{feature_set}"
                )

                counts = (
                    EXPANDED_FEATURE_COUNTS
                    if feature_set == "86"
                    else PAPER_FEATURE_COUNTS
                )

                feature_rows.append({
                    "method": f"{modality}_{method}",
                    "feature_set": feature_set,
                    "feature_count": counts[modality],
                    **result
                })

    # Compare storage-memory fusion configurations
    validation = align_fused(
        expanded,
        paper,
        "validation"
    )

    test = align_fused(
        expanded,
        paper,
        "test"
    )

    for feature_set in ("86", "23"):
        for storage_weight, memory_weight in FUSION_WEIGHTS:
            score_column = (f"fused_{feature_set}_{int(storage_weight * 100)}")

            # Calculate weighted fusion scores
            for dataframe in (validation, test):
                dataframe[score_column] = (
                    storage_weight
                    * dataframe["normalized_temporal_score_storage_" + feature_set]
                    + memory_weight
                    * dataframe["normalized_temporal_score_memory_" + feature_set]
                )

            result = evaluate_variant(
                validation,
                test,
                score_column,
                f"fusion_{feature_set}_{storage_weight}"
            )

            row = {
                "method": "DTBAD_combined",
                "feature_set": feature_set,
                "feature_count": int(feature_set),
                "storage_weight": storage_weight,
                "memory_weight": memory_weight,
                **result
            }

            weight_rows.append(row)

            # Add default 50/50 fusion to feature comparison
            if np.isclose(storage_weight, 0.50):
                feature_rows.append({
                    key: value
                    for key, value in row.items()
                    if key not in ("storage_weight", "memory_weight")
                })

    # Convert results to DataFrames
    feature_results = pd.DataFrame(feature_rows)
    weight_results = pd.DataFrame(weight_rows)

    # Save feature comparison metrics
    feature_results.to_csv(
        OUTPUT_DIR / "feature_comparison.csv",
        index=False
    )

    # Save fusion weight comparison metrics
    weight_results.to_csv(
        OUTPUT_DIR / "fusion_weight_comparison.csv",
        index=False
    )

    return feature_results, weight_results

# Generate Ablation Report
def write_report(features, weights):
    report = [
        "DTBAD - STEP 08A ABLATION STUDY",
        "\n",
        "Original: 86 features (26 storage, 60 memory)",
        "Paper:    23 features (5 storage, 18 memory)\n",

        "FEATURE COMPARISON (50/50 fusion for combined method)",
        features[
            [
                "method",
                "feature_set",
                "executions",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
                "roc_auc"
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        ),
        "\n",
        "FUSION WEIGHT COMPARISON",
        weights[
            [
                "feature_set",
                "storage_weight",
                "memory_weight",
                "validation_f1",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
                "roc_auc"
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    ]

    # Select the best fusion weight using validation F1
    for feature_set, group in weights.groupby("feature_set"):
        # Prefer 50/50 when validation F1 scores tie
        best = group.assign(
            distance_from_equal=(group["storage_weight"] - 0.50).abs()
        ).sort_values(
            [
                "validation_f1",
                "distance_from_equal"
            ],
            ascending=[False, True]
        ).iloc[0]

        report.append(
            f"{feature_set} features: "
            f"storage={best['storage_weight']:.2f}, "
            f"memory={best['memory_weight']:.2f}, "
            f"validation F1={best['validation_f1']:.4f}\n"
        )

    content = "\n".join(report) + "\n"

    # Save the report
    output = OUTPUT_DIR / "ablation_report.txt"

    output.write_text(content, encoding="utf-8")

    # Display the report in the terminal
    print("\n" + content)

# Image Output: Save and Display
def save_and_show(fig, filename):
    path = OUTPUT_DIR / filename

    # Store the image permanently in the output folder
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    print(f"Saved image: {path}")

    # Display in a notebook or graphical Python environment
    plt.show()

    # Release figure memory
    plt.close(fig)

# Image: Feature Comparison
def plot_feature_comparison(features):
    methods = list(features["method"].drop_duplicates())

    labels = [
        method.replace("_", " ")
        for method in methods
    ]

    positions = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    for offset, feature_set in (
        (-width / 2, "23"),
        (width / 2, "86")
    ):

        subset = features.loc[features["feature_set"] == feature_set].set_index("method")

        values = subset.reindex(methods)["f1_score"].to_numpy(dtype=float)

        ax.bar(
            positions + offset,
            values,
            width,
            label=f"{feature_set} features"
        )

    ax.set_xticks(
        positions,
        labels,
        rotation=30,
        ha="right"
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out test F1-score")
    ax.set_title("Optional - Feature Ablation (50/50 Fusion)" )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    # Save and display the image
    save_and_show(
        fig,
        "08A_feature_comparison.png"
    )

# Image: Fusion Weight Comparison
def plot_fusion_weight_comparison(weights):
    fig, ax = plt.subplots(figsize=(9, 5))

    for feature_set in ("23", "86"):
        subset = weights.loc[weights["feature_set"] == feature_set].sort_values("storage_weight")

        x = subset["storage_weight"].to_numpy(dtype=float) * 100

        # Validation F1 is used for selecting fusion weights
        ax.plot(
            x,
            subset["validation_f1"],
            marker="o",
            linestyle="--",
            label=f"{feature_set} features - validation"
        )

        # Test F1 is reported for performance analysis
        ax.plot(
            x,
            subset["f1_score"],
            marker="s",
            linestyle="-",
            label=f"{feature_set} features - test"
        )

    ax.set_xticks(
        [weight * 100 for weight, _ in FUSION_WEIGHTS],
        [
            f"{weight:.0%}/{memory:.0%}"
            for weight, memory in FUSION_WEIGHTS
        ],
    )

    ax.set_ylim(0, 1.05)

    ax.set_xlabel("Storage / memory fusion weight")
    ax.set_ylabel("Execution-level F1-score")
    ax.set_title("Step 08A - Fusion Weight Comparison")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    # Save and display the image
    save_and_show(
        fig,
        "08A_fusion_weight_comparison.png"
    )

# Image: Confusion Matrices
def plot_fusion_confusion_matrices(features):
    subset = features.loc[features["method"] == "DTBAD_combined"].set_index("feature_set")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4)
    )

    for ax, feature_set in zip(
        axes,
        ("23", "86")
    ):
        record = subset.loc[feature_set]

        # Rows: actual labels
        # Columns: predicted labels
        matrix = np.array([
            [record["tn"], record["fp"]],
            [record["fn"], record["tp"]]
        ], dtype=int)

        ax.imshow(matrix, cmap="Blues")

        # Display the count in each matrix cell
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=13
                )

        ax.set_xticks([0, 1], ["Benign", "Ransomware"])
        ax.set_yticks([0, 1], ["Benign", "Ransomware"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{feature_set} features - 50/50 fusion")

    fig.suptitle("Step 08A - Held-out Test Confusion Matrices")
    fig.tight_layout()

    # Save and display the image
    save_and_show(
        fig,
        "08A_fusion_confusion_matrices.png",
    )

# Main
if __name__ == "__main__":
    # Set random seeds
    set_seed()

    # Validate fusion weight configurations
    for storage_weight, memory_weight in FUSION_WEIGHTS:

        if (
            min(storage_weight, memory_weight) < 0
            or not np.isclose(
                storage_weight + memory_weight,
                1.0
            )
        ):
            raise ValueError("Fusion weights must be nonnegative and sum to one")

    # File title
    print("\nDTBAD Ablation Study\n")
    print(f"Device: {DEVICE}")

    # Load paper feature splits
    paper = {}
    feature_columns = {}

    for modality in MODALITIES:
        paper[modality] = {}

        for split in SPLITS:
            dataframe, columns = read_paper_split(modality, split)

            # All partitions must have the same feature order
            if (
                modality in feature_columns
                and feature_columns[modality] != columns
            ):
                raise ValueError(f"{modality}: inconsistent paper feature order across splits")

            feature_columns[modality] = columns
            paper[modality][split] = dataframe

    # Confirm paper data uses original execution assignments
    validate_assignments(paper)

    # Load original Step 08 execution scores
    expanded = load_expanded_scores()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train and evaluate the 23-feature configuration
    paper_scores = run_paper_branch(paper, feature_columns)

    # Compare the 23-feature and 86-feature configurations
    features, weights = run_comparisons(expanded, paper_scores)

    # Generate the text report
    write_report(features, weights)

    # Generate, save, and display all visualizations
    plot_feature_comparison(features)
    plot_fusion_weight_comparison(weights)
    plot_fusion_confusion_matrices(features)

    # Closure
    print(f"\nAblation Study Complete. \nFiles saved in {OUTPUT_DIR.resolve()}")