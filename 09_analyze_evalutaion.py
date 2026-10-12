"""
Analyze DTBAD evaluation results.

This is descriptive analysis of labeled novelty scores. Accuracy/precision/recall/F1 require a validated binary decision rule.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
EVALUATION_DIR = Path("evaluation")
OUTPUT_DIR = Path("analysis")
EXECUTION_KEYS = ["dataset", "execution_id"]
WINDOW_KEYS = EXECUTION_KEYS + ["window_start_ns", "window_end_ns"]

INPUT_FILES = {
    "summary": "evaluation_summary.csv",
    "storage": "storage_execution_scores.csv",
    "memory": "memory_execution_scores.csv",
    "combined": "combined_execution_scores.csv",
    "storage_windows": "storage_window_scores.csv",
    "memory_windows": "memory_window_scores.csv",
}

# Columns produced by Step 08. Absent optional columns are skipped
EXECUTION_METRICS = [
    "ae_mean_error",
    "ae_novelty_rate",
    "temporal_mean_score",
    "temporal_max_score",
    "temporal_novelty_rate",
    "max_consecutive_novel",
    "normalized_temporal_score",
    "if_mean_score",
    "if_novelty_rate"
]

WINDOW_METRICS = [
    "ae_error", "temporal_score", "if_score",
    "ae_novel", "temporal_novel", "if_novel"
]

FUSION_METRICS = [
    "combined_score",
    "normalized_temporal_score_storage",
    "normalized_temporal_score_memory"
]

OUTLIER_QUANTILE = 0.95

# Read all Step 08 outputs; tolerate a headerless empty fusion CSV
def load_results():
    loaded = {}

    for name, filename in INPUT_FILES.items():
        path = EVALUATION_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required Step 08 output missing: {path}. Run Step 08 first."
            )

        try:
            frame = pd.read_csv(
                path, dtype={"dataset": "string", "execution_id": "string"}
            )

        except pd.errors.EmptyDataError:
            if name == "combined":
                frame = pd.DataFrame()

            else:
                raise ValueError(f"Required CSV has no header or rows: {path}")

        if frame.empty and name != "combined":
            raise ValueError(f"Required Step 08 result is empty: {path}")

        loaded[name] = frame

    return loaded

# Check identifiers, duplicates, score validity and fusion provenance
def validate_data(data):
    rows = []

    for name, frame in data.items():
        if frame.empty:
            rows.append({
                "file": name,
                "rows": 0,
                "unique_executions": 0,
                "duplicate_keys": 0,
                "missing_ids": 0,
                "invalid_scores": 0,
                "note": "No paired fusion results" if name == "combined" else ""
            })

            continue

        keys = WINDOW_KEYS if name.endswith("_windows") else EXECUTION_KEYS
        if name == "summary":
            keys = ["method"]

        missing = [c for c in keys if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing key columns: {missing}")

        # Labels must be present and consistent in every execution-level result
        if name != "summary":
            if "label" not in frame.columns:
                raise ValueError(f"{name} is missing the label column")

            if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
                raise ValueError(f"{name} contains invalid binary labels")

            if (frame.groupby(EXECUTION_KEYS)["label"].nunique() > 1).any():
                raise ValueError(f"{name} contains conflicting execution labels")

        duplicates = int(frame.duplicated(keys).sum())

        missing_ids = int(frame[keys].isna().any(axis=1).sum())
        if duplicates or missing_ids:
            raise ValueError(
                f"{name}: {duplicates} duplicate keys, {missing_ids} missing keys"
            )

        metrics = [
            c for c in EXECUTION_METRICS + WINDOW_METRICS + FUSION_METRICS
            if c in frame.columns
        ]

        invalid_scores = 0

        for col in metrics:
            numbers = pd.to_numeric(frame[col], errors="coerce")

            invalid_scores += int((~np.isfinite(numbers.to_numpy(dtype=float))).sum())

        if invalid_scores:
            raise ValueError(f"{name} contains {invalid_scores} invalid score values")

        count = (frame[EXECUTION_KEYS].drop_duplicates().shape[0]
                 if all(c in frame.columns for c in EXECUTION_KEYS) else 0)

        rows.append({
            "file": name,
            "rows": len(frame),
            "unique_executions": count,
            "duplicate_keys": duplicates,
            "missing_ids": missing_ids,
            "invalid_scores": invalid_scores,
            "note": ""
        })

    # Fusion must contain matched RanSMAP or Benign executions from both modalities
    combined = data["combined"]

    if not combined.empty:
        if not combined["dataset"].isin(["RanSMAP", "Benign"]).all():
            raise ValueError("Fusion includes executions outside RanSMAP and Benign")

        storage_keys = set(map(tuple, data["storage"][EXECUTION_KEYS].to_numpy()))
        memory_keys = set(map(tuple, data["memory"][EXECUTION_KEYS].to_numpy()))
        fusion_keys = set(map(tuple, combined[EXECUTION_KEYS].to_numpy()))

        if not fusion_keys.issubset(storage_keys & memory_keys):
            raise ValueError("Fusion contains unmatched storage/memory executions")

        storage_labels = data["storage"].set_index(EXECUTION_KEYS)["label"]
        memory_labels = data["memory"].set_index(EXECUTION_KEYS)["label"]
        fusion_labels = combined.set_index(EXECUTION_KEYS)["label"]

        for key, label in fusion_labels.items():
            if label != storage_labels.loc[key] or label != memory_labels.loc[key]:
                raise ValueError(f"Fusion label mismatch for {key}")

    for left_name, right_name in (("storage", "memory"), ("storage", "combined"), ("memory", "combined")):
        left = data[left_name]
        right = data[right_name]

        if left.empty or right.empty:
            continue

        shared = left[EXECUTION_KEYS + ["label"]].merge(
            right[EXECUTION_KEYS + ["label"]],
            on=EXECUTION_KEYS, how="inner",
            suffixes=("_left", "_right")
        )

        if not shared["label_left"].eq(shared["label_right"]).all():
            raise ValueError(f"Inconsistent labels between {left_name} and {right_name}")

    return pd.DataFrame(rows)

# Summarize numeric scores without interpreting novelty as ransomware
def score_statistics(data):
    records = []

    for name, frame in data.items():
        if name == "summary" or frame.empty:
            continue

        columns = (FUSION_METRICS if name == "combined"
                   else WINDOW_METRICS if name.endswith("_windows")
                   else EXECUTION_METRICS)

        for column in columns:
            if column not in frame.columns:
                continue

            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)

            records.append({
                "source": name,
                "metric": column,
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values))
            })

    return pd.DataFrame(records)

# Flag top 5% within each modality, using temporal or fusion scores
def identify_outliers(data):
    records = []

    for source, metric in (
        ("storage", "temporal_mean_score"),
        ("memory", "temporal_mean_score"),
        ("combined", "combined_score")
    ):
        frame = data[source]

        if frame.empty or metric not in frame:
            continue

        values = frame[metric].to_numpy(dtype=float)

        cutoff = float(np.quantile(values, OUTLIER_QUANTILE))

        # A strict '>' avoids labeling every execution if all scores are equal
        selected = frame.loc[frame[metric] > cutoff, EXECUTION_KEYS + ["label", metric]]

        for record in selected.to_dict("records"):
            records.append({
                "source": source, **record,
                "score_metric": metric,
                "score": record[metric],
                "p95_cutoff": cutoff,
                "interpretation": "High relative novelty; not a ransomware label"
            })

    return pd.DataFrame(records, columns=[
        "source", "dataset", "execution_id", "label", "score_metric", "score",
        "p95_cutoff", "interpretation"])

# Save and display descriptive histograms for each model and the fusion score
def plot_distributions(data):
    choices = [
        ("storage", "temporal_mean_score", "Storage temporal score"),
        ("memory", "temporal_mean_score", "Memory temporal score"),
        ("combined", "combined_score", "RanSMAP/Benign fused score")
    ]

    available = [(name, metric, label) for name, metric, label in choices
                 if not data[name].empty and metric in data[name]]

    fig, axes = plt.subplots(len(available), 1, figsize=(9, 3.4 * len(available)))

    if len(available) == 1:
        axes = [axes]

    for ax, (name, metric, label) in zip(axes, available):
        ax.hist(data[name][metric].to_numpy(dtype=float), bins=30)
        ax.set_title(label)
        ax.set_xlabel("Novelty score (not ransomware probability)")
        ax.set_ylabel("Execution count")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_score_distributions.png", dpi=160)

# Save and display window novelty rates summarized per execution
def plot_novelty_rates(data):
    choices = [
        ("storage", "ae_novelty_rate", "Storage AE"),
        ("storage", "temporal_novelty_rate", "Storage temporal"),
        ("storage", "if_novelty_rate", "Storage IF"),
        ("memory", "ae_novelty_rate", "Memory AE"),
        ("memory", "temporal_novelty_rate", "Memory temporal"),
        ("memory", "if_novelty_rate", "Memory IF")
    ]

    labels, means = [], []

    for name, col, label in choices:
        if col in data[name]:
            labels.append(label)
            means.append(float(data[name][col].mean()))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, means)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean fraction of novel windows")
    ax.set_title("Validation-calibrated novelty rates on test executions")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_novelty_rates.png", dpi=160)

# Produce a readable report with defensible interpretations
def write_report(data, quality, statistics, outliers):
    storage = data["storage"]
    memory = data["memory"]
    combined = data["combined"]
    summary = data["summary"]
    lines = [
        f"Storage test executions: {len(storage):,}",
        f"\tRanSAP: {(storage['dataset'] == 'RanSAP').sum():,}",
        f"\tRanSMAP: {(storage['dataset'] == 'RanSMAP').sum():,}",
        f"\tBenign: {(storage['dataset'] == 'Benign').sum():,}",
        f"\tBenign (label 0): {(storage['label'] == 0).sum():,}",
        f"\tRansomware (label 1): {(storage['label'] == 1).sum():,}",
        f"Memory test executions: {len(memory):,}",
        f"\tBenign (label 0): {(memory['label'] == 0).sum():,}",
        f"\tRansomware (label 1): {(memory['label'] == 1).sum():,}",
        f"Fused RanSMAP/Benign executions: {len(combined):,}",

        "\nEvaluation summary (unmodified):",
        summary.to_string(index=False),

        "\nScore statistics:",
        statistics.to_string(index=False, float_format=lambda x: f"{x:.6g}"),

        f"\nTop-{(1 - OUTLIER_QUANTILE) * 100:.0f}% high-score executions: {len(outliers)}",

        "\nData quality:",
        quality.to_string(index=False)
    ]

    report = "\n".join(lines) + "\n"

    (OUTPUT_DIR / "analysis_report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)


if __name__ == "__main__":
    # File title
    print("\nAnalyze DTBAD Evaluation Results")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    quality = validate_data(data)
    statistics = score_statistics(data)
    outliers = identify_outliers(data)

    quality.to_csv(OUTPUT_DIR / "data_quality.csv", index=False)
    statistics.to_csv(OUTPUT_DIR / "score_statistics.csv", index=False)
    outliers.to_csv(OUTPUT_DIR / "execution_outliers.csv", index=False)
    plot_distributions(data)
    plot_novelty_rates(data)
    write_report(data, quality, statistics, outliers)

    print(f"Storage executions: {len(data['storage']):,}")
    print(f"Memory executions: {len(data['memory']):,}")
    print(f"Fused RanSMAP/Benign executions: {len(data['combined']):,}")
    print(f"Outlier candidates: {len(outliers):,}")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")

    # Show both saved figures in interactive windows
    plt.show()
    plt.close("all")

    # Closure
    print("\nEvaluation Analysis Complete")