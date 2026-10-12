"""
DTBAD Ransomware Detection Performance Evaluation

Labels: 0 = benign (FireFox / Office / Zip); 1 = ransomware (RanSAP / RanSMAP)
"""

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Configuration
EVALUATION_DIR = Path("evaluation")
OUTPUT_DIR = Path("performance")

# benign validation EXECUTION-level scores only
THRESHOLD_PERCENTILE = 95

EXECUTION_KEYS = ["dataset", "execution_id"]
LABEL_COLUMN = "label"

# The score column names match the execution summaries
METHODS = {
    "storage_autoencoder": ("storage", "ae_mean_error"),
    "storage_temporal": ("storage", "temporal_mean_score"),
    "storage_isolation_forest": ("storage", "if_mean_score"),
    "memory_autoencoder": ("memory", "ae_mean_error"),
    "memory_temporal": ("memory", "temporal_mean_score"),
    "memory_isolation_forest": ("memory", "if_mean_score"),
    "DTBAD_combined": ("combined", "combined_score"),
}

INPUT_FILES = {
    "storage": {
        "validation": "storage_validation_execution_scores.csv",
        "test": "storage_execution_scores.csv",
    },
    "memory": {
        "validation": "memory_validation_execution_scores.csv",
        "test": "memory_execution_scores.csv",
    },
    "combined": {
        "validation": "combined_validation_execution_scores.csv",
        "test": "combined_execution_scores.csv",
    },
}


# Load execution scores; Step 08 may create a headerless empty fusion file
def load_results():
    data = {}

    for source, files in INPUT_FILES.items():
        data[source] = {}

        for split, filename in files.items():
            path = EVALUATION_DIR / filename

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing Step 08 output: {path}. "
                    "Run the updated Step 08 before Step 10."
                )

            try:
                frame = pd.read_csv(
                    path, dtype={"dataset": str, "execution_id": str}
                )
            except pd.errors.EmptyDataError:
                if source == "combined":
                    frame = pd.DataFrame()
                else:
                    raise ValueError(f"Required Step 08 output is empty: {path}")

            if frame.empty and source != "combined":
                raise ValueError(f"Required Step 08 output has no executions: {path}")

            data[source][split] = frame

    return data


# Validate keys, binary labels, score columns, and execution-level splits
def validate_results(data):
    execution_splits = {}
    execution_labels = {}

    for source, splits in data.items():
        for split, frame in splits.items():
            if frame.empty:
                continue

            required = EXECUTION_KEYS + [LABEL_COLUMN]
            required += [column for name, (method_source, column) in METHODS.items()
                         if method_source == source]
            missing = sorted(set(required) - set(frame.columns))

            if missing:
                raise ValueError(f"{source}/{split}: missing columns {missing}")

            if frame[EXECUTION_KEYS + [LABEL_COLUMN]].isna().any().any():
                raise ValueError(f"{source}/{split}: missing execution ID or label")

            if frame.duplicated(EXECUTION_KEYS).any():
                raise ValueError(f"{source}/{split}: duplicate execution IDs")

            labels = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
            if not labels.isin([0, 1]).all():
                raise ValueError(f"{source}/{split}: labels must be 0 or 1")

            # The current project's datasets have known ground-truth classes
            expected_labels = {"Benign": 0, "RanSAP": 1, "RanSMAP": 1}
            for dataset, label in zip(frame["dataset"], labels):
                if dataset not in expected_labels:
                    raise ValueError(f"{source}/{split}: unexpected dataset {dataset!r}")
                if int(label) != expected_labels[dataset]:
                    raise ValueError(
                        f"{source}/{split}: label {label} disagrees with dataset {dataset}"
                    )

            for column in set(required) - set(EXECUTION_KEYS + [LABEL_COLUMN]):
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"{source}/{split}: nonfinite scores in {column}")

            if source == "combined" and not frame["dataset"].isin(
                ["RanSMAP", "Benign"]
            ).all():
                raise ValueError("Fusion scores must contain only RanSMAP/Benign")

            for dataset, execution_id, label in frame[
                EXECUTION_KEYS + [LABEL_COLUMN]].itertuples(index=False, name=None):
                key = (dataset, execution_id)
                previous_split = execution_splits.get(key)

                if previous_split is not None and previous_split != split:
                    raise ValueError(f"Execution {key} appears in both validation and test")

                execution_splits[key] = split
                previous_label = execution_labels.get(key)

                if previous_label is not None and previous_label != int(label):
                    raise ValueError(f"Inconsistent labels across modalities for {key}")

                execution_labels[key] = int(label)

    # A fused execution must also exist in storage and memory for the same split
    for split in ("validation", "test"):
        combined = data["combined"][split]

        if combined.empty:
            continue

        fused_keys = set(map(tuple, combined[EXECUTION_KEYS].to_numpy()))

        for source in ("storage", "memory"):
            other_keys = set(map(tuple, data[source][split][EXECUTION_KEYS].to_numpy()))

            if not fused_keys.issubset(other_keys):
                raise ValueError(f"Fused {split} executions do not all exist in {source}")


# Set a threshold from benign validation executions only; never use test labels
def calibrate_threshold(validation, score_column):
    benign = validation.loc[validation[LABEL_COLUMN] == 0, score_column]

    if benign.empty:
        raise ValueError(
            f"No benign validation executions for {score_column}; cannot calibrate an unsupervised detection threshold."
        )

    if len(benign) < 20:
        warnings.warn(
            f"Only {len(benign)} benign validation executions for {score_column}; "
            "the 95th-percentile threshold may be unstable.",
            stacklevel=2
        )

    threshold = float(np.percentile(benign.to_numpy(dtype=float), THRESHOLD_PERCENTILE))

    return threshold, int(len(benign))


# Calculate execution-level binary metrics from a fixed threshold
def evaluate_predictions(labels, predictions, scores):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    specificity = tn / (tn + fp) if tn + fp else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    fnr = fn / (fn + tp) if fn + tp else float("nan")

    has_both_classes = (np.any(labels == 0) and np.any(labels == 1))
    roc_auc = float(roc_auc_score(labels, scores)) if has_both_classes else float("nan")
    balanced_accuracy = (recall + specificity) / 2 if has_both_classes else float("nan")

    return {
        "test_executions": len(labels),
        "benign_executions": int(np.sum(labels == 0)),
        "ransomware_executions": int(np.sum(labels == 1)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# Save and display confusion matrices for all evaluated methods
def plot_confusion_matrices(metrics):
    count = len(metrics)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.6 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, record in zip(axes, metrics):
        matrix = np.array([[record["tn"], record["fp"]],
                           [record["fn"], record["tp"]]], dtype=int)
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row, column]),
                        ha="center", va="center", fontsize=13)

        ax.set_xticks([0, 1], ["Benign", "Ransomware"])
        ax.set_yticks([0, 1], ["Benign", "Ransomware"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(record["method"].replace("_", " "), fontsize=10)

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle("Step 10 — Test Execution Confusion Matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_confusion_matrices.png", dpi=160)

    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# Generate a readable report for the paper and the console
def create_report(metrics, thresholds, skipped):
    lines = [
        "Ground truth: 0 = benign (Benign), 1 = ransomware (RanSAP/RanSMAP)\n",
        f"Decision rule: score > {THRESHOLD_PERCENTILE}% of benign",

        "\nClassification Metrics",
        pd.DataFrame(metrics)[
            ["method", "test_executions", "benign_executions",
             "ransomware_executions", "accuracy", "precision", "recall",
             "f1_score", "false_positive_rate", "false_negative_rate", "roc_auc"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    ]

    lines += ["", "Confusion Matrices (TN, FP, FN, TP)"]
    for record in metrics:
        lines.append(
            f"{record['method']}: TN={record['tn']}, FP={record['fp']}, "
            f"FN={record['fn']}, TP={record['tp']}"
        )

    lines += ["", "Detection Thresholds"]
    for method, settings in thresholds["methods"].items():
        lines.append(
            f"{method}: threshold={settings['threshold']:.8g} "
            f"({settings['benign_validation_executions']} benign validation executions)"
        )

    if skipped:
        lines += ["", "Skipped Methods"]
        lines.extend(skipped)

    return "\n".join(lines) + "\n"

# Main
if __name__ == "__main__":
    if not 0 < THRESHOLD_PERCENTILE < 100:
        raise ValueError("THRESHOLD_PERCENTILE must be between 0 and 100")

    # File title
    print("\nDTBAD Ransomware Detection Performance Evaluation")

    data = load_results()
    validate_results(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = []
    predictions = []
    confusion_rows = []
    skipped = []

    thresholds = {
        "label_0": "benign",
        "label_1": "ransomware",
        "calibration": "benign_validation_execution_percentile",
        "percentile": THRESHOLD_PERCENTILE,
        "decision_rule": "score > threshold => 1; otherwise 0",
        "methods": {}
    }

    for method, (source, score_column) in METHODS.items():
        validation = data[source]["validation"]
        test = data[source]["test"]

        # Fusion is optional if Step 08 had no matching executions
        if source == "combined" and (validation.empty or test.empty):
            reason = f"{method}: no paired validation/test executions"
            skipped.append(reason)
            print(f"Skipping {reason}")

            continue

        threshold, benign_count = calibrate_threshold(validation, score_column)

        labels = test[LABEL_COLUMN].to_numpy(dtype=int)
        scores = test[score_column].to_numpy(dtype=float)

        predicted = (scores > threshold).astype(int)

        record = {
            "method": method,
            "score_column": score_column,
            "threshold": threshold,
            "benign_validation_executions": benign_count,
            **evaluate_predictions(labels, predicted, scores)
        }
        metrics.append(record)

        thresholds["methods"][method] = {
            "source": source,
            "score_column": score_column,
            "threshold": threshold,
            "benign_validation_executions": benign_count
        }

        for (dataset, execution_id), label, score, prediction in zip(
            test[EXECUTION_KEYS].itertuples(index=False, name=None),
            labels, scores, predicted
        ):
            predictions.append({
                "method": method,
                "dataset": dataset,
                "execution_id": execution_id,
                "actual_label": int(label),
                "predicted_label": int(prediction),
                "score": float(score),
                "threshold": threshold,
                "correct": int(label == prediction)
            })

        for actual, predicted_label, count in (
            (0, 0, record["tn"]),
            (0, 1, record["fp"]),
            (1, 0, record["fn"]),
            (1, 1, record["tp"])
        ):
            confusion_rows.append({
                "method": method,
                "actual_label": actual,
                "predicted_label": predicted_label,
                "count": count
            })

    if not metrics:
        raise ValueError("No evaluation methods could be calculated")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)

    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / "execution_predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
    (OUTPUT_DIR / "detection_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")

    report = create_report(metrics, thresholds, skipped)
    (OUTPUT_DIR / "performance_report.txt").write_text(report, encoding="utf-8")

    # Display report on screen and show chart; retain every output on disk
    print("\n" + report)
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")

    plot_confusion_matrices(metrics)

    # Closure
    print("\nPerformance Evaluation Complete")