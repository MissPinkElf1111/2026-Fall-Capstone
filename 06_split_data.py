"""
Split six behavioral feature datasets into:
    training (70%)
    validation (15%),
    testing (15%)
by complete execution.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Configuration
FEATURES_DIR = Path("features")
SPLITS_DIR = Path("splits")
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Step 05 outputs: storage contains RanSAP and RanSMAP; memory contains RanSMAP.
FEATURE_FILES = {
    "storage_read": "storage_read_features.csv",
    "storage_write": "storage_write_features.csv",
    "memory_read": "memory_read_features.csv",
    "memory_write": "memory_write_features.csv",
    "memory_readwrite": "memory_readwrite_features.csv",
    "memory_execute": "memory_execute_features.csv",
}

EXECUTION_COLUMNS = ["dataset", "execution_id"]
WINDOW_COLUMNS = EXECUTION_COLUMNS + ["window_start_ns", "window_end_ns"]
SPLIT_NAMES = ("train", "validation", "test")

# Check that all three ratios form a valid partition
def validate_configuration():
    ratios = (TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO)
    if any(not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("Split ratios must be between 0 and 1.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must equal 1.")

# Load a complete Step 05 output without sampling or dropping rows
def load_feature_data(data_type, filename):
    path = FEATURES_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Feature file is empty: {path}")

    missing = set(WINDOW_COLUMNS) - set(dataframe.columns)

    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    if dataframe[WINDOW_COLUMNS].isna().any().any():
        raise ValueError(f"{filename} has missing execution/window identifiers.")

    if dataframe.duplicated(subset=WINDOW_COLUMNS).any():
        raise ValueError(f"{filename} contains duplicate execution windows.")

    # The two storage files must contain both source datasets.
    if data_type.startswith("storage_"):
        found = set(dataframe["dataset"].unique())
        expected = {"RanSAP", "RanSMAP"}

        if not expected.issubset(found):
            raise ValueError(f"{filename} must contain RanSAP and RanSMAP; found {sorted(found)}")

    return dataframe

# Calculate execution counts, keeping each global partition nonempty
def calculate_split_counts(number_of_executions):
    if number_of_executions < 3:
        raise ValueError("At least three distinct executions are required.")

    train_count = max(1, int(number_of_executions * TRAIN_RATIO))
    validation_count = max(1, int(number_of_executions * VALIDATION_RATIO))
    test_count = number_of_executions - train_count - validation_count

    if test_count < 1:
        raise ValueError("Not enough executions for three nonempty splits.")

    return train_count, validation_count, test_count

# Assign each (dataset, execution_id) pair exactly once across all modalities
def create_global_execution_split(feature_data):
    executions = pd.concat(
        [frame[EXECUTION_COLUMNS] for frame in feature_data.values()],
        ignore_index=True,
    ).drop_duplicates().sort_values(EXECUTION_COLUMNS).reset_index(drop=True)

    count = len(executions)
    train_count, validation_count, _ = calculate_split_counts(count)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_indices = rng.permutation(count)

    assignments = np.empty(count, dtype=object)
    assignments[shuffled_indices[:train_count]] = "train"
    assignments[shuffled_indices[train_count:train_count + validation_count]] = "validation"
    assignments[shuffled_indices[train_count + validation_count:]] = "test"

    executions["split"] = assignments

    return executions

# Join the shared assignment onto every original feature window
def apply_execution_split(dataframe, global_split, data_type):
    result = dataframe.merge(
        global_split,
        on=EXECUTION_COLUMNS,
        how="left",
        sort=False,
        validate="many_to_one"
    )

    if len(result) != len(dataframe):
        raise RuntimeError(f"Row count changed for {data_type}.")

    if result["split"].isna().any():
        raise RuntimeError(f"Some {data_type} windows have no split assignment.")

    if (result.groupby(EXECUTION_COLUMNS)["split"].nunique() > 1).any():
        raise RuntimeError(f"Execution leakage detected in {data_type}.")

    return result

# Write three CSVs for one feature type, retaining the original schema
def save_split_files(dataframe, data_type):
    total_saved = 0

    for split_name in SPLIT_NAMES:
        partition = dataframe.loc[dataframe["split"] == split_name].drop(columns="split")
        output = SPLITS_DIR / f"{data_type}_{split_name}.csv"
        partition.to_csv(output, index=False)
        total_saved += len(partition)
        execution_count = len(partition[EXECUTION_COLUMNS].drop_duplicates())

        print(f"  {split_name}: {execution_count} executions, {len(partition)} windows")

    if total_saved != len(dataframe):
        raise RuntimeError(f"Rows were lost when saving {data_type} splits.")


if __name__ == "__main__":
    validate_configuration()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # File title
    print("\nExecution-Level Data Splitting\n")

    # Load and validate all six datasets before assigning any executions
    feature_data = {
        data_type: load_feature_data(data_type, filename)
        for data_type, filename in FEATURE_FILES.items()
    }

    # One assignment shared by RanSAP/RanSMAP storage and RanSMAP memory
    global_split = create_global_execution_split(feature_data)
    global_split.to_csv(SPLITS_DIR / "execution_split_assignments.csv", index=False)

    print(f"Unique executions across all files: {len(global_split)}")

    for split_name in SPLIT_NAMES:
        print(f"  {split_name}: {(global_split['split'] == split_name).sum()} executions")

    # Apply that assignment to every feature file, without changing features
    for data_type, dataframe in feature_data.items():
        print(f"\n{data_type}: {len(dataframe)} windows")

        split_data = apply_execution_split(dataframe, global_split, data_type)
        save_split_files(split_data, data_type)

    # Closure
    print("Data splitting complete")