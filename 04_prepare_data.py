"""
This python file is for preparing and cleaning the combined data
"""

from pathlib import Path
import pandas as pd

# Configuration
RESULTS_DIR = Path("results")
PREPARED_DIR = Path("prepared")

# Create a folder for holding prepared data
PREPARED_DIR.mkdir(exist_ok=True)

# Set chunk size as 100000 rows
CHUNK_SIZE = 100000

# Dataset schemas
# Get column names based on the dataset and file type according to the dataset README
def get_columns(name, file_type):
    # RanSAP
    if name == "RanSAP":
        # Storage read
        if file_type == "ata_read.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size"
            ]

        # Storage write
        elif file_type == "ata_write.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "Entropy #1",
                "Entropy #2"
            ]

    # RanSMAP and Benign use the same CSV format
    elif name in ("RanSMAP", "Benign"):
        # Storage read
        if file_type == "ata_read.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "-1",
                "0"
            ]

        # Storage write
        elif file_type == "ata_write.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "entropy",
                "0"
            ]

        # Memory read
        elif file_type == "mem_read.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "0",
                "-1",
                "type"
            ]

        # Memory write
        elif file_type == "mem_write.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "size",
                "entropy",
                "type"
            ]

        # Memory read and write
        elif file_type == "mem_readwrite.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "size",
                "entropy",
                "type"
            ]

        # Memory execute
        elif file_type == "mem_exec.csv":
            return [
                "source_file_id",
                "source_row",
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "0",
                "-1",
                "type"
            ]

    # Skip unknown file types
    return None

# Unused raw columns
# Remove only columns that are placeholders or unused fields in the raw dataset format
def get_unused_columns(name, file_type):
    # RanSAP
    if name == "RanSAP":
        # Keep all meaningful RanSAP columns
        return []

    # RanSMAP and Benign use the same unused column definitions
    elif name in ("RanSMAP", "Benign"):
        # Storage read
        if file_type == "ata_read.csv":
            return ["-1", "0"]

        # Storage write
        elif file_type == "ata_write.csv":
            return ["0"]

        # Memory read
        elif file_type == "mem_read.csv":
            return ["0", "-1"]

        # Memory write
        elif file_type == "mem_write.csv":
            return []

        # Memory read and write
        elif file_type == "mem_readwrite.csv":
            return []

        # Memory execute
        elif file_type == "mem_exec.csv":
            return ["0", "-1"]

    # Keep everything for unknown cases
    return []

# File identification
# Identify dataset and file type from the result filename
def identify_file(file_name):
    # RanSAP
    if file_name.startswith("RanSAP_"):
        if file_name.endswith("_ata_read.csv"):
            return "RanSAP", "ata_read.csv"

        elif file_name.endswith("_ata_write.csv"):
            return "RanSAP", "ata_write.csv"

    # RanSMAP
    elif file_name.startswith("RanSMAP_"):
        if file_name.endswith("_ata_read.csv"):
            return "RanSMAP", "ata_read.csv"

        elif file_name.endswith("_ata_write.csv"):
            return "RanSMAP", "ata_write.csv"

        elif file_name.endswith("_mem_read.csv"):
            return "RanSMAP", "mem_read.csv"

        elif file_name.endswith("_mem_write.csv"):
            return "RanSMAP", "mem_write.csv"

        elif file_name.endswith("_mem_readwrite.csv"):
            return "RanSMAP", "mem_readwrite.csv"

        elif file_name.endswith("_mem_exec.csv"):
            return "RanSMAP", "mem_exec.csv"

    # Benign
    elif file_name.startswith("Benign_"):
        if file_name.endswith("_ata_read.csv"):
            return "Benign", "ata_read.csv"

        elif file_name.endswith("_ata_write.csv"):
            return "Benign", "ata_write.csv"

        elif file_name.endswith("_mem_read.csv"):
            return "Benign", "mem_read.csv"

        elif file_name.endswith("_mem_write.csv"):
            return "Benign", "mem_write.csv"

        elif file_name.endswith("_mem_readwrite.csv"):
            return "Benign", "mem_readwrite.csv"

        elif file_name.endswith("_mem_exec.csv"):
            return "Benign", "mem_exec.csv"

    # Unknown file
    return None, None

# Prepare One File
def prepare_file(input_file):
    # Identify dataset and file type
    name, file_type = identify_file(input_file.name)

    # Skip files that cannot be identified
    if name is None:
        print(f"Skipped: {input_file.name}")
        return

    # Get the expected schema
    columns = get_columns(name, file_type)

    if columns is None:
        print(f"Skipped: {input_file.name}")
        return

    # Get raw placeholder columns that can safely be removed
    unused_columns = get_unused_columns(name, file_type)

    output_file = (PREPARED_DIR / input_file.name)

    # Statistics
    total_rows = 0
    removed_source_id = 0
    removed_null = 0
    removed_non_numeric = 0
    kept_rows = 0

    # Used to write the CSV header only once
    first_chunk = True

    try:
        # Read the combined file in chunks so large datasets do not need to be loaded entirely into memory
        for chunk in pd.read_csv(
            input_file,
            header=0,
            names=columns,
            chunksize=CHUNK_SIZE,
            dtype=str,
            skipinitialspace=True
        ):
            total_rows += len(chunk)

            # Validate source file ID
            source_id = (chunk["source_file_id"].fillna("").astype(str).str.strip())

            valid_source_id = (source_id.ne("") & source_id.str.lower().str.endswith(".csv"))

            removed_source_id += int((~valid_source_id).sum())

            chunk = (chunk.loc[valid_source_id].copy())

            if chunk.empty:
                continue

            # Remove unused raw columns
            # Drop placeholder fields
            chunk = chunk.drop(
                columns=unused_columns,
                errors="ignore"
            )

            # Remove null values
            # Only meaningful retained columns are considered during null validation
            null_rows = (chunk.isna().any(axis=1))

            removed_null += int(null_rows.sum())

            chunk = (chunk.loc[~null_rows].copy())

            if chunk.empty:
                continue

            # Validate numeric values
            # All remaining columns should contain numeric values
            numeric_columns = [
                column
                for column in chunk.columns
                if column != "source_file_id"
            ]

            numeric_valid = pd.Series(
                True,
                index=chunk.index
            )

            for column in numeric_columns:
                converted = pd.to_numeric(chunk[column], errors="coerce")

                numeric_valid &= (converted.notna())

            invalid_numeric = (~numeric_valid)

            removed_non_numeric += int(invalid_numeric.sum())

            chunk = (chunk.loc[numeric_valid].copy())

            if chunk.empty:
                continue

            kept_rows += len(chunk)

            # Save cleaned data
            chunk.to_csv(
                output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

        # Display cleaning result
        print(f"{name}: {input_file.name}")

        print(f"Columns removed: {len(unused_columns)}")

        if unused_columns:
            print(f"Removed columns: {', '.join(unused_columns)}")
        else:
            print("Removed columns: None")

        print(f"Rows: {total_rows}")

        print(f"Removed - invalid source ID: {removed_source_id}")

        print(f"Removed - null values: {removed_null}")

        print(f"Removed - non-numeric values: {removed_non_numeric}")

        print(f"Kept: {kept_rows}\n")

    except Exception as e:
        print(f"{name}: Error preparing {input_file.name}")
        print(f"Reason: {e}")

# Main
if __name__ == "__main__":
    # File title
    print("\nRansomware Data Preparation\n")

    # Check that results folder exists
    if not RESULTS_DIR.exists():
        print("Error: results folder not found.")

    else:
        # Find CSV files only inside results
        result_files = sorted(RESULTS_DIR.glob("*.csv"))

        if not result_files:
            print("No CSV files found in results.")

        else:
            print(f"Found {len(result_files)} result files.\n")

            # Prepare every combined result file
            for input_file in result_files:
                prepare_file(input_file)

            # Closure
            print("Data Preparation Complete")
