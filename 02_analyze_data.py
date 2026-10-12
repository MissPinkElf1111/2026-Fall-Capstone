"""
This python file is for analyzing the data from CSV files
"""

from pathlib import Path
import pandas as pd

# Configuration
RANSAP_DIR = Path("RanSAP")
RANSMAP_DIR = Path("RanSMAP")
BEGIN_DIR = Path("Begin")


# Get column names based on the dataset according to README
def get_columns(name, file_name):
    # RanSAP
    if name == "RanSAP":
        # read data file
        if file_name.endswith("ata_read.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size"
            ]
        # write data file
        elif file_name.endswith("ata_write.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "Entropy #1",
                "Entropy #2"
            ]

    # RanSMAP and Begin use the same CSV format
    elif name in ("RanSMAP", "Begin"):
        # read file data of storage access
        if file_name.endswith("ata_read.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "-1",
                "0"
            ]
        # write file data of storage access
        elif file_name.endswith("ata_write.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "entropy",
                "0"
            ]
        # read file data of memory access
        elif file_name.endswith("mem_read.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "0",
                "-1",
                "type"
            ]
        # write file data of memory access
        elif file_name.endswith("mem_write.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "size",
                "entropy",
                "type"
            ]
        # read & write data file of memory access
        elif file_name.endswith("mem_readwrite.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "size",
                "entropy",
                "type"
            ]
        # execute data file of memory access
        elif file_name.endswith("mem_exec.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "0",
                "-1",
                "type"
            ]

    # If the file name is not belongs to any of the above, then skip
    return None


# Analyze one dataset
def analyze_dataset(name, dataset):
    # Check whether the dataset exists
    if not dataset.exists():
        print(f"{name}: Dataset not found.")
        return

    # Find all CSV files
    csv_files = list(dataset.rglob("*.csv"))

    # File types to analyze
    file_types = [
        "ata_read.csv",
        "ata_write.csv",
        "mem_read.csv",
        "mem_write.csv",
        "mem_readwrite.csv",
        "mem_exec.csv"
    ]

    # Display the name of current dataset
    print(f"Dataset: {name}")

    # Analyze each file type
    for file_type in file_types:
        # Find all files belonging to this type and put them into an array
        matching_files = [
            path for path in csv_files
            if path.name.endswith(file_type)
        ]

        # Skip if this file type does not exist
        if not matching_files:
            continue

        # Get column names
        columns = get_columns(name, file_type)

        # If it gets no column, then skip
        if columns is None:
            continue

        # Statistics
        total_rows = 0
        total_missing = 0
        missing_columns = set()

        # Analyze every source file
        for path in matching_files:
            try:
                # Read the file in chunks (100000 rows everytime)
                for chunk in pd.read_csv(
                    path,
                    header=None,
                    names=columns,
                    chunksize=100000
                ):

                    # Count rows
                    total_rows += len(chunk)

                    # Count missing values
                    missing = chunk.isnull().sum()
                    total_missing += int(missing.sum())

                    # Record columns with missing values
                    for column in columns:
                        if missing[column] > 0:
                            missing_columns.add(column)

            # Display error message if any error happens
            except Exception as e:
                print(f"Error analyzing: {path.name}")
                print(f"Reason: {e}")

        # Display results
        print(f"\nFile type: {file_type}")
        print(f"Files: {len(matching_files)}")
        print(f"Rows: {total_rows}")
        print(f"Columns: {len(columns)}")
        print(f"Column names: {', '.join(columns)}")
        print(f"Missing values: {total_missing}")

        # Show all the columns that contain missing value
        if missing_columns:
            print(
                f"Columns with missing values: "
                f"{', '.join(missing_columns)}"
            )
        else:
            print("Columns with missing values: None")

# Main
if __name__ == "__main__":
    # File Title
    print("\nRansomware Dataset Analysis\n")

    # Analyze RanSAP
    analyze_dataset("RanSAP", RANSAP_DIR)

    print("\n" + "-" * 50 + "\n")

    # Analyze RanSMAP
    analyze_dataset("RanSMAP", RANSMAP_DIR)

    print("\n" + "-" * 50 + "\n")

    # Analyze Begin
    analyze_dataset("Begin", BEGIN_DIR)

    # Closure
    print("\nAnalysis complete")
