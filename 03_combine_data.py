"""
This python file is for combining CSV files by file type
"""

from pathlib import Path

# Configuration
RANSAP_DIR = Path("RanSAP")
RANSMAP_DIR = Path("RanSMAP")

# Create a folder to store combined data
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# Get column names based on the dataset according to README
def get_columns(name, file_name):
    # RanSAP
    if name == "RanSAP":
        # read file
        if file_name.endswith("ata_read.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size"
            ]
        # write file
        elif file_name.endswith("ata_write.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "Entropy #1",
                "Entropy #2"
            ]

    # RanSMAP
    elif name == "RanSMAP":
        # read file of storage access
        if file_name.endswith("ata_read.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "-1",
                "0"
            ]
        # write file of storage access
        elif file_name.endswith("ata_write.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "LBA",
                "size",
                "entropy",
                "0"
            ]
        # read file of memory access
        elif file_name.endswith("mem_read.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "0",
                "-1",
                "type"
            ]
        # write file of memory access
        elif file_name.endswith("mem_write.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "size",
                "entropy",
                "type"
            ]
        # read & write file of memory access
        elif file_name.endswith("mem_readwrite.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "size",
                "entropy",
                "type"
            ]
        # execute file of memory access
        elif file_name.endswith("mem_exec.csv"):
            return [
                "UNIX time in sec",
                "UNIX time in ns",
                "GPA",
                "0",
                "-1",
                "type"
            ]

    # If a file does not belong to any of the above, then skip
    return None


# Combine one dataset
def combine_dataset(name, dataset):
    # Check whether the dataset exists
    if not dataset.exists():
        print(f"{name}: Dataset not found.")
        return

    # Find all CSV files
    csv_files = list(dataset.rglob("*.csv"))

    # File types to combine
    file_types = [
        "ata_read.csv",
        "ata_write.csv",
        "mem_read.csv",
        "mem_write.csv",
        "mem_readwrite.csv",
        "mem_exec.csv"
    ]

    # Process each file type
    for file_type in file_types:
        # Find all files belonging to this type and put them into an array
        matching_files = [
            path for path in csv_files
            if path.name.endswith(file_type)
        ]

        # Skip if no matching files exist
        if not matching_files:
            continue

        # Get column names
        columns = get_columns(name, file_type)

        # If the file type has no defined columns, then skip
        if columns is None:
            continue

        # Create output filename
        output = RESULTS_DIR / (f"{name}_{Path(file_type).stem}.csv")

        # Combine the same type files into one
        try:
            # Open the output file with write permission
            with open(
                output,
                "w",
                encoding="utf-8",
                newline=""
            ) as destination:
                # Write metadata columns and original columns
                destination.write(
                    "source_file_id,source_row,"
                    + ",".join(columns)
                    + "\n"
                )

                # Copy every source file
                for path in matching_files:
                    # Store the relative path as the source file ID
                    source_file_id = path.relative_to(
                        dataset
                    ).as_posix()

                    # Open the input file with write permission
                    with open(
                        path,
                        "r",
                        encoding="utf-8",
                        newline=""
                    ) as source:
                        # Track the original row number
                        for row_number, line in enumerate(
                            source,
                            start=1
                        ):
                            # Remove the existing newline
                            line = line.rstrip("\r\n")

                            # Skip empty lines
                            if not line:
                                continue

                            # Add source information to the row
                            destination.write(
                                f"{source_file_id},"
                                f"{row_number},"
                                f"{line}\n"
                            )

            # Display result
            print(f"{name}: {file_type} -> {output.name} ({len(matching_files)} files)")

        # Display error message when any error occurs with reason
        except Exception as e:
            print(f"{name}: Error combining {file_type}")
            print(f"Reason: {e}")

    # Closure
    print(f"{name}: Data combination complete")


# Main
if __name__ == "__main__":
    # File Title
    print("\nRansomware Dataset Combination\n")

    # Combine RanSAP
    combine_dataset("RanSAP", RANSAP_DIR)

    print("\n" + "-" * 50 + "\n")

    # Combine RanSMAP
    combine_dataset("RanSMAP", RANSMAP_DIR)