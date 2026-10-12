
"""
This python file is for inspecting the data from CSV files
"""

from pathlib import Path

# Configuration
RANSAP_DIR = Path("RanSAP")
RANSMAP_DIR = Path("RanSMAP")
BENIGN_DIR = Path("Begin")

# Find CSV Files
def find_csv(name, dataset):
    # Open the folder to find all the CSV files
    csv_files = list(dataset.rglob("*.csv"))

    # Display how many CSV files were found
    print(f"Dataset: {name}")
    print(f"CSV files: {len(csv_files)}")

    # Return CSV files
    return csv_files

# Inspect one dataset
def inspect_dataset(name, dataset):
    # Check whether the dataset exists and contains CSV files
    if not dataset.exists():
        print("Status: Not Found")
        print(f"Path: {dataset.resolve()}")
        return

    # Get the CSV files
    csv = find_csv(name, dataset)

    # Display the path of the CSV files
    print(f"Path: {dataset.resolve()}")

    # Display whether it found any CSV files
    if len(csv) == 0:
        print("Status: No CSV files found")
        return
    else:
        print("Status: Dataset files detected")

    # Create two different arrays to hold storage and memory CSV files
    storage = []
    memory = []

    # Separate storage and memory CSV files
    for file in csv:
        if "ata_" in file.name:
            storage.append(file)
        elif "mem_" in file.name:
            memory.append(file)

    # Display the numbers of files
    print(f"Storage files: {len(storage)}")
    print(f"Memory files: {len(memory)}")

# Main
if __name__ == "__main__":
    # File Title
    print("\nRansomware Dataset Inspector\n")

    # Inspect the CSV files
    inspect_dataset("RanSAP", RANSAP_DIR)
    print("\n" + "-" * 50 + "\n")
    inspect_dataset("RanSMAP", RANSMAP_DIR)
    print("\n" + "-" * 50 + "\n")
    inspect_dataset("Benign", BENIGN_DIR)
    print("\n")

    # Closure
    print("Dataset Inspection Complete")
