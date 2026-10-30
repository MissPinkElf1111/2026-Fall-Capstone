"""
Step 05A: Paper Feature Engineering

Extract the original 23 behavioral features described in
the RanSMAP research paper.

Storage features: 5
    - Read throughput
    - Write throughput
    - Read LBA variance
    - Write LBA variance
    - Average write entropy

Memory features: 18
    - Write entropy
    - Readwrite entropy
    - 12 memory page access counts
    - 4 GPA variances

This script is used for the ablation study comparing
the paper's 23 features with our expanded 86 features.

Input:
    prepared/

Output:
    features_paper/
        storage_paper_features.csv
        memory_paper_features.csv

The original Step 05 outputs remain unchanged.
"""

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


# Configuration
PREPARED_DIR = Path("prepared")
# Separate directory for the paper's 23 features
OUTPUT_DIR = Path("features_paper")
# Number of CSV rows processed at once to reduce memory usage
CHUNK_SIZE = 250_000

# Use the same 0.1-second window size
WINDOW_SIZE_NS = 100_000_000
WINDOW_SIZE_SECONDS = WINDOW_SIZE_NS / 1_000_000_000

# Metadata columns used to identify executions and time windows
KEY_COLUMNS = [
    "dataset",
    "execution_id",
    "label",
    "window_start_ns",
    "window_end_ns",
]

# Five storage features defined in the RanSMAP paper
STORAGE_FEATURES = [
    "read_throughput",
    "write_throughput",
    "read_lba_variance",
    "write_lba_variance",
    "write_entropy",
]

# Eighteen memory features defined in the RanSMAP paper
MEMORY_FEATURES = [
    "write_entropy",
    "readwrite_entropy",

    # Count accesses for each page type and operation
    *[
        f"{page}_{op}_count"
        for page in ("4kb", "2mb", "mmio")
        for op in ("read", "write", "readwrite", "execute")
    ],

    # GPA variance for each memory operation
    *[
        f"{op}_gpa_variance"
        for op in ("read", "write", "readwrite", "execute")
    ]
]


# Map memory access type codes to their page categories
PAGE_TYPES = {
    1: "4kb",
    2: "2mb",
    4: "mmio"
}

# Identify the operation and modality from each CSV filename
OP_SUFFIXES = {
    ("storage", "read"): "_ata_read.csv",
    ("storage", "write"): "_ata_write.csv",
    ("memory", "read"): "_mem_read.csv",
    ("memory", "write"): "_mem_write.csv",
    ("memory", "readwrite"): "_mem_readwrite.csv",
    ("memory", "execute"): "_mem_exec.csv"
}

# Input File Identification
# Determine the dataset, modality, and operation of a CSV file
def identify_file(path):
    name = path.name

    # Recognize the supported datasets
    dataset = next(
        (
            d for d in ("RanSAP", "RanSMAP", "Benign")
            if name.startswith(d + "_")
        ),
        None
    )

    # Ignore files that do not belong to supported datasets
    if dataset is None:
        return None

    # Identify the access operation from the filename
    for (modality, operation), suffix in OP_SUFFIXES.items():
        if name.endswith(suffix):

            # RanSAP contains storage access traces only
            if modality == "memory" and dataset == "RanSAP":
                raise ValueError("RanSAP is expected to contain storage only")

            return {
                "path": path,
                "dataset": dataset,
                "modality": modality,
                "operation": operation
            }

    return None


# Locate all supported prepared CSV files
def find_inputs():
    # Ensure Step 04 has produced the prepared directory
    if not PREPARED_DIR.is_dir():
        raise FileNotFoundError(f"Prepared data folder does not exist: {PREPARED_DIR}")

    # Collect supported CSV files in a consistent order
    inputs = [
        item
        for path in sorted(PREPARED_DIR.glob("*.csv"))
        if (item := identify_file(path)) is not None
    ]

    if not inputs:
        raise FileNotFoundError(f"No supported prepared CSV files in {PREPARED_DIR}")

    return inputs

# Execution Identification and Timestamp Processing
# Extract consistent execution IDs from source filenames
def execution_ids(series, item):
    suffix = OP_SUFFIXES[(item["modality"], item["operation"])]

    values = series.astype(str)
    mapping = {}

    for source in values.unique():
        source_path = Path(source)
        base = source_path.name

        # Remove the operation suffix to obtain the execution ID
        if base.endswith(suffix):
            execution_id = base[:-len(suffix)]

        # Handle source IDs that use directory paths
        elif base == suffix[1:]:
            execution_id = source_path.parent.as_posix()

        else:
            raise ValueError(f"Unexpected source_file_id: {source!r} in {item['path']}")

        mapping[source] = execution_id

    return values.map(mapping)


# Read large CSV files in smaller chunks
def read_chunks(item, columns):
    try:
        for chunk in pd.read_csv(
            item["path"],
            usecols=columns,
            chunksize=CHUNK_SIZE
        ):
            yield chunk

    except ValueError as exc:
        raise ValueError(f"Missing or invalid columns in {item['path']}: {exc}") from exc

# Convert timestamps to nanoseconds and obtain execution IDs
def time_and_ids(chunk, item):
    required = [
        "source_file_id",
        "UNIX time in sec",
        "UNIX time in ns"
    ]

    # Remove rows without essential timestamp information
    chunk = chunk.dropna(subset=required).copy()

    if chunk.empty:
        return (
            chunk,
            pd.Series(dtype="object"),
            pd.Series(dtype="int64")
        )

    # Convert UNIX seconds and nanoseconds into numeric values
    sec = pd.to_numeric(
        chunk["UNIX time in sec"],
        errors="raise"
    ).astype("int64")

    ns = pd.to_numeric(
        chunk["UNIX time in ns"],
        errors="raise"
    ).astype("int64")

    # Combine the two timestamp components
    timestamps = sec * 1_000_000_000 + ns

    return (
        chunk,
        execution_ids(chunk["source_file_id"], item),
        timestamps
    )

# Find the earliest access timestamp for each execution
def discover_starts(items):
    # Store the starting timestamp of each execution
    starts = {}

    for item in items:
        n = 0

        for raw in read_chunks(
            item,
            [
                "source_file_id",
                "UNIX time in sec",
                "UNIX time in ns"
            ]
        ):
            chunk, ids, timestamps = time_and_ids(raw, item)

            if chunk.empty:
                continue

            # Find the earliest timestamp in each execution
            minima = pd.DataFrame({
                "id": ids,
                "time": timestamps
            }).groupby("id")["time"].min()

            # Keep the earliest timestamp across operations
            for execution_id, start in minima.items():
                key = (item["dataset"], execution_id)

                starts[key] = min(starts.get(key, int(start)), int(start))

            n += len(chunk)

        print(f"\tStart times: {item['path'].name} ({n:,} events)", flush=True)

    return starts

# Statistical Calculations
# Calculate sample variance incrementally across CSV chunks
# This avoids loading all LBA or GPA values into memory
class RunningVariance:
    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, values):
        # Convert values to numeric and remove invalid entries
        arr = pd.to_numeric(
            values,
            errors="coerce"
        ).dropna().to_numpy(dtype=np.float64)

        if not len(arr):
            return

        # Calculate statistics for the current group
        n = len(arr)
        mean = float(arr.mean())
        m2 = float(np.sum((arr - mean) ** 2))

        # Combine the current group with previous observations
        delta = mean - self.mean
        total = self.n + n

        self.m2 += (m2 + delta * delta * self.n * n / total)

        self.mean += delta * n / total
        self.n = total

    # Return sample variance using N - 1
    def variance(self):
        return (
            self.m2 / (self.n - 1)
            if self.n > 1
            else 0.0
        )


# Calculate average entropy incrementally
class RunningMean:
    __slots__ = ("total", "n")

    def __init__(self):
        self.total = 0.0
        self.n = 0

    def add(self, values):
        # Ignore missing or nonnumeric entropy values
        arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)

        self.total += float(arr.sum())
        self.n += len(arr)

    # Return -1 when no valid entropy observations exist
    def average(self):
        return (
            self.total / self.n
            if self.n
            else -1.0
        )

# Required Columns for Each Access Operation
# Select only the raw columns needed for the 23 features
def required_columns(item):
    fields = [
        "source_file_id",
        "UNIX time in sec",
        "UNIX time in ns"
    ]

    if item["modality"] == "storage":
        # Storage features require addresses and request sizes
        fields += ["LBA", "size"]

        # Entropy is available for storage write operations
        if item["operation"] == "write":
            fields.append(
                "Entropy #1"
                if item["dataset"] == "RanSAP"
                else "entropy"
            )

    else:
        # Memory features require guest physical addresses and memory access page types
        fields += ["GPA", "type"]

        # Memory entropy is calculated for write/readwrite
        if item["operation"] in ("write", "readwrite"):
            fields.append("entropy")

    return fields

# Feature Aggregation
# Aggregate storage and memory events by execution and time window
def aggregate(items, starts):
    # Store intermediate statistics for each execution/window
    states = defaultdict(dict)

    # Track the last observed window of each execution
    last_window = defaultdict(lambda: -1)

    for item in items:
        modality = item["modality"]
        op = item["operation"]
        dataset = item["dataset"]

        processed = 0

        for raw in read_chunks(item, required_columns(item)):
            chunk, ids, timestamps = time_and_ids(raw, item)

            if chunk.empty:
                continue

            # Convert absolute timestamps to relative execution time
            begin = ids.map(lambda eid: starts[(dataset, eid)]).astype("int64")

            relative = timestamps - begin

            # Ignore invalid negative relative timestamps
            valid = relative >= 0

            if not valid.any():
                continue

            chunk = chunk.loc[valid].copy()

            chunk["execution_id"] = ids.loc[valid]

            # Assign each access to a 0.1-second window
            chunk["window_index"] = (relative.loc[valid] // WINDOW_SIZE_NS).astype("int64")

            # Process each execution/window separately
            for (eid, window), group in chunk.groupby(
                ["execution_id", "window_index"],
                sort=False
            ):
                window = int(window)
                key = (dataset, eid, window)

                state = states[key]

                last_window[(dataset, eid)] = max(
                    last_window[(dataset, eid)],
                    window
                )

                # Storage Feature Aggregation
                if modality == "storage":
                    # Sum bytes read or written within the window
                    bytes_key = f"{op}_bytes"

                    size = pd.to_numeric(group["size"], errors="coerce")

                    state[bytes_key] = (state.get(bytes_key, 0.0) + float(size.sum()))

                    # Accumulate LBA statistics for variance
                    variance_key = f"{op}_lba"

                    state.setdefault(variance_key, RunningVariance()).add(group["LBA"])

                    # Calculate entropy for storage writes only
                    if op == "write":
                        entropy_column = (
                            "Entropy #1"
                            if dataset == "RanSAP"
                            else "entropy"
                        )

                        state.setdefault("write_entropy", RunningMean()).add(group[entropy_column])

                # Memory Feature Aggregation
                else:
                    # Accumulate GPA statistics for each operation
                    variance_key = f"{op}_gpa"

                    state.setdefault(variance_key, RunningVariance()).add(group["GPA"])

                    # Identify memory page types
                    page_types = pd.to_numeric(group["type"], errors="coerce")

                    # Count accesses to 4 KB, 2 MB, and MMIO pages
                    for code, page in PAGE_TYPES.items():
                        counter = f"{page}_{op}_count"

                        state[counter] = (state.get(counter, 0) + int((page_types == code).sum()))

                    # Calculate entropy for write/readwrite only
                    if op in ("write", "readwrite"):
                        state.setdefault(
                            f"{op}_entropy",
                            RunningMean()
                        ).add(group["entropy"])

            processed += len(chunk)

        print(f"\tAggregated: {item['path'].name} ({processed:,} events)", flush=True)

    return states, last_window

# Feature Vector Generation
# Convert aggregated statistics into fixed-length feature vectors
def build_features(modality, starts, states, last_window):
    # Select the appropriate feature set
    features = (
        STORAGE_FEATURES
        if modality == "storage"
        else MEMORY_FEATURES
    )

    rows = []

    for dataset, eid in sorted(starts):
        # Generate every window up to the last observed access
        # Windows without activity receive default feature values
        for win in range(last_window[(dataset, eid)] + 1):
            state = states.get((dataset, eid, win), {})

            # Add execution metadata and window boundaries
            row = {
                "dataset": dataset,
                "execution_id": eid,

                # Benign = 0, ransomware = 1
                "label": 0 if dataset == "Benign" else 1,

                "window_start_ns": win * WINDOW_SIZE_NS,
                "window_end_ns": (
                    (win + 1) * WINDOW_SIZE_NS
                )
            }

            # Generate Five Storage Features
            if modality == "storage":
                row.update({
                    # Throughput = total bytes / window duration
                    "read_throughput": (state.get("read_bytes", 0.0) / WINDOW_SIZE_SECONDS),
                    "write_throughput": (state.get("write_bytes", 0.0) / WINDOW_SIZE_SECONDS),

                    # Sample variance of read/write LBAs
                    "read_lba_variance": (
                        state["read_lba"].variance()
                        if "read_lba" in state
                        else 0.0
                    ),
                    "write_lba_variance": (
                        state["write_lba"].variance()
                        if "write_lba" in state
                        else 0.0
                    ),

                    # Average entropy of written sectors
                    "write_entropy": (
                        state["write_entropy"].average()
                        if "write_entropy" in state
                        else -1.0
                    )
                })

            # Generate Eighteen Memory Features
            else:
                # Two memory entropy features
                row["write_entropy"] = (
                    state["write_entropy"].average()
                    if "write_entropy" in state
                    else -1.0
                )
                row["readwrite_entropy"] = (
                    state["readwrite_entropy"].average()
                    if "readwrite_entropy" in state
                    else -1.0
                )

                # Twelve page-type access count features
                for page in ("4kb", "2mb", "mmio"):
                    for op in ("read", "write", "readwrite", "execute"):
                        name = f"{page}_{op}_count"
                        row[name] = state.get(name, 0)

                # Four GPA variance features
                for op in ("read", "write", "readwrite", "execute"):
                    name = f"{op}_gpa_variance"
                    source = f"{op}_gpa"

                    row[name] = (
                        state[source].variance()
                        if source in state
                        else 0.0
                    )

            rows.append(row)

    # Create the final feature DataFrame
    result = pd.DataFrame(rows, columns=KEY_COLUMNS + features)

    # Validate Generated Features
    # Ensure the output contains at least one window
    if result.empty:
        raise RuntimeError(f"No {modality} features generated")

    # Check for duplicate execution/window combinations
    if result.duplicated(KEY_COLUMNS[:2] + KEY_COLUMNS[3:]).any():
        raise RuntimeError(f"Duplicate {modality} execution windows")

    # Ensure no missing values exist
    if result.isna().any().any():
        raise RuntimeError(f"Missing {modality} feature values")

    # Ensure every feature value is finite
    if not np.isfinite(result[features].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Non-finite {modality} feature values")

    return result

# Main Execution
if __name__ == "__main__":
    # Validate configuration values
    if WINDOW_SIZE_NS <= 0 or CHUNK_SIZE <= 0:
        raise ValueError("WINDOW_SIZE_NS and CHUNK_SIZE must be positive")

    # Verify that exactly 23 features are defined
    if (
        len(STORAGE_FEATURES) != 5
        or len(MEMORY_FEATURES) != 18
    ):
        raise RuntimeError("Expected 5 storage and 18 memory features")

    # Discover the prepared input files
    inputs = find_inputs()

    # Create the output directory if necessary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # File title
    print("\nRanSMAP Paper's 23 Feature Definitions\n")

    # Process storage and memory separately
    for modality in ("storage", "memory"):
        items = [
            item for item in inputs
            if item["modality"] == modality
        ]

        if not items:
            print(f"No {modality} files; skipping", flush=True)
            continue

        print(f"Processing {modality}:", flush=True)

        # Identify the starting timestamp of each execution
        starts = discover_starts(items)

        # Aggregate raw access events into window statistics
        states, last_window = aggregate(items, starts)

        # Generate the paper's feature vectors
        result = build_features(modality, starts, states, last_window)

        # Save the resulting CSV separately from Step 05
        path = (OUTPUT_DIR / f"{modality}_paper_features.csv")

        result.to_csv(path, index=False)

        # Display the number of generated windows and features
        print(
            f"Saved {path}: "
            f"{len(result):,} windows, "
            f"{len(result.columns) - len(KEY_COLUMNS)} features, "
            f"{len(starts)} executions\n",
            flush=True
        )

    # Closure
    print(
        "23 Feature Engineering Complete\n"
        "Original outputs are unchanged."
    )
