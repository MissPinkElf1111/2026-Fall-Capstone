# Ransomware Detection Using Storage and Memory Access Patterns

A Fall 2026 capstone project that investigates ransomware detection through storage and memory access behavior.

The Python pipeline processes CSV access traces, extracts behavioral features, learns normal behavior from benign executions, and evaluates anomaly scores against labeled benign and ransomware executions.

The implementation includes behavioral autoencoders, temporal smoothing, combined storage–memory scoring (`DTBAD_combined`), and Isolation Forest baselines.

## Project Objectives

- Prepare storage and memory traces for behavioral analysis.
- Extract activity, size, address, throughput, and entropy features.
- Learn normal behavior from benign executions.
- Compare storage-only, memory-only, and combined detection methods.
- Evaluate detection using execution-level metrics and confusion matrices.

## Methodology

1. **Inspect and analyze data:** Locate source CSV files and examine their schemas, row counts, and missing values.
2. **Combine and clean data:** Combine files by dataset and operation, preserve source identifiers, remove unused placeholder fields, and reject invalid records.
3. **Extract features:** Aggregate activity into **0.1-second windows**, producing **26 storage features** and **60 memory features** across six operation-specific CSV files.
4. **Split by execution:** Assign executions to approximately **70% training, 15% validation, and 15% testing**, stratified by binary label. Shared assignments keep each execution in one partition across operations and modalities.
5. **Train models:** Train separate storage and memory autoencoders and Isolation Forest models using **benign training windows only**. Fit feature scaling on training data and use benign validation data for early stopping and novelty calibration.
6. **Score behavior:** Calculate autoencoder reconstruction errors and Isolation Forest anomaly scores. Apply a five-window rolling mean to reconstruction errors within each execution.
7. **Combine modalities:** Normalize temporal scores using benign validation statistics and combine matched storage and memory execution summaries with equal weights.
8. **Evaluate detection:** Set each detection threshold at the **95th percentile of benign validation execution scores**, then evaluate held-out test executions. Scores above the threshold are classified as ransomware.

Window-level novelty flags from Step 08 and execution-level detection decisions from Step 10 use separate calibrations. Novelty rates alone do not represent classification accuracy.

## Datasets

The pipeline expects three dataset folders in the repository root.

**Raw datasets are not included in this repository and must be obtained separately.**

| Folder | Class | Label | Supported Traces |
| --- | --- | --- | --- |
| `RanSAP/` | Ransomware | `1` | Storage read and write |
| `RanSMAP/` | Ransomware | `1` | Storage read/write and memory read/write/readwrite/execute |
| `Benign/` | Benign | `0` | Storage read/write and memory read/write/readwrite/execute |

### Input File Format

Source files are discovered recursively and must be headerless CSV files matching the schemas defined in `02_analyze_data.py` and `03_combine_data.py`.

| Filename Suffix | Operation |
| --- | --- |
| `ata_read.csv` | Storage read |
| `ata_write.csv` | Storage write |
| `mem_read.csv` | Memory read |
| `mem_write.csv` | Memory write |
| `mem_readwrite.csv` | Memory read and write |
| `mem_exec.csv` | Memory execute |

Depending on the dataset and operation, fields include:

- UNIX timestamps in seconds and nanoseconds.
- Storage logical block addresses (LBA).
- Guest physical memory addresses (GPA).
- Access sizes.
- Entropy values.
- Operation-specific type or placeholder fields.

Schemas differ between datasets and operations. Preserve the original column order.

### Execution Naming

Preserve execution naming when organizing source files.

For example:

- `sample01_ata_read.csv`
- `sample01_ata_write.csv`
- `sample01_mem_read.csv`
- `sample01_mem_write.csv`

These filenames map to execution ID `sample01`.

Files named only by operation, such as `ata_read.csv`, use their relative parent directory as the execution ID.

Execution IDs must identify distinct runs within each dataset and match across storage and memory traces for fusion.

Combined scoring requires paired storage and memory executions from `RanSMAP` or `Benign`. `RanSAP` is evaluated through the storage models only.

## Requirements

- Python 3.10 or newer is a suggested starting environment; the repository does not specify a tested Python version.
- Git.
- The following Python packages:
  - NumPy
  - pandas
  - scikit-learn
  - PyTorch
  - joblib
  - Matplotlib
- Sufficient disk space for raw traces and generated outputs.
- Sufficient RAM for feature tables and model training.

PyTorch automatically uses CUDA when available and otherwise runs on the CPU.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MissPinkElf1111/2026-Fall-Capstone.git
cd 2026-Fall-Capstone
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn torch joblib matplotlib
```

The repository currently has no dependency lockfile or pinned package versions. Record the package versions used for any reported experiment.

### 5. Add the Datasets

Place the raw dataset folders in the repository root:

```text
2026-Fall-Capstone/
├── RanSAP/
├── RanSMAP/
├── Benign/
├── 01_inspect_data.py
├── 02_analyze_data.py
├── 03_combine_data.py
├── 04_prepare_data.py
├── 05_feature_engineering.py
├── 06_split_data.py
├── 07_train_models.py
├── 08_evaluate_models.py
├── 09_analyze_evalutaion.py
├── 10_evaluate_detection.py
└── README.md
```

## Running the Pipeline

Run the scripts from the repository root in the following order:

```bash
python 01_inspect_data.py
python 02_analyze_data.py
python 03_combine_data.py
python 04_prepare_data.py
python 05_feature_engineering.py
python 06_split_data.py
python 07_train_models.py
python 08_evaluate_models.py
python 09_analyze_evalutaion.py
python 10_evaluate_detection.py
```

**Note:** Step 09 is named `09_analyze_evalutaion.py` in the repository. Use that exact spelling.

Paths are relative to the current working directory. Configuration values are defined as constants inside the scripts.

### Pipeline Stages

| Script | Purpose | Main Output |
| --- | --- | --- |
| `01_inspect_data.py` | Locate datasets and count storage and memory files | Console inspection summary |
| `02_analyze_data.py` | Examine schemas, row counts, and missing values | Console analysis summary |
| `03_combine_data.py` | Combine traces by dataset and operation while preserving source metadata | `results/` |
| `04_prepare_data.py` | Clean combined records and remove unused fields | `prepared/` |
| `05_feature_engineering.py` | Compute time-window behavioral features | `features/` |
| `06_split_data.py` | Create shared execution-level train, validation, and test partitions | `splits/` |
| `07_train_models.py` | Train benign-only autoencoders and Isolation Forest baselines | `models/` |
| `08_evaluate_models.py` | Score windows, summarize executions, and combine modalities | `evaluation/` |
| `09_analyze_evalutaion.py` | Analyze score distributions, novelty rates, and data quality | `analysis/` |
| `10_evaluate_detection.py` | Apply detection thresholds and calculate classification metrics | `performance/` |

Steps 09 and 10 both use Step 08 outputs. Step 10 does not require Step 09.

Output directories are created automatically. Rerunning stages can overwrite generated files, so retain separate copies when comparing experiments.

Steps 09 and 10 display figures. For a headless environment, set `MPLBACKEND=Agg` before running them.

## Feature Engineering

Features are calculated within 0.1-second windows.

### Storage Features

The pipeline generates 26 storage features covering:

- Read and write operation counts.
- Bytes read and written.
- Read, write, and total throughput.
- Average, variance, and maximum request sizes.
- Storage address variance and range.
- Read and write ratios.
- Write entropy statistics.

### Memory Features

The pipeline generates 60 memory features describing read, write, readwrite, and execute behavior, including:

- Operation counts and ratios.
- Reported access sizes and byte totals.
- Access-size statistics.
- Memory address behavior.
- Entropy statistics where available.

Feature outputs retain dataset, execution, label, and window metadata for alignment and evaluation.

## Models and Scoring

### Behavioral Autoencoders

Separate PyTorch autoencoders are trained for storage and memory features.

Each autoencoder compresses an input feature vector and reconstructs it. Reconstruction error measures how unfamiliar a window is relative to learned benign behavior.

### Temporal Scoring

A rolling mean over five consecutive feature windows smooths reconstruction errors within each execution.

This adds temporal context to the behavioral scores.

### Combined Storage and Memory Scoring

Matched storage and memory execution summaries are combined using equal weights:

```text
Combined score =
    0.50 × normalized storage temporal score
  + 0.50 × normalized memory temporal score
```

Normalization parameters are estimated from benign validation data.

### Isolation Forest Baselines

Separate Isolation Forest models are trained on benign storage and memory windows to provide baseline anomaly scores.

## Evaluation

Detection thresholds are calibrated using benign validation execution scores. Test labels are used to evaluate predictions, not to select thresholds.

The decision rule is:

```text
score > threshold → ransomware (1)
score ≤ threshold → benign (0)
```

The evaluation reports:

- Accuracy.
- Precision.
- Recall.
- F1 score.
- Specificity.
- False positive rate.
- False negative rate.
- Balanced accuracy.
- ROC AUC.
- True negatives, false positives, false negatives, and true positives.

Report each method's test execution count alongside its metrics because storage, memory, and combined methods may cover different execution sets.

## Generated Outputs

### Data and Features

| Directory | Contents |
| --- | --- |
| `results/` | Combined traces with source metadata |
| `prepared/` | Cleaned traces |
| `features/` | Operation-specific feature tables |
| `splits/` | Partitioned feature files and execution split assignments |

### Models

The `models/` directory contains:

- `storage_autoencoder.pt`
- `memory_autoencoder.pt`
- `storage_isolation_forest.joblib`
- `memory_isolation_forest.joblib`

Model bundles preserve the preprocessing information and feature order needed for evaluation.

### Anomaly Evaluation

The `evaluation/` directory contains window scores, validation and test execution summaries, combined execution scores, and `evaluation_summary.csv`.

### Analysis

The `analysis/` directory contains data quality checks, score statistics, outlier candidates, a text report, and score distribution and novelty plots.

### Detection Performance

| File | Contents |
| --- | --- |
| `performance/classification_metrics.csv` | Classification metrics for each evaluated method |
| `performance/execution_predictions.csv` | Actual labels, predictions, scores, and thresholds |
| `performance/confusion_matrices.csv` | Confusion matrix counts |
| `performance/detection_thresholds.json` | Threshold values and benign validation execution counts |
| `performance/performance_report.txt` | Readable performance summary |
| `performance/10_confusion_matrices.png` | Confusion matrix figure |

No measured performance figures or trained models are committed in the current repository. Generate results by running the pipeline with the required datasets.

## Configuration

| Setting | Default | Script |
| --- | --- | --- |
| Feature window | 0.1 seconds | `05_feature_engineering.py` |
| Training / validation / test ratios | 70% / 15% / 15% | `06_split_data.py` |
| Random seed | 42 | `06_split_data.py`, `07_train_models.py` |
| Training batch size | 256 | `07_train_models.py` |
| Maximum training epochs | 100 | `07_train_models.py` |
| Early stopping patience | 10 | `07_train_models.py` |
| Learning rate | 0.001 | `07_train_models.py` |
| Isolation Forest estimators | 200 | `07_train_models.py` |
| Temporal rolling window | 5 windows | `08_evaluate_models.py` |
| Storage / memory fusion weights | 0.50 / 0.50 | `08_evaluate_models.py` |
| Novelty percentile | 95 | `07_train_models.py`, `08_evaluate_models.py` |
| Detection threshold percentile | 95 | `10_evaluate_detection.py` |

After changing upstream settings, rerun the affected stage and downstream stages to keep features, models, and evaluation outputs consistent.

## Limitations

- The implementation performs offline analysis of existing traces. Live monitoring, trace collection, and automated ransomware containment are not included.
- Detection depends on representative benign traces and consistent execution identifiers.
- Unusual benign activity can receive a high anomaly score.
- Small benign validation sets can make percentile thresholds unstable. Step 10 warns when fewer than 20 benign validation executions are available for a method.
- Combined detection is skipped in Step 10 when paired validation or test executions are unavailable.
- The full pipeline requires suitable storage and memory data for the individual models.
- Although raw-data stages use chunked reads, later stages load feature tables into memory.

## Academic Context

Developed as a Fall 2026 capstone project on ransomware detection using storage and memory access patterns.

Repository: [2026-Fall-Capstone](https://github.com/MissPinkElf1111/2026-Fall-Capstone)

## License

No license file is currently included in this repository.

Check dataset licensing and attribution requirements with the original data providers before redistributing data.