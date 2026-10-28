# Ransomware Detection Using Storage and Memory Access Patterns

## Project Overview

This Fall 2026 capstone project investigates ransomware detection using behavioral features extracted from storage and memory access traces.

The implementation processes traces from the RanSAP, RanSMAP, and Benign datasets. It converts raw access records into time-window features, trains models on benign behavior, and evaluates whether anomaly scores can distinguish ransomware executions from benign executions.

The project implements:

- Separate behavioral autoencoders for storage and memory activity.
- Temporal scoring based on rolling autoencoder reconstruction errors.
- Combined storage and memory execution scores, identified as `DTBAD_combined`.
- Separate Isolation Forest models as anomaly detection baselines.
- Execution-level detection thresholds calibrated using benign validation data.
- Held-out testing with classification metrics and confusion matrices.

The repository contains an offline experimental pipeline. Trace collection and live ransomware response are outside the current implementation.

## Research Objectives

The project examines:

1. Whether storage access patterns distinguish ransomware from benign activity.
2. Whether memory access patterns provide useful detection signals.
3. Whether combining storage and memory scores improves detection compared with either modality alone.
4. How autoencoder-based behavioral scores compare with Isolation Forest baselines.
5. How the trained models perform on executions excluded from training and validation.

## Dataset Organization

The scripts expect the following folders in the repository root:

| Dataset Folder | Ground-Truth Class | Label | Access Traces Used |
| --- | --- | --- | --- |
| `RanSAP/` | Ransomware | `1` | Storage read and write |
| `RanSMAP/` | Ransomware | `1` | Storage and memory |
| `Benign/` | Benign | `0` | Storage and memory |

Raw datasets are not included in the repository.

### Supported Trace Files

| Filename Suffix | Trace Type |
| --- | --- |
| `ata_read.csv` | Storage read |
| `ata_write.csv` | Storage write |
| `mem_read.csv` | Memory read |
| `mem_write.csv` | Memory write |
| `mem_readwrite.csv` | Memory read and write |
| `mem_exec.csv` | Memory execute |

The source CSV files are read without headers. Column definitions are assigned according to the dataset and trace type.

The expected schemas are implemented in:

- `02_analyze_data.py`
- `03_combine_data.py`
- `04_prepare_data.py`

Storage records include timestamp components, logical block addresses, access sizes, and write entropy where available.

Memory records include timestamp components, guest physical addresses, and operation-dependent size, entropy, and type fields.

RanSAP and RanSMAP do not use identical storage schemas. The preparation stage handles these differences before feature extraction.

### Execution Identification

The pipeline preserves the relationship between source files and their executions.

During combination, each record receives:

- `source_file_id`: the source file's path relative to its dataset folder.
- `source_row`: the record's original row number.

During feature engineering, operation suffixes are removed to derive execution IDs.

For example:

```text
execution01_ata_read.csv
execution01_ata_write.csv
execution01_mem_read.csv
execution01_mem_write.csv
```

These files map to execution ID `execution01`.

For files named only by operation, such as `ata_read.csv`, the relative parent directory supplies the execution ID.

Execution names must distinguish runs within a dataset and match across modalities when those traces belong to the same execution.

## Repository Contents

| File | Responsibility |
| --- | --- |
| `01_inspect_data.py` | Check dataset availability and count CSV, storage, and memory files |
| `02_analyze_data.py` | Analyze source schemas, row counts, and missing values |
| `03_combine_data.py` | Combine source files by dataset and operation while preserving source metadata |
| `04_prepare_data.py` | Clean combined records and remove unused placeholder columns |
| `05_feature_engineering.py` | Extract storage and memory features in 0.1-second windows |
| `06_split_data.py` | Create shared execution-level training, validation, and test assignments |
| `07_train_models.py` | Train benign-only autoencoders and Isolation Forest models |
| `08_evaluate_models.py` | Calculate window scores, temporal scores, execution summaries, and combined scores |
| `09_analyze_evalutaion.py` | Analyze score distributions, novelty rates, outliers, and evaluation data quality |
| `10_evaluate_detection.py` | Calibrate execution thresholds and calculate held-out detection performance |

The spelling `09_analyze_evalutaion.py` matches the actual filename.

## Experimental Pipeline

### 1. Dataset Inspection

`01_inspect_data.py` searches recursively for CSV files in the three dataset folders.

It reports:

- Whether each dataset directory exists.
- The number of CSV files.
- The number of storage files.
- The number of memory files.

### 2. Source Data Analysis

`02_analyze_data.py` assigns the expected schema to each supported trace type and reads records in chunks of 100,000 rows.

It reports:

- File counts by operation.
- Total row counts.
- Expected column names.
- Missing-value counts.
- Columns containing missing values.

### 3. Trace Combination

`03_combine_data.py` combines files separately for each dataset and operation.

For example, storage read files from RanSAP are written to:

```text
results/RanSAP_ata_read.csv
```

The output contains source metadata followed by the original trace fields. This preserves execution information for downstream feature engineering.

### 4. Data Preparation

`04_prepare_data.py` reads the combined files in chunks and:

- Validates source file identifiers.
- Removes operation-specific placeholder fields.
- Removes rows with missing values in retained columns.
- Checks numeric fields and removes records that cannot be converted.
- Writes cleaned files to `prepared/`.

### 5. Feature Engineering

`05_feature_engineering.py` aggregates activity into windows of:

```text
100,000,000 nanoseconds = 0.1 seconds
```

The implementation defines:

- **26 storage features**
- **60 memory features**

Identification and alignment columns are retained separately from the model inputs:

```text
dataset
execution_id
label
window_start_ns
window_end_ns
```

#### Storage Features

The storage feature vector describes:

- Read, write, and total operation counts.
- Bytes read, bytes written, and total bytes.
- Read, write, and total throughput.
- Average, variance, and maximum read and write sizes.
- Read and write logical block address variance and range.
- Read ratio, write ratio, and read/write ratio.
- Mean, variance, minimum, and maximum write entropy.

#### Memory Features

The memory feature vector covers read, write, readwrite, and execute activity.

It includes operation counts and ratios, reported byte totals, access-size statistics, address behavior, and entropy statistics where supported by the trace.

#### Feature Files

The features are distributed across six files:

```text
features/storage_read_features.csv
features/storage_write_features.csv
features/memory_read_features.csv
features/memory_write_features.csv
features/memory_readwrite_features.csv
features/memory_execute_features.csv
```

Aligned window keys allow the operation-specific files to be merged into storage and memory model inputs.

### 6. Execution-Level Data Splitting

`06_split_data.py` uses the following target proportions:

| Partition | Target Proportion |
| --- | --- |
| Training | 70% |
| Validation | 15% |
| Test | 15% |

The random seed is `42`.

Splitting occurs at the execution level, rather than by randomly distributing individual windows.

A global assignment is created using:

```text
(dataset, execution_id)
```

The assignment is stratified by binary label and applied across the feature files. Consequently, windows from one execution remain in the same partition across operations and modalities.

The assignments are saved to:

```text
splits/execution_split_assignments.csv
```

This procedure prevents windows from the same identified execution from appearing in both training and testing.

### 7. Benign-Only Model Training

`07_train_models.py` trains separate storage and memory models.

Although the split files contain both classes, the training script filters the training and validation data to:

```text
label == 0
```

Ransomware windows are excluded from model fitting.

#### Feature Scaling

A `StandardScaler` is fitted on benign training features.

The fitted scaler transforms validation features and is preserved with the model information for later evaluation.

#### Behavioral Autoencoders

Each PyTorch autoencoder uses the following structure:

```text
Input → Hidden → Latent → Hidden → Reconstruction
```

ReLU activation is applied after the first encoder layer and the first decoder layer.

The architecture depends on the input feature count:

| Model | Input Features | Hidden Units | Latent Units |
| --- | --- | --- | --- |
| Storage autoencoder | 26 | 13 | 6 |
| Memory autoencoder | 60 | 30 | 15 |

Training settings are:

| Setting | Value |
| --- | --- |
| Batch size | 256 |
| Maximum epochs | 100 |
| Learning rate | 0.001 |
| Early stopping patience | 10 |
| Minimum improvement | 0.00001 |
| Random seed | 42 |

The models learn to reconstruct benign feature vectors. Reconstruction error is subsequently used as an anomaly score.

#### Isolation Forest Baselines

Separate Isolation Forest models are trained on the scaled benign storage and memory features.

| Setting | Value |
| --- | --- |
| Number of estimators | 200 |
| Contamination | `auto` |
| Random seed | 42 |

The evaluation uses negative `score_samples` values so that higher scores indicate greater unfamiliarity.

#### Saved Models

```text
models/storage_autoencoder.pt
models/memory_autoencoder.pt
models/storage_isolation_forest.joblib
models/memory_isolation_forest.joblib
```

The saved bundles include model information, feature order, preprocessing information, and novelty thresholds.

### 8. Behavioral and Temporal Evaluation

`08_evaluate_models.py` loads the trained models and partitioned features.

For each modality, it calculates:

- Autoencoder reconstruction error.
- Isolation Forest anomaly score.
- Temporal reconstruction score.
- Window-level novelty indicators.
- Execution-level score summaries.

The temporal score is a rolling mean of reconstruction errors over five windows, calculated separately for each execution.

At the default feature window size, five full windows represent 0.5 seconds. Initial windows use the available shorter history.

Window novelty thresholds are calibrated at the 95th percentile of benign validation window scores.

### 9. Combined Storage and Memory Scoring

Step 08 combines storage and memory summaries for matched executions from:

- `RanSMAP`
- `Benign`

RanSAP is excluded from combined scoring because its supported inputs are storage traces.

Temporal scores are normalized using the median and interquartile range of benign validation scores. The implementation includes fallback scales when the interquartile range is zero.

The combined score is:

```text
combined_score =
    0.50 × normalized_temporal_score_storage
  + 0.50 × normalized_temporal_score_memory
```

Storage and memory summaries are matched using dataset and execution ID.

### 10. Evaluation Analysis

`09_analyze_evalutaion.py` examines the Step 08 outputs.

It produces:

- Evaluation data quality checks.
- Score distribution statistics.
- High-score outlier candidates.
- Score distribution plots.
- Novelty-rate plots.
- A readable analysis report.

High anomaly scores and novelty rates describe model behavior. Classification performance is calculated separately in Step 10.

### 11. Held-Out Detection Testing

`10_evaluate_detection.py` evaluates seven methods when the required inputs are available:

| Method | Execution-Level Score |
| --- | --- |
| `storage_autoencoder` | Mean storage reconstruction error |
| `storage_temporal` | Mean storage temporal score |
| `storage_isolation_forest` | Mean storage Isolation Forest score |
| `memory_autoencoder` | Mean memory reconstruction error |
| `memory_temporal` | Mean memory temporal score |
| `memory_isolation_forest` | Mean memory Isolation Forest score |
| `DTBAD_combined` | Combined normalized temporal score |

For each method, the threshold is the 95th percentile of its benign validation execution scores.

The decision rule is:

```text
score > threshold → ransomware (1)
score ≤ threshold → benign (0)
```

Test labels are used to measure detection performance after threshold calibration.

The script checks for duplicate execution IDs, invalid labels, nonfinite scores, inconsistent modality labels, and overlap between validation and test executions.

## Training Results

Numerical training results were not included in the repository reviewed for this README.

The training script prints the number of benign training and validation windows, selected epoch, and validation reconstruction loss.

| Model | Benign Training Windows | Benign Validation Windows | Best Epoch | Best Validation MSE |
| --- | --- | --- | --- | --- |
| Storage autoencoder | Pending | Pending | Pending | Pending |
| Memory autoencoder | Pending | Pending | Pending | Pending |

Populate this table from the output of `07_train_models.py`.

## Held-Out Test Results

Numerical test results were not included in the repository reviewed for this README.

After running Step 10, populate the following table from:

```text
performance/classification_metrics.csv
```

| Method | Test Executions | Accuracy | Precision | Recall | F1 | False Positive Rate | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Storage autoencoder | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Storage temporal | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Storage Isolation Forest | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Memory autoencoder | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Memory temporal | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Memory Isolation Forest | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| DTBAD combined | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

The evaluation also reports specificity, false negative rate, balanced accuracy, and confusion matrix counts.

### Scope of Generalisation Testing

The implemented split measures performance on held-out executions from the available datasets.

It does not explicitly implement ransomware-family holdout, cross-dataset testing, or chronological holdout. Results therefore support conclusions about held-out execution performance within this experimental setup.

Storage, memory, and combined methods can cover different execution sets. Their test counts and class distributions should accompany any performance comparison.

## Installation

Clone the repository:

```bash
git clone https://github.com/MissPinkElf1111/2026-Fall-Capstone.git
cd 2026-Fall-Capstone
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the imported dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn torch joblib matplotlib
```

The repository does not currently pin dependency versions or declare a tested Python version.

PyTorch uses CUDA when available and otherwise runs on the CPU.

## Running the Experiment

Place `RanSAP/`, `RanSMAP/`, and `Benign/` in the repository root.

Run the scripts from that directory:

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

Steps 09 and 10 both consume Step 08 outputs. Step 10 does not depend on Step 09.

The scripts create output directories automatically. Rerunning a stage can overwrite its previous outputs.

For headless plotting, set the environment variable `MPLBACKEND` to `Agg`.

## Output Files

| Directory | Contents |
| --- | --- |
| `results/` | Combined traces with source identifiers and row numbers |
| `prepared/` | Cleaned trace files |
| `features/` | Six operation-specific feature files |
| `splits/` | Partitioned feature files and shared execution assignments |
| `models/` | Storage and memory autoencoders and Isolation Forest bundles |
| `evaluation/` | Window scores, validation/test execution scores, combined scores, and evaluation summary |
| `analysis/` | Data quality checks, score statistics, outlier candidates, reports, and plots |
| `performance/` | Classification metrics, predictions, thresholds, and confusion matrices |

The final detection outputs are:

| File | Purpose |
| --- | --- |
| `classification_metrics.csv` | Metrics for each evaluated detection method |
| `execution_predictions.csv` | Actual and predicted labels, scores, and thresholds by execution |
| `confusion_matrices.csv` | True negative, false positive, false negative, and true positive counts |
| `detection_thresholds.json` | Calibrated thresholds and benign validation execution counts |
| `performance_report.txt` | Readable detection performance report |
| `10_confusion_matrices.png` | Confusion matrix plots |

## Experimental Considerations

- Benign training and validation data are required for both storage and memory models.
- Combined detection requires matching storage and memory execution IDs.
- Step 10 skips combined detection if paired validation or test executions are unavailable.
- Step 10 warns when a method has fewer than 20 benign validation executions because the percentile threshold may be unstable.
- Raw-data processing uses chunked reads, but later stages load feature tables into memory.
- Changes to feature windows, split assignments, or preprocessing require rerunning the affected downstream stages.
- Detection performance depends on how well the benign data represents legitimate activity.

## Academic Context

Fall 2026 Capstone Project

**Project:** Ransomware Detection Using Storage and Memory Access Patterns

**Repository:** [MissPinkElf1111/2026-Fall-Capstone](https://github.com/MissPinkElf1111/2026-Fall-Capstone)