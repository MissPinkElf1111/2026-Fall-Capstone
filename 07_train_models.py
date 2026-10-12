"""
Train the behavioral models for our proposed DTBAD architecture.

Component A:
    Storage Autoencoder
    Memory Autoencoder

Baseline models:
    Storage Isolation Forest
    Memory Isolation Forest
"""

from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Configuration
SPLITS_DIR = Path("splits")
MODELS_DIR = Path("models")

RANDOM_SEED = 42

BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 0.001

PATIENCE = 10
MIN_DELTA = 0.00001

# Percentile used to define unfamiliar behavior
NOVELTY_PERCENTILE = 95

# Isolation Forest configuration
IF_ESTIMATORS = 200

# Execution identifiers and window boundaries are metadata, not behavioral ML features
NON_FEATURE_COLUMNS = [
    "dataset",
    "execution_id",
    "window_start_ns",
    "window_end_ns",
    "label"
]

# Two models
DATA_TYPES = ["storage", "memory"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reproducibility
def set_random_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Request deterministic operations where supported
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Rejoin operation-specific columns by the exact execution/window key
SPLIT_FILE_GROUPS = {
    "storage": ("storage_read", "storage_write"),
    "memory": ("memory_read", "memory_write", "memory_readwrite", "memory_execute"),
}
EXPECTED_FEATURE_COUNTS = {"storage": 26, "memory": 60}
KEY_COLUMNS = [column for column in NON_FEATURE_COLUMNS if column != "label"]

# Load every row from the six Step 06 outputs; join same-window columns
def load_split_data(data_type, split_name):
    result = None

    for group in SPLIT_FILE_GROUPS[data_type]:
        path = SPLITS_DIR / f"{group}_{split_name}.csv"

        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")

        current = pd.read_csv(path)

        if current.empty:
            raise ValueError(f"Input file is empty: {path}")

        if current.columns.duplicated().any():
            raise ValueError(f"Duplicate column names in {path}")

        if "label" not in current.columns or not current["label"].isin([0, 1]).all():
            raise ValueError(f"Missing or invalid binary labels in {path}")

        missing = set(KEY_COLUMNS) - set(current.columns)

        if missing:
            raise ValueError(f"Missing window keys in {path}: {sorted(missing)}")

        if current[KEY_COLUMNS].isna().any().any():
            raise ValueError(f"Missing window identifiers in {path}")

        if current.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"Duplicate execution windows in {path}")

        if result is None:
            result = current
            continue

        # Check overlapping feature values and labels before dropping duplicate columns
        shared = [c for c in current.columns if c not in KEY_COLUMNS and c in result.columns]

        check = result[KEY_COLUMNS + shared].merge(
            current[KEY_COLUMNS + shared],
            on=KEY_COLUMNS,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_left", "_right"),
            sort=False
        )

        if (len(check) != len(result) or len(current) != len(result) or not
        check["_merge"].eq("both").all()):
            raise RuntimeError(f"Window mismatch while joining {path}")

        for column in shared:
            left = pd.to_numeric(check[f"{column}_left"], errors="raise").to_numpy()
            right = pd.to_numeric(check[f"{column}_right"], errors="raise").to_numpy()

            if not np.allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=False):
                raise RuntimeError(f"Shared feature {column} differs in {path}")

        new_columns = [c for c in current.columns
                       if c not in KEY_COLUMNS and c not in result.columns]

        result = result.merge(current[KEY_COLUMNS + new_columns],
                              on=KEY_COLUMNS,
                              how="left",
                              validate="one_to_one",
                              sort=False
                              )

        if len(result) != len(check):
            raise RuntimeError(f"Rows changed while joining {path}")

    actual = len([c for c in result.columns if c not in NON_FEATURE_COLUMNS])

    if actual != EXPECTED_FEATURE_COUNTS[data_type]:
        raise ValueError(f"{data_type}: expected {EXPECTED_FEATURE_COUNTS[data_type]} unique features, found {actual}")

    return result

# Select behavioral features
def get_feature_columns(dataframe):
    feature_columns = [column for column in dataframe.columns if column not in NON_FEATURE_COLUMNS]

    if not feature_columns:
        raise ValueError("No behavioral features found.")

    return feature_columns

# Validate Training and Validation Data
def validate_data(train_dataframe, validation_dataframe, feature_columns):
    required_columns = (NON_FEATURE_COLUMNS + feature_columns)

    for name, dataframe in [("training", train_dataframe), ("validation", validation_dataframe)]:
        missing_columns = set(required_columns) - set(dataframe.columns)

        if missing_columns:
            raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")

        if dataframe[required_columns].isna().any().any():
            raise ValueError(f"{name} contains missing values.")

        # Reject nonnumeric or infinite feature values
        numeric_data = dataframe[feature_columns].apply(pd.to_numeric, errors="raise")

        if not np.isfinite(numeric_data.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} contains infinite values.")

    # Prevent the same execution appearing in both sets
    train_executions = set(zip(train_dataframe["dataset"], train_dataframe["execution_id"]))

    validation_executions = set(zip(validation_dataframe["dataset"], validation_dataframe["execution_id"]))

    if train_executions & validation_executions:
        raise RuntimeError("Execution leakage between training and validation.")

# Prepare feature matrices
def prepare_features(train_dataframe, validation_dataframe, feature_columns):
    x_train = train_dataframe[feature_columns].to_numpy(dtype=np.float64)
    x_validation = validation_dataframe[feature_columns].to_numpy(dtype=np.float64)

    # Fit preprocessing only on training data
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)

    # Neural networks use float32
    x_train = x_train.astype(np.float32)
    x_validation = x_validation.astype(np.float32)

    if (not np.isfinite(x_train).all() or not np.isfinite(x_validation).all()):
        raise ValueError("Non-finite values after scaling.")

    return (x_train, x_validation, scaler)

# Component A: Behavioral Autoencoder
class BehavioralAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        # Compress the behavioral feature vector
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

        # Reconstruct the original feature vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return reconstruction

# Determine autoencoder architecture
def get_architecture(input_size):
    # Scale the network size with the number of behavioral features
    hidden_size = max(8, input_size // 2)
    latent_size = max(2, input_size // 4)

    # Keep the latent representation smaller than the input and hidden layers
    latent_size = min(latent_size, hidden_size - 1, input_size - 1)

    if input_size < 3:
        raise ValueError("Autoencoder requires at least 3 features.")

    return hidden_size, latent_size

# Create data loader
def create_data_loader(features, shuffle):
    tensor = torch.from_numpy(features)

    dataset = TensorDataset(tensor)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

# Calculate reconstruction error
def calculate_reconstruction_errors(model, features):
    loader = create_data_loader(features, shuffle=False)

    model.eval()

    errors = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)

            reconstruction = model(batch)

            # Mean squared reconstruction error for each individual time window
            batch_errors = torch.mean((batch - reconstruction) ** 2, dim=1)

            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)

# Train autoencoder
def train_autoencoder(x_train, x_validation):
    input_size = x_train.shape[1]

    hidden_size, latent_size = get_architecture(input_size)

    model = BehavioralAutoencoder(input_size, hidden_size, latent_size).to(DEVICE)

    train_loader = create_data_loader(x_train, shuffle=True)

    validation_loader = create_data_loader(x_validation, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.MSELoss()

    best_validation_loss = float("inf")
    best_state = None

    patience_counter = 0
    best_epoch = 0

    # Train until the maximum number of epochs or until validation loss stops improving
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for (batch,) in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(reconstruction, batch)
            loss.backward()

            optimizer.step()

        # Evaluate reconstruction loss on validation.
        model.eval()

        validation_loss_sum = 0.0
        validation_samples = 0

        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(DEVICE)

                reconstruction = model(batch)

                loss = criterion(reconstruction, batch)

                validation_loss_sum += (loss.item() * len(batch))
                validation_samples += len(batch)

        validation_loss = (validation_loss_sum / validation_samples)

        if not np.isfinite(validation_loss):
            raise RuntimeError("Autoencoder validation loss became non-finite.")

        # Keep the model with the lowest validation reconstruction loss
        if (validation_loss < best_validation_loss - MIN_DELTA):
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)

    return (model, input_size, hidden_size, latent_size, best_epoch, best_validation_loss)

# Calibrate novelty threshold
def calculate_threshold(errors):
    # The threshold describes unfamiliarity relative to validation benign behavior
    return float(np.percentile(errors, NOVELTY_PERCENTILE))

# Save autoencoder
def save_autoencoder(model, data_type, scaler, feature_columns, input_size, hidden_size, latent_size, threshold):
    output_path = (MODELS_DIR / f"{data_type}_autoencoder.pt")

    # Save architecture, model weights, scaler parameters, and feature order together
    bundle = {
        "model_type": "behavioral_autoencoder",
        "data_type": data_type,
        "architecture": {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "latent_size": latent_size
        },
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value
            in model.state_dict().items()
        },
        "feature_columns": feature_columns,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "novelty_threshold": threshold,
        "novelty_percentile": NOVELTY_PERCENTILE,
        "random_seed": RANDOM_SEED
    }

    torch.save(bundle, output_path)

# Train isolation forest baseline
def train_isolation_forest(x_train, x_validation):
    model = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(x_train)

    # Higher values indicate greater unfamiliarity
    validation_scores = -model.score_samples(x_validation)

    threshold = calculate_threshold(validation_scores)

    return model, threshold

# Save isolation forest
def save_isolation_forest(model, data_type, scaler, feature_columns, threshold):
    output_path = (MODELS_DIR / f"{data_type}_isolation_forest.joblib")

    bundle = {
        "model_type": "isolation_forest",
        "data_type": data_type,
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "novelty_threshold": threshold,
        "novelty_percentile": NOVELTY_PERCENTILE,
        "random_seed": RANDOM_SEED
    }

    joblib.dump(bundle, output_path)

# Process one data type
def process_data_type(data_type):
    print(f"\n{data_type.capitalize()} model training")

    # Load only training and validation data
    train_dataframe = load_split_data(data_type, "train")

    validation_dataframe = load_split_data(data_type, "validation")

    feature_columns = get_feature_columns(train_dataframe)

    # Learn normal behavior from benign windows only; keep ransomware for testing
    train_dataframe = train_dataframe.loc[train_dataframe["label"] == 0].copy()
    validation_dataframe = validation_dataframe.loc[validation_dataframe["label"] == 0].copy()

    if train_dataframe.empty or validation_dataframe.empty:
        raise ValueError(f"{data_type}: benign training and validation windows are required.")

    validate_data(train_dataframe, validation_dataframe, feature_columns)

    (x_train, x_validation, scaler) = prepare_features(
        train_dataframe,
        validation_dataframe,
        feature_columns
    )

    print(f"Features: {len(feature_columns)}")
    print(f"Training windows: {len(x_train)}")
    print(f"Validation windows: {len(x_validation)}")

    # Component A: train behavioral autoencoder
    (autoencoder, input_size, hidden_size, latent_size, best_epoch, best_loss) = train_autoencoder(
        x_train,
        x_validation
    )

    validation_errors = calculate_reconstruction_errors(autoencoder, x_validation)

    ae_threshold = calculate_threshold(validation_errors)

    save_autoencoder(
        autoencoder,
        data_type,
        scaler,
        feature_columns,
        input_size,
        hidden_size,
        latent_size,
        ae_threshold
    )

    print(f"Autoencoder: complete (epoch {best_epoch}, validation MSE {best_loss:.6f})")

    # Train isolation forest baseline
    (isolation_forest, if_threshold) = train_isolation_forest(x_train, x_validation)

    save_isolation_forest(
        isolation_forest,
        data_type,
        scaler,
        feature_columns,
        if_threshold
    )

    print("Isolation Forest: complete")

# Main
if __name__ == "__main__":
    set_random_seed()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # File title
    print("\nDTBAD Model Training\n")

    print(f"Device: {DEVICE}")

    for data_type in DATA_TYPES:
        process_data_type(data_type)

    # Closure
    print("\nModel Training Complete")

