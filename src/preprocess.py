"""Preprocessing script for the California Housing dataset.
This script loads the dataset, removes outliers, scales features, and splits the data into training"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Paths
RAW_DATA_PATH = "data/raw/housing.csv"
PROCESSED_DIR = "data/processed"
SCALER_PATH = os.path.join("models", "scaler.pkl")


def load_data(path: str) -> pd.DataFrame:
    """Load California Housing dataset from CSV."""
    if os.path.exists(path):
        logger.info(f"Loading dataset from: {path}")
        return pd.read_csv(path)
    else:
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"{path} does not exist.")


def remove_outliers_iqr(X: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    return X[mask]


def preprocess_data(
    df: pd.DataFrame, scale: bool = True, save_scaler: bool = True
) -> tuple:
    """Preprocess dataset for ML training and save processed files."""
    logger.info("Starting preprocessing...")

    # Drop missing values
    if df.isnull().sum().sum() > 0:
        logger.warning("Missing values found. Dropping them.")
        df.dropna(inplace=True)

    # Separate features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Remove outliers
    logger.info("Removing outliers using IQR")
    X = remove_outliers_iqr(X)
    y = y.loc[X.index]  # Align y with filtered X

    # Feature scaling
    if scale:
        logger.info("Applying StandardScaler to features")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if save_scaler:
            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            joblib.dump(scaler, SCALER_PATH)
            logger.info(f"Scaler saved to {SCALER_PATH}")
    else:
        X = X.values

    # Train/Test split
    logger.info("Splitting into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    logger.info(f"Processed data saved to {PROCESSED_DIR}")

    return X_train, X_test, y_train, y_test


def main(data_path: str = RAW_DATA_PATH):
    df = load_data(data_path)
    preprocess_data(df)
    logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python preprocess.py <data_path>")
    main(sys.argv[1])
