"""train_tree.py
This script trains a Decision Tree Regressor on the California Housing dataset,"""
import os
from pathlib import Path
from dotenv import load_dotenv
import sys
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_config, save_model, load_environment_variables
from utils.logger import get_logger

logger = get_logger(__name__)

load_environment_variables()

def train_decision_tree():
    """
    Train a Decision Tree Regressor on preprocessed California Housing data.
    Logs the model training run using MLflow, including:
    - Parameters and metrics (MSE, R²)
    - Input signature and input example
    - Training time
    - Model artifacts (saved locally and to MLflow)

    Raises:
        Exception: If training or logging fails
    """
    start_time = time.time()
    logger.info("Loading configuration and data...")
    config = load_config()
    params = config["decision_tree"]

    # Load preprocessed data
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    logger.info("Starting MLflow run for Decision Tree...")
    try:
        # Set MLflow Tracking URI from env variable
        tracking_url = os.getenv("MLFLOW_TRACKING_URI")
        logger.info(f"MLflow Tracking URI: {tracking_url}")
        mlflow.set_tracking_uri(tracking_url)
        mlflow.set_experiment("california_housing")
        with mlflow.start_run(run_name="DecisionTreeRegressor"):
            model = DecisionTreeRegressor(max_depth=params["max_depth"])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Logging parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            logger.info(f"Model training complete. MSE: {mse:.4f}, R2: {r2:.4f}")
            # Infer signature and log model with input example
            signature = infer_signature(X_test, y_pred)
            input_example = X_test[:1]

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
                registered_model_name="decision_tree"
            )

            # Save locally
            model_path = os.path.join("models", "decision_tree.pkl")
            save_model(model, model_path)
            logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.exception("Error occurred during Decision Tree training.")
        raise e
    finally:
        duration = time.time() - start_time
        logger.info(f"Total training time: {duration:.2f} seconds")
        mlflow.log_metric("training_time_sec", duration)

if __name__ == "__main__":
    train_decision_tree()
