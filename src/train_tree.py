import os
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
from utils.common import load_config, save_model
from utils.logger import get_logger

logger = get_logger(__name__)

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
    try:
        logger.info("Loading configuration and data...")
        config = load_config()
        params = config["decision_tree"]

        # Load preprocessed data
        X_train = np.load("data/processed/X_train.npy")
        X_test = np.load("data/processed/X_test.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")

        logger.info("Starting MLflow run for Decision Tree...")
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

            # Infer signature and log model with input example
            signature = infer_signature(X_test, y_pred)
            input_example = X_test[:1]

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=None  # or "decision_tree_model" if using model registry
            )

            # Save locally
            save_model(model, "models/decision_tree.pkl")

            logger.info(f"Model training complete. MSE: {mse:.4f}, R2: {r2:.4f}")
    except Exception as e:
        logger.exception("Error occurred during Decision Tree training.")
        raise e
    finally:
        duration = time.time() - start_time
        logger.info(f"Total training time: {duration:.2f} seconds")
        mlflow.log_metric("training_time_sec", duration)

if __name__ == "__main__":
    train_decision_tree()
