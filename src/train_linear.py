import os
import sys
import time
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_config, save_model
from utils.logger import get_logger

logger = get_logger(__name__)

def train_linear_regression():
    """
    Train a Linear Regression model on California Housing data,
    log experiment using MLflow, and persist the best model.
    Logs:
        - Model parameters from config.yaml
        - MSE and R² as evaluation metrics
        - Training duration
        - Trained model artifact to MLflow
    """
    start_time = time.time()
    logger.info("Loading configuration and data...")
    # Load configuration
    config = load_config()
    params = config.get('linear_regression', {})

    # Load training data
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    logger.info("Starting MLflow run for Linear Regression...")
    try:
        with mlflow.start_run(run_name="LinearRegression"):
            model = LinearRegression()
            model.fit(X_train, y_train)

            logger.info("Model training complete. Evaluating...")
            logger.info(f"Training completed in {duration:.2f} seconds.")
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log hyperparameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)       

            logger.info(f"Logged metrics - MSE: {mse:.4f}, R2: {r2:.4f}")

            # Log model with signature and input example (optional but recommended)
            input_example = X_test[:5]  # small batch
            mlflow.sklearn.log_model(
                model,
                "model",
                input_example=input_example,
                registered_model_name=None  
            )
            logger.info("Model logged to MLflow.")

            # Save model locally
            model_path = os.path.join("models", "linear_regression.pkl")
            save_model(model, model_path)
            logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.exception(f"Training failed: {str(e)}")
    finally:
        duration = time.time() - start_time
        logger.info(f"Total training time: {duration:.2f} seconds")
        mlflow.log_metric("training_time_sec", duration)

if __name__ == "__main__":
    train_linear_regression()
