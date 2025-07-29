import os
import sys
import time
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_config, save_model
from utils.logger import get_logger

logger = get_logger(__name__)

def train():
    """
    Train a Linear Regression model using preprocessed California Housing data,
    log experiment with MLflow, and save the trained model.
    """
    logger.info("Loading configuration and data...")
    
    start_time = time.time()
    config = load_config()
    params = config.get('linear_regression', {})

    # Define data paths
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
            duration = time.time() - start_time
            logger.info(f"Training completed in {duration:.2f} seconds.")
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log hyperparameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)       
            mlflow.log_metric("training_duration_sec", duration)

            logger.info(f"Logged metrics - MSE: {mse:.4f}, R2: {r2:.4f}, Duration: {duration:.2f} seconds")

            # Log model artifact to MLflow
            mlflow.sklearn.log_model(model, "model")
            logger.info("Model logged to MLflow.")

            # Save model locally
            model_path = os.path.join("models", "linear_regression.pkl")
            save_model(model, model_path)
            logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.exception(f"Training failed: {str(e)}")

if __name__ == "__main__":
    train()
