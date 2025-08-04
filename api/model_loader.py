import os
import mlflow
from logger import get_logger

logger = get_logger(__name__)

# Set MLflow Tracking URI from env variable
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))


def load_best_model_from_registry(
    model_name: str = "best_model",
    stage: str = None
):
    """
    Loads the best model from the MLflow model registry.
    Args:
        model_name (str): Name of the registered model.
        stage (str): Stage to load the model from ('Production', 'Staging', etc.).
        relative_tracking_path (str): Path to the MLflow tracking directory.

    Returns:
        Loaded MLflow model or None if loading fails.
    """
    try:
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        if stage is None:
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded model '{model_name}' from stage '{stage}'.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow registry: {e}")
        return None
