import os
from pathlib import Path
import mlflow
from logger import get_logger 

logger = get_logger(__name__)

# Dynamically get path to `mlruns` relative to this file
mlruns_path = Path(__file__).resolve().parents[1] / "mlruns"
mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")

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
        if stage == "None":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{stage}"

        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded model '{model_name}' from stage '{stage}'.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow registry: {e}")
        return None
