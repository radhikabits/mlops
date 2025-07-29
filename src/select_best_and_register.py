import os
import sys
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.entities import Run
from urllib.parse import urlparse

# Append root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger

logger = get_logger(__name__)


def register_best_model(
    experiment_name: str = "california_housing",
    model_name: str = "best_model",
    metric_key: str = "mse",
    greater_is_better: bool = False
) -> Optional[str]:
    """
    Registers the best model from an MLflow experiment based on a given metric.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.
    - model_name (str): Name to register the best model under.
    - metric_key (str): The metric used to determine the best model (e.g., "mse", "r2").
    - greater_is_better (bool): Set to True if a higher metric is better (e.g., for r2), else False.

    Returns:
    - run_id (str): ID of the best run that was registered (if any), else None.
    """
    try:
        client = MlflowClient()

        # Get the experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)

        order = "DESC" if greater_is_better else "ASC"
        logger.info(f"Searching best run in experiment '{experiment_name}' by metric '{metric_key}' ({order})")

        # Search for the best run
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_key} {order}"],
            max_results=1
        )

        if not runs:
            logger.warning(f"No runs found in experiment '{experiment_name}'.")
            return None

        best_run: Run = runs[0]
        metric_value = best_run.data.metrics.get(metric_key)

        if metric_value is None:
            logger.warning(f"The metric '{metric_key}' was not logged for the best run.")
            return None

        run_id = best_run.info.run_id
        artifact_uri = best_run.info.artifact_uri
        model_path = os.path.join(artifact_uri, "model")

        logger.info(f"Best run ID: {run_id} | {metric_key}: {metric_value}")
        logger.info(f"Model path: {model_path}")

        # Check and register the model
        try:
            client.get_registered_model(model_name)
            logger.info(f"Model '{model_name}' already exists. Adding new version.")
        except RestException:
            logger.info(f"Creating new registered model: '{model_name}'")
            client.create_registered_model(model_name)

        # Validate model path
        if urlparse(model_path).scheme not in ["file", "s3", "dbfs", "gs"]:
            logger.error(f"Unsupported model artifact URI: {model_path}")
            return None

        # Register model version
        result = client.create_model_version(
            name=model_name,
            source=model_path,
            run_id=run_id
        )
        logger.info(f"Registered model '{model_name}' version {result.version} from run {run_id}")
        return run_id

    except Exception as e:
        logger.exception("Failed to register the best model.")
        return None


if __name__ == "__main__":
    register_best_model(
        experiment_name="california_housing",
        model_name="best_model",
        metric_key="mse",  # Change to 'r2' if needed
        greater_is_better=False  # Set to True for r2
    )
