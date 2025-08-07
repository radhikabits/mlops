"""select_best_and_register.py
This script selects the best model from the MLflow registry based on a specified metric,"""
import os
import sys
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger
from utils.common import load_environment_variables

logger = get_logger(__name__)


def register_best_from_registry(
    candidate_models: list,
    metric_key: str = "mse",
    greater_is_better: bool = False,
    best_model_name: str = "best_model",
    tracking_uri: Optional[str] = None
) -> Optional[str]:

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        logger.info("Tracking URI set to: %s", tracking_uri)

    client = MlflowClient()
    best_metric = None
    best_model_uri = None
    best_run_id = None

    for model_name in candidate_models:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except RestException:
            logger.warning("Model '%s' not found. Skipping...", model_name)
            continue

        for version in versions:
            run_id = version.run_id
            run = client.get_run(run_id)
            metric_value = run.data.metrics.get(metric_key)

            if metric_value is None:
                logger.info("Metric '%s' not found for run_id=%s. Skipping.", metric_key, run_id)
                continue

            if (best_metric is None) or (
                greater_is_better and metric_value > best_metric
            ) or (
                not greater_is_better and metric_value < best_metric
            ):
                best_metric = metric_value
                best_model_uri = version.source
                best_run_id = run_id

    if best_model_uri is None or best_run_id is None:
        logger.warning("No valid models found with metric: %s", metric_key)
        return None

    # Register best model with mlflow.register_model (auto-creates registered model if needed)
    model_uri = f"{best_model_uri}"
    result = mlflow.register_model(
        model_uri=model_uri,
        name=best_model_name
    )
    logger.info("Best model (run_id=%s, %s=%.5f) registered as '%s' (version %s)",
                best_run_id, metric_key, best_metric, best_model_name, result.version)

    return result.version


def main():
    load_environment_variables()
    # Load tracking URI from environment variable
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    register_best_from_registry(
        candidate_models=["decision_tree", "linear_regression"],
        metric_key="mse",
        greater_is_better=False,
        best_model_name="best_model",
        tracking_uri=TRACKING_URI
    )


if __name__ == "__main__":
    main()
