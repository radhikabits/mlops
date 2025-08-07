import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger
from src.fetch_data import save_housing_data
from src.preprocess import main as preprocess_main
from src.train_linear import main as train_linear_main
from src.train_tree import main as train_tree_main
from src.select_best_and_register import main as select_best_main

logger = get_logger(__name__)


def run_pipeline(data_path):
    logger.info(f"[Pipeline] Starting training pipeline with {data_path}")
    os.environ["DATA_PATH"] = data_path

    if data_path is None or not os.path.exists(data_path):
        logger.info("[Pipeline] Data path not found. Fetching fresh data.")
        data_path = save_housing_data()  # returns full path

    # Step 2: Preprocess
    preprocess_main(data_path)

    # Step 3: Train models
    train_linear_main()
    train_tree_main()

    # Step 4: Select best and register
    select_best_main()

    logger.info("[Pipeline] Training pipeline completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python run_training_pipeline.py <data_path>")
    run_pipeline(sys.argv[1])
