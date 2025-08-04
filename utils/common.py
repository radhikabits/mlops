from pathlib import Path
import yaml
import joblib
import os
from dotenv import load_dotenv

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_environment_variables():
    """Load environment variables from .env file."""
    if os.environ.get("DOCKER_ENV", "false").lower() == "true":
        # Running in Docker – load Docker-specific env
        dotenv_path = Path(__file__).resolve().parents[2] / "api/.env.docker"
    else:
        # Running locally – load default .env
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"

    # Load the chosen .env file
    load_dotenv(dotenv_path=dotenv_path)
