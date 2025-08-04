# mlops
# MLOps Project: Build, Track, Package, Deploy and Monitor an ML Model

## Project Overview

This project demonstrates a complete **MLOps pipeline** using best practices. The goal is to train, track, version, deploy, and monitor a machine learning model using a well-known dataset: **California Housing** (regression task) or **Iris** (classification task).

We have chosen dataset: **California Housing**
---

## Objectives

- Version control code and data
- Track experiments and model artifacts using MLflow
- Package the ML model as a REST API using FastAPI
- Containerize the API with Docker
- Automate CI/CD using GitHub Actions
- Implement logging and basic monitoring

---

## Project Architecture

```bash
mlops/
├── data/                   → Raw and processed datasets
│   ├── raw/                 
│   └── processed/
├── src/
│   ├── fetch_data.py       → Fetches the California housing dataset
│   ├── preprocess.py       → Preprocessing script
│   ├── train_linear.py     → trains a Linear Regression model
│   ├── train_tree.py       → trains a Decision Tree model
│   └── select_best_and_register.py  → selects the best model from the MLflow registry
├── utils/
│   ├── common.py
│   ├── config.ymal
│   ├── logger.py    
├── api/
│   ├── router/agent.py     → FastAPI app with prediction endpoint
│   ├── main.py             → Entrypoint for running the API
│   └── requirements.txt    → API dependencies
│   └── logger.py
│   └── models.py    → API models
│   └── model_loader.py      
├── docker/
│   ├── Dockerfile          → Docker build file
│   └── Dockerfile.trainer
├── test/                   → Contains pytest
├── dvc.yaml                → DVC pipeline file (for California Housing)
├── mlruns/                 → MLflow tracking logs
├── docker-cpmpose.yml      → compose file
├── model/                  → Saved model artifacts
├── logs/                   → Prediction and app logs
├── README.md               → Project summary (this file)
├── summary.pdf             → 1-page architecture summary
├── video-demo.mp4          → 5-min project walkthrough
└── requirements.txt        → Project-level dependencies

## Project Setup
Prerequisites
Make sure you have the following installed:

Python 3.8+

pip (Python package manager)

venv for isolated environments
Clone and Set Up the Project

# Clone the repository
git clone https://github.com/radhikabits/mlops.git

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate     # On Linux/macOS
venv\Scripts\activate        # On Windows

# Install dependencies (if any)
pip install -r requirements.txt

# Steps to follow

# 1. Raw Data
# Run below file to fetch the raw data
py "src\fetch_data.py"

# Data Version Control, is an open-source tool that helps you manage and version control data

# How It Works
- dvc init → Sets up DVC in your Git repo.
- Starts tracking your dataset.
    dvc add data/raw/housing.csv (to track it with DVC)
- dvc push → Uploads data to remote storage.
- dvc pull → Downloads exact data version when needed.
- dvc run → Defines pipeline stages with dependencies and outputs.

# 2. Preprocessing
The California Housing dataset is preprocessed before model training to ensure data quality and consistency. Preprocessing includes:

1. Dropping missing values
2. Removing outliers using the IQR method
3. Feature-target separation
4. Scaling features with StandardScaler
5. Splitting into training and testing sets

# How to Run
py "src\preprocess.py"

# Model Training with MLflow
# We use MLflow to manage the end-to-end model lifecycle:

1. Train two models: Linear Regression and Decision Tree

2. Log parameters, metrics (MSE, R²), and artifacts

3. Compare and select the best model

4. Register the best model in the MLflow Model Registry

🔧 3. How to Run
# Train models and log experiments

python src/train_linear.py
python src/train_tree.py

# Launch MLflow UI (optional)
mlflow ui  # Visit http://127.0.0.1:5000

# 4. Select and register the best model

After training multiple models, the `select_best_and_register.py` script compares them using a selected metric (default: `mse`) and registers the best-performing model in MLflow.

**Steps Performed:**
- Retrieves the MLflow experiment runs
- Selects the run with the lowest MSE
- Checks if the model is already registered
- Registers the model (or adds a new version)

python src/select_best_and_register.py

# To Run the API
cd api
pip install -r requirements.txt
uvicorn main:app --reload
 # in debug mode
uvicorn main:app --reload --log-level debug

API base URL: http://127.0.0.1:8000
Interactive Swagger Docs: http://127.0.0.1:8000/docs
ReDoc Docs: http://127.0.0.1:8000/redoc
Health Check Endpoint: http://127.0.0.1:8000/health

# To Run the Tests
pytest tests/

# Build and run the Docker container

# Install Docker Desktop
    1. Go to Docker official website
        https://www.docker.com/products/docker-desktop/
    2. Download Docker Desktop for Windows
    3. Install it following the instructions.
    4. Restart your machine after installation (important).
# Install WSL
    1. wsl --install

# Build the image and run the container
    1. Open WSL Terminal or PowerShell
    2. Navigate to Your Project Directory
        cd /mnt/c/BITS/Degree/Sem3/MLOps/Assignment/mlops
    3. Build the Docker Image
        docker-compose build
    4. Run the Docker Container
        docker-compose up
    5. You can now visit:
        API: http://localhost:8000
        MLflow UI: http://localhost:5000
# CICD
1. GitHub Secrets
- Go to your GitHub repo → Settings → Secrets and variables → Actions → New repository secret, and
    Secret  Name	            Description
    DOCKERHUB_USERNAME	    Your Docker Hub username
    DOCKERHUB_TOKEN	        Docker Hub access token/password
    DOCKER_IMAGE_NAME	    e.g., yourusername/mlops-api

# Run flake8
flake8 .
