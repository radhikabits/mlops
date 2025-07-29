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
├── notebooks/              → EDA and experiment notebooks
├── src/
│   ├── data/               → Preprocessing scripts
│   ├── models/             → Training, evaluation, prediction scripts
│   ├── utils/              → Logger and input schemas
│   └── config.py           → Configuration and parameters
├── api/
│   ├── app.py              → FastAPI app with prediction endpoint
│   ├── main.py             → Entrypoint for running the API
│   └── requirements.txt    → API dependencies
├── docker/
│   ├── Dockerfile          → Docker build file
│   └── entrypoint.sh       → Startup script (if needed)
├── test/                   → Contains pytest
├── dvc.yaml                → DVC pipeline file (for California Housing)
├── mlruns/                 → MLflow tracking logs
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

# To Run the API
uvicorn api.main:app --reload

API base URL: http://127.0.0.1:8000

Interactive Swagger Docs: http://127.0.0.1:8000/docs

ReDoc Docs: http://127.0.0.1:8000/redoc

Health Check Endpoint: http://127.0.0.1:8000/health

# To Run the Tests
pytest tests/

# Raw Data
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

# Preprocessing
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

🔧 How to Run
# Train models and log experiments

python src/train_linear.py
python src/train_tree.py

# Launch MLflow UI (optional)
mlflow ui  # Visit http://127.0.0.1:5000

## # Select and register the best model

After training multiple models, the `select_best_and_register.py` script compares them using a selected metric (default: `mse`) and registers the best-performing model in MLflow.

**Steps Performed:**
- Retrieves the MLflow experiment runs
- Selects the run with the lowest MSE
- Checks if the model is already registered
- Registers the model (or adds a new version)

python src/select_best_and_register.py
