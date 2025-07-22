# mlops
# MLOps Project: Build, Track, Package, Deploy and Monitor an ML Model

## Project Overview

This project demonstrates a complete **MLOps pipeline** using best practices. The goal is to train, track, version, deploy, and monitor a machine learning model using a well-known dataset: **California Housing** (regression task) or **Iris** (classification task).

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

# Running Tests
pytest tests/


