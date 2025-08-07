
# MLOps Project: Build, Track, Package, Deploy, and Monitor an ML Model

## 🚀 Overview

This project demonstrates a complete **MLOps pipeline** following industry best practices. It covers the full lifecycle of a machine learning model—from data acquisition and preprocessing to model training, versioning, packaging, deployment, and monitoring.

**Dataset used**: [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)  
**Task**: Regression (predicting median house value)

---

## 🎯 Objectives

- ✅ Version control data and code
- ✅ Track experiments and model artifacts using **MLflow**
- ✅ Serve the model as a REST API using **FastAPI**
- ✅ Containerize the application using **Docker**
- ✅ Automate CI/CD workflows using **GitHub Actions**
- ✅ Implement logging and basic monitoring with **Prometheus + Grafana**
- ✅ Auto-retrain model when new data arrives

---

## 🏗️ Project Structure

```bash
mlops/
├── api/                     # FastAPI application
│   ├── main.py              # App entry point
│   ├── router/agent.py      # Prediction endpoint
│   ├── model_loader.py      # Load latest model
│   ├── models.py            # Input/output schema
│   ├── logger.py            # API logger
│   └── requirements.txt     # API dependencies
├── data/
│   ├── new/                 # New Raw dataset                    
│   ├── raw/                 # Raw dataset
│   └── processed/           # Cleaned & preprocessed data
├── docker/                 
│   ├── Dockerfile           # For serving the API
│   └── Dockerfile.trainer   # For model training
├── logs/                    # Application and prediction logs
├── mlruns/                  # MLflow run tracking
├── model/                   # Saved model artifacts
├── src/                     # Source scripts
│   ├── fetch_data.py        # Load raw dataset
│   ├── preprocess.py        # Clean and transform data
│   ├── train_linear.py      # Linear Regression model
│   ├── train_tree.py        # Decision Tree model
│   └── select_best_and_register.py  # Registers best model in MLflow
│   └── run_training_pipeline.py  # Runs the training pipeline
│   └── watch_and_train.py  # poll for retraining the model
├── test/                    # Unit tests with pytest
├── utils/                   
│   ├── common.py            
│   ├── config.yaml          
│   └── logger.py            
├── dvc.yaml                 # DVC pipeline config
├── docker-compose.yml       # Docker Compose for Remote setup
├── docker-compose.local.yml # Docker Compose for Local setup
├── requirements.txt         # Project-level dependencies
├── summary.pdf              # Architecture overview
├── video-demo.mp4           # Project walkthrough
└── README.md                # This file
```

---

## 🛠️ Setup Instructions

### 📦 Prerequisites

- Python 3.8+
- pip
- [venv](https://docs.python.org/3/library/venv.html)
- Docker Desktop
- WSL (for Windows)
- Git

### 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/radhikabits/mlops.git
cd mlops

# Create a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```
---

## 🚀 Run the Full Training Pipeline

```bash
python src/run_training_pipeline.py data/raw/housing.csv

## 🔍 What This Pipeline Does - Step-by-Step Execution

### 📁 1. Data Acquisition

```bash
python src/fetch_data.py
```

**Track data using DVC:**

```bash
dvc init
dvc add data/raw/housing.csv
dvc push  # to remote storage
```

### 🧹 2. Data Preprocessing

```bash
python src/preprocess.py
```

Steps:
- Remove missing values
- Remove outliers (IQR method)
- Feature scaling (StandardScaler)
- Train-test split

---

### 🎓 3. Model Training & Experiment Tracking

Train models and log results with MLflow:

```bash
python src/train_linear.py
python src/train_tree.py
```

Launch MLflow UI (optional):

```bash
mlflow ui  # http://127.0.0.1:5000
```

---

### 🏆 4. Select and Register Best Model

```bash
python src/select_best_and_register.py
```

This compares model runs, selects the one with lowest MSE, and registers it in the MLflow model registry.


### 🌐 Serve the Model via FastAPI

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

Access:

- Base URL: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Redoc: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/metrics`

---

### ✅ Run Unit Tests

```bash
pytest test/
```

---

## 🐳 Docker Deployment

### 🧱 Build & Run Locally

Ensure Docker and WSL are installed.

```bash
# Build and run container
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up
```

Access:

- API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`

---

## ⚙️ CI/CD with GitHub Actions

### 🔐 GitHub Secrets (Setup)

| Secret Name         | Description                        |
|---------------------|------------------------------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username           |
| `DOCKERHUB_TOKEN`    | Docker Hub access token/password   |
| `DOCKER_IMAGE_NAME`  | e.g., `yourusername/mlops-api`     |

### 🧪 Lint Code

```bash
flake8 .
```

---

## 📊 Monitoring with Prometheus & Grafana

### Local URLs

| Service     | URL                         | Description              |
|-------------|-----------------------------|--------------------------|
| Prometheus  | http://localhost:9090       | Metrics explorer         |
| Grafana     | http://localhost:3000       | Dashboard visualization  |

- **Grafana credentials**: `admin / admin`

### 📈 Prometheus Queries

- Total requests: `http_requests_total`
- Latency histogram: `http_request_duration_seconds_bucket`
- Error rate: `http_requests_total{status_code=~"5.."}`  
- Per endpoint: `sum by (handler) (http_requests_total)`

### 🧩 Grafana Dashboard Import

1. Open Grafana → `+` → **Import**
2. Dashboard ID: `16110` (FastAPI Observability)
3. Set data source: `Prometheus`
4. Click **Import**

---

## 🔁 Automatic Model Retraining on New Data

A file watcher observes the `data/new/` folder and retriggers the pipeline when a new `.csv` file is added.

### 👁️ Watcher Behavior

- **Monitors**: `data/new/`
- **Trigger**: New file addition
- **Action**: Runs `run_training_pipeline.py`
- **Observer**: `PollingObserver` (Docker-compatible)

---

## 📄 Resources

- 📄 `summary.pdf` – High-level architecture
- 🎥 `video-demo.mp4` – 5-min project walkthrough

---

## 🙌 Acknowledgements

This project was built as part of the **BITS Pilani WILP MLOps coursework**, integrating key learnings on ML lifecycle management, deployment, and observability in real-world ML systems.
