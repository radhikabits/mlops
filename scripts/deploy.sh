#!/bin/bash
set -e

echo "[INFO] Environment variables loaded from GitHub Actions context."

# Validate required variables
: "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is not set}"

echo "[INFO] Pulling latest images from Docker Hub..."
docker pull ${DOCKERHUB_USERNAME}/mlops-api:latest
docker pull ${DOCKERHUB_USERNAME}/trainer:latest

echo "[INFO] Stopping existing containers..."
docker compose down

echo "[INFO] Starting services using docker compose..."
docker compose up -d

echo "[INFO] Waiting for containers to stabilize..."
sleep 5

echo "[INFO] Checking container status..."
docker ps

echo "[INFO] Verifying MLflow tracking server..."
if curl -s http://localhost:5000 | grep -i mlflow; then
  echo "[SUCCESS] MLflow is reachable."
else
  echo "[WARN] MLflow may not be reachable. Check logs below."
fi

echo "[INFO] Showing logs for API service..."
docker logs -f mlops-api

echo "[INFO] Showing logs for Trainer service..."
docker logs -f mlops_trainer

echo "[INFO] Stopping all services..."
docker compose down
echo "[INFO] Deployment script completed successfully."
exit 0
# End of deploy.sh
# This script is intended to be run in a CI/CD pipeline or manually for deployment purposes.
