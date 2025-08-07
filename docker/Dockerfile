# docker/Dockerfile

# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app/api

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire app
COPY api/ ./
# COPY mlruns/ ./api/mlruns/

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
