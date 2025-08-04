"""FastAPI Module"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from router import agent

# Load environment variables
if os.environ.get("DOCKER_ENV", "false").lower() == "true":
    # Running in Docker – load Docker-specific env
    dotenv_path = Path(__file__).resolve().parents[2] / "api/.env.docker"
else:
    # Running locally – load default .env
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"

# Load the chosen .env file
load_dotenv(dotenv_path=dotenv_path)
# Create FastAPI app
app = FastAPI(
    title="Prediction API",
    description="API for model prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(agent.router)


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
