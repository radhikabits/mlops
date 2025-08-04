"This code is part of a FastAPI application that handles prediction requests for a machine learning model. It includes an endpoint for generating predictions based on input features."
import pandas as pd
from models import PredictionRequest, PredictionResponse
from fastapi import APIRouter
from fastapi import HTTPException
from model_loader import load_best_model_from_registry
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["agents"],
)


@router.post("/prediction", response_model=PredictionResponse)
async def prediction(request: PredictionRequest):
    """Endpoint to generate predictions based on input features."""
    logger.info(f"Incoming prediction request: {request.dict()}")
    try:
        # Load the model on demand
        model = load_best_model_from_registry()  # Load on demand
        if model is None:
            raise RuntimeError("Model not loaded. Check MLflow registry or URI.")

        input_df = pd.DataFrame([request.dict()])
        prediction = model.predict(input_df)
        logger.info(f"Model prediction response: {prediction.tolist()}")
        return PredictionResponse(predicted_price=float(prediction[0]))

    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e
