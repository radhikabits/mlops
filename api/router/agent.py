"This code is part of a FastAPI application that handles prediction requests for a machine learning model. It includes an endpoint for generating predictions based on input features."
from api.models import PredictionRequest, PredictionResponse
from fastapi import APIRouter
from fastapi import HTTPException
from utils.logger import get_logger  

logger = get_logger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["agents"],
)

@router.post("/prediction", response_model=PredictionResponse)
async def prediction(request: PredictionRequest):
    """Endpoint to generate predictions based on input features."""
    try:
        return PredictionResponse(
            predicted_class="setosa",  # Dummy value for example
            confidence=0.92  # Dummy value for example
        )
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e
 
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e
