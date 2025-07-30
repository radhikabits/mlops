from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """
    Schema for input features based on California Housing dataset.
    """
    MedInc: float = Field(..., example=8.3, description="Median income in block group")
    HouseAge: float = Field(..., example=42, description="Median house age in block group")
    AveRooms: float = Field(..., example=6.5, description="Average number of rooms per household")
    AveBedrms: float = Field(..., example=1.1, description="Average number of bedrooms per household")
    Population: float = Field(..., example=322, description="Block group population")
    AveOccup: float = Field(..., example=2.6, description="Average house occupancy")
    Latitude: float = Field(..., example=34.2, description="Latitude of the block group")
    Longitude: float = Field(..., example=-118.4, description="Longitude of the block group")


class PredictionResponse(BaseModel):
    """Schema for the prediction response."""
    predicted_price: float = Field(..., example=2.85, description="Predicted median house price")
