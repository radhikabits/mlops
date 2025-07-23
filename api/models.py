from pydantic import BaseModel, Field
from typing import List

# TODO: This will change based on the dataset selected
# For now, we are using a dummy dataset with 4 features and 3 classes
class PredictionRequest(BaseModel):
    """Schema for input features sent in JSON."""
    features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2])

class PredictionResponse(BaseModel):
    """Schema for the prediction response."""
    predicted_class: str = Field(..., example="setosa")
    confidence: float = Field(..., example=0.92)
