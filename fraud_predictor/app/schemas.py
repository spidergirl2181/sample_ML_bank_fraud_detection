from pydantic import BaseModel, Field
from typing import List

class FraudInput(BaseModel):
    average_age: float = Field(..., example=35)
    is_1st_ratio: float = Field(..., example=0.2)
    avg_amount: float = Field(..., example=120)
    txn_velocity: float = Field(..., example=5)


class PredictRequest(BaseModel):
    data: List[FraudInput]


class PredictResponse(BaseModel):
    predictions: List[float]