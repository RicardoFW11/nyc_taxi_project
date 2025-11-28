"""
Pydantic schemas for API request/response validation
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    """
    Request schema for taxi trip prediction.
    All fields needed to make a fare prediction.
    """
    VendorID: int = Field(
        ..., 
        ge=1, 
        le=2, 
        description="Vendor ID (1=Creative Mobile, 2=VeriFone)"
    )
    passenger_count: int = Field(
        ..., 
        ge=1, 
        le=6, 
        description="Number of passengers (1-6)"
    )
    trip_distance: float = Field(
        ..., 
        gt=0, 
        le=100, 
        description="Trip distance in miles"
    )
    payment_type: int = Field(
        ..., 
        ge=1, 
        le=4, 
        description="Payment type (1=Credit, 2=Cash, 3=No charge, 4=Dispute)"
    )
    pickup_datetime: str = Field(
        ..., 
        description="Pickup datetime in format 'YYYY-MM-DD HH:MM:SS'",
        examples=["2022-05-15 14:30:00"]
    )

    @field_validator('pickup_datetime')
    @classmethod
    def validate_datetime(cls, v):
        """Validate datetime format"""
        try:
            datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return v
        except ValueError:
            raise ValueError(
                "Invalid datetime format. Use 'YYYY-MM-DD HH:MM:SS'"
            )

    class Config:
        json_schema_extra = {
            "example": {
                "VendorID": 1,
                "passenger_count": 2,
                "trip_distance": 5.3,
                "payment_type": 1,
                "pickup_datetime": "2022-05-15 14:30:00"
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for fare prediction
    """
    predicted_fare: float = Field(
        ..., 
        description="Predicted fare amount in USD"
    )
    model_version: str = Field(
        ..., 
        description="Model version used for prediction"
    )
    prediction_timestamp: str = Field(
        ..., 
        description="When the prediction was made"
    )
    input_features: dict = Field(
        ..., 
        description="Features used for prediction (for debugging)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_fare": 18.45,
                "model_version": "linear_regression_v1",
                "prediction_timestamp": "2024-11-28T15:30:00",
                "input_features": {
                    "VendorID": 1,
                    "passenger_count": 2,
                    "trip_distance": 5.3,
                    "payment_type": 1,
                    "pickup_hour": 14,
                    "pickup_day_of_week": 0,
                    "pickup_month": 5,
                    "distance_euclidean": 5.3
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str