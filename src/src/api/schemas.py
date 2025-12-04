from pydantic import BaseModel
from datetime import datetime

class TripInput(BaseModel):
    pickup_datetime: datetime
    pickup_location_id: int
    dropoff_location_id: int
    passenger_count: int
    trip_distance: float

class PredictionOutput(BaseModel):
    predicted_fare: float
    predicted_duration_minutes: float
    model_version: str
