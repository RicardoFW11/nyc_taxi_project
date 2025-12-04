import joblib
import pandas as pd
from src.configs.paths import MODELS_DIR
from src.api.schemas import TripInput

class Predictor:
    def __init__(self):
        self.model_fare = None
        self.model_duration = None
        self.load_models()

    def load_models(self):
        # Only load if files exist (prevents errors during build phase if models aren't trained yet)
        if (MODELS_DIR / "xgboost_fare.pkl").exists():
            self.model_fare = joblib.load(MODELS_DIR / "xgboost_fare.pkl")
        if (MODELS_DIR / "xgboost_duration.pkl").exists():
            self.model_duration = joblib.load(MODELS_DIR / "xgboost_duration.pkl")

    def preprocess_input(self, input_data: TripInput) -> pd.DataFrame:
        """Transforms API input to model features."""
        data = {
            'PULocationID': [input_data.pickup_location_id],
            'DOLocationID': [input_data.dropoff_location_id],
            'trip_distance': [input_data.trip_distance],
            'passenger_count': [input_data.passenger_count],
            'pickup_hour': [input_data.pickup_datetime.hour],
            'pickup_dayofweek': [input_data.pickup_datetime.weekday()]
        }
        return pd.DataFrame(data)

    def predict(self, input_data: TripInput):
        if not self.model_fare or not self.model_duration:
            raise RuntimeError("Models not loaded. Train models first.")
            
        X = self.preprocess_input(input_data)
        fare = self.model_fare.predict(X)[0]
        duration = self.model_duration.predict(X)[0]
        
        return {
            "fare": max(2.5, float(fare)), # Minimum NYC fare
            "duration": max(1.0, float(duration))
        }

# Singleton instance
predictor_service = Predictor()
