"""
Predictor class for loading model and making predictions
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd

from src.api.schemas import TripRequest
from src.config.settings import BASELINE_MODELS_DIR, MODEL_FEATURES


class FarePredictor:
    """
    Loads trained model and makes fare predictions
    """

    def __init__(self, model_name: str = "linear_fare.pkl"):
        """
        Initialize predictor and load model

        Args:
            model_name: Name of the model file in baseline models directory
        """
        self.model_path = BASELINE_MODELS_DIR / model_name
        self.model = None
        self.model_version = model_name.replace(".pkl", "_v1")
        self.load_model()

    def load_model(self):
        """Load the trained model from disk"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using train_model.py"
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def extract_features(self, trip_data: TripRequest) -> pd.DataFrame:
        """
        Extract and engineer features from trip request

        Args:
            trip_data: TripRequest object with trip information

        Returns:
            DataFrame with features in the correct order for model
        """
        # Parse datetime
        pickup_dt = datetime.strptime(
            trip_data.pickup_datetime, "%Y-%m-%d %H:%M:%S"
        )

        # Extract time-based features
        pickup_hour = pickup_dt.hour
        pickup_day_of_week = pickup_dt.weekday()
        pickup_month = pickup_dt.month

        # Calculate distance (for MVP, just use trip_distance)
        distance_euclidean = trip_data.trip_distance

        # Create feature dictionary matching MODEL_FEATURES order
        features = {
            "VendorID": trip_data.VendorID,
            "passenger_count": trip_data.passenger_count,
            "trip_distance": trip_data.trip_distance,
            "payment_type": trip_data.payment_type,
            "pickup_hour": pickup_hour,
            "pickup_day_of_week": pickup_day_of_week,
            "pickup_month": pickup_month,
            "distance_euclidean": distance_euclidean,
        }

        # Convert to DataFrame with correct column order
        df = pd.DataFrame([features])[MODEL_FEATURES]

        return df, features

    def predict(self, trip_data: TripRequest) -> dict:
        """
        Make fare prediction for a trip

        Args:
            trip_data: TripRequest object with trip information

        Returns:
            Dictionary with prediction and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Extract features
        features_df, features_dict = self.extract_features(trip_data)

        # Make prediction
        prediction = self.model.predict(features_df)[0]

        # Ensure positive fare
        predicted_fare = max(0.0, float(prediction))

        # Create response
        response = {
            "predicted_fare": round(predicted_fare, 2),
            "model_version": self.model_version,
            "prediction_timestamp": datetime.now().isoformat(),
            "input_features": features_dict,
        }

        return response

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
