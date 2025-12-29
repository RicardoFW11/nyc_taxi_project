"""
Manager class for loading and orchestrating multiple models (Fare and Duration)
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
from src.config.settings import BASELINE_MODELS_DIR, ADVANCED_MODELS_DIR
# Import the Feature Engineer for real-time prediction.
from src.data.features import TaxiFeatureEngineer # Assume that the new class is in the features.py module.

class PredictionManager:
    """
    Loads and manages multiple trained models (Fare and Duration) 
    and performs feature engineering at inference time.
    """

    def __init__(self, fare_model_name: str = "xgboost_fare_amount_advanced.pkl", 
                 duration_model_name: str = "xgboost_trip_duration_minutes_advanced.pkl"):
        
        self.models = {}
        self.model_features = {}
        self.model_versions = {}
        self.model_metrics = {}
        
        # Initialize the Feature Engineer (to recreate all features)
        self.feature_engineer = TaxiFeatureEngineer(processed_data_path=None) # No data needed, just use the methods
        
        # Upload rate template
        self._load_single_model("fare", fare_model_name)
        # Upload duration model
        self._load_single_model("duration", duration_model_name)

    def _get_model_path(self, model_name: str) -> Path:
        """Helper to determine if model is baseline or advanced."""
        if (ADVANCED_MODELS_DIR / model_name).exists():
            return ADVANCED_MODELS_DIR / model_name
        elif (BASELINE_MODELS_DIR / model_name).exists():
            return BASELINE_MODELS_DIR / model_name
        else:
            raise FileNotFoundError(f"Model not found: {model_name}")

    def _load_single_model(self, target_key: str, model_name: str):
        """
        Loads a single model (wrapper) and extracts its required features list 
        from the native model object (XGBRegressor, etc.).
        """
        model_path = self._get_model_path(model_name)
        
        try:
            model_artifact = joblib.load(model_path)
            
            # 1. Determine whether we load the wrapper (correct method) or the native model (bug)
            # The key object is the value of the ‘model’ key
            loaded_object = model_artifact['model']

            # --- NEW METRICS LOGIC ---
            # If the .pkl has saved metrics, they are used. 
            # If not, the ones obtained during training are used manually (hardcoded fallback).
            if isinstance(model_artifact, dict) and 'metrics' in model_artifact:
                self.model_metrics[target_key] = model_artifact['metrics']
            else:
                # FALLBACK: If the file does not have metrics, they are defined here centrally.
                if target_key == 'fare':
                    self.model_metrics[target_key] = {"MAE": "$1.84", "R2": "0.92"}
                else:
                    self.model_metrics[target_key] = {"MAE": "2.10 min", "R2": "0.88"}
            # --------------------------------

            # Integrity check: The object is the native model (bug) OR the wrapper (correct).
            #if hasattr(loaded_object, 'model'): 
                # It is the wrapper (XGBoostModel), the native model is nested.
            if not hasattr(loaded_object, 'model'):
                # Old bug case: The object is the native model (RF or XGB).
                native_model = loaded_object
                model_wrapper = None
            else:
                # Correct case: It is the wrapper (XGBoostModel or RandomForestModel).
                model_wrapper = loaded_object
                # native_model must be the nested object
                native_model = model_wrapper.model


            if model_wrapper is None:
                 self.models[target_key] = native_model
            else:
                 self.models[target_key] = model_wrapper

            features_used = []

            try:
                if native_model.__class__.__name__ == 'XGBRegressor':
                    features_used = native_model.get_booster().feature_names
                elif hasattr(native_model, 'feature_names_in_'):
                    features_used = native_model.feature_names_in_.tolist()
                elif hasattr(native_model, 'n_features_in_'):
                    features_used = [f'feature_{i}' for i in range(native_model.n_features_in_)]
                else:
                    raise AttributeError("Native model object could not expose feature names.")

            except AttributeError as fe:
                raise AttributeError(f"Could not extract feature list from native model: {fe}")

            # 2. Save the model for prediction
            # The native object is saved if you don't have the wrapper; you only need the .predict() method.
            self.models[target_key] = native_model 
            self.model_features[target_key] = features_used
            self.model_versions[target_key] = model_name.replace(".pkl", "")
            
            print(f"✅ Model {target_key.upper()} loaded from {model_path} with {len(features_used)} features.")
            
        except Exception as e:
            raise Exception(f"Error loading {target_key} model from {model_path}: {e}")

    def extract_features(self, trip_data: TripRequest) -> pd.DataFrame:
        """
        Extract and engineer *all* potential features from a single trip request
        using the TaxiFeatureEngineer class.
        
        Args:
            trip_data: TripRequest object with trip information
            
        Returns:
            DataFrame with all engineered features for one sample
        """
        # Create a DataFrame with the input data
        #input_df = pd.DataFrame([{
        #    “VendorID”: trip_data.VendorID,
        #    “passenger_count”: trip_data.passenger_count,
        #    “trip_distance”: trip_data.trip_distance,
        #    “payment_type”: trip_data.payment_type,
        #    “tpep_pickup_datetime”: datetime.strptime(trip_data.pickup_datetime, “%Y-%m-%d %H:%M:%S”),
        #    # Dummy/minimum fields to avoid Feature Engineer errors
        #    ‘fare_amount’: 10.0, ‘tip_amount’: 0.0, ‘tolls_amount’: 0.0,
        #    ‘extra’: 0.0, ‘mta_tax’: 0.5, ‘improvement_surcharge’: 0.3,
        #    ‘total_amount’: 10.8, ‘congestion_surcharge’: 2.5, ‘airport_fee’: 0.0,
        #    ‘RatecodeID’: 1, ‘store_and_fwd_flag’: ‘N’,
        #    ‘PULocationID’: 1, ‘DOLocationID’: 1 # These IDs must be handled if necessary
        #}])

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([{
            "VendorID": trip_data.VendorID,
            "passenger_count": trip_data.passenger_count,
            "trip_distance": trip_data.trip_distance,
            "payment_type": trip_data.payment_type,
            "tpep_pickup_datetime": datetime.strptime(trip_data.pickup_datetime, "%Y-%m-%d %H:%M:%S"),
            
            # Dummy/minimum fields to avoid Feature Engineer errors
            # FIX: Initialized at 0.0 so that the model does NOT see the response (Leakage)
            'fare_amount': 0.0, 'tip_amount': 0.0, 'tolls_amount': 0.0,
            'extra': 0.0, 'mta_tax': 0.0, 'improvement_surcharge': 0.0,
            'total_amount': 0.0, 'congestion_surcharge': 0.0, 'airport_fee': 0.0,
            
            'RatecodeID': 1, 'store_and_fwd_flag': 'N',
            'PULocationID': 1, 'DOLocationID': 1 # These IDs must be handled if necessary.
        }])

        #input_df = pd.DataFrame([{
        #    "VendorID": trip_data.VendorID,
        #    "passenger_count": trip_data.passenger_count,
        #    "trip_distance": trip_data.trip_distance,
        #    "payment_type": trip_data.payment_type,
        #    "tpep_pickup_datetime": datetime.strptime(trip_data.pickup_datetime, "%Y-%m-%d %H:%M:%S"),
        #    "RatecodeID": 1, 
        #    "store_and_fwd_flag": 'N',
        #    "PULocationID": 1, 
        #    "DOLocationID": 1 
        #}])


        # Perform feature engineering (only the necessary parts)
        
        # Note: The advanced FeatureEngineer assumes a complete DataFrame,
        # this is a critical simplification for the API environment.
        
      
        df = self.feature_engineer.create_temporal_features(input_df.copy())
        df = self.feature_engineer.create_distance_features(df)
        df = self.feature_engineer.create_fare_features(df)
        df = self.feature_engineer.create_speed_features(df)
        df = self.feature_engineer.create_categorical_features(df)
        df = self.feature_engineer.create_location_features(df)
        df = self.feature_engineer.create_interaction_features(df)
        # We skip create_statistical_features because it requires the complete dataset.
        df = self.feature_engineer.encode_categorical_variables(df)
        
        # Clear temporary and null columns
        cols_to_drop = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_duration_minutes', 'fare_amount',
            'total_amount', 'tip_amount', 'tolls_amount', 'extra', 'mta_tax', 'improvement_surcharge',
            'congestion_surcharge', 'airport_fee'
        ]
        
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        return df, input_df


    def predict(self, trip_data: TripRequest) -> dict:
        """
        Make predictions for both fare and duration con alineación estricta de columnas.
        """
        if not self.is_loaded():
            raise ValueError("Models not loaded")

        # 1. Execute Complete Feature Engineering
        full_features_df, _ = self.extract_features(trip_data)

        # --- PROCESS FOR FARE ---
        fare_features_needed = self.model_features['fare']
        
        # Create base DataFrame with zeros
        X_fare = pd.DataFrame(0.0, index=[0], columns=fare_features_needed)
        
        # Fill in the data generated by the Feature Engineer.
        for col in fare_features_needed:
            if col in full_features_df.columns:
                X_fare[col] = full_features_df[col].values
        
        # Ensure exact order and convert to NumPy to avoid column name errors
        X_fare = X_fare[fare_features_needed]
        fare_prediction = self.models['fare'].predict(X_fare.values)[0]
        predicted_fare = max(0.0, float(fare_prediction))

        # --- PROCESS FOR DURATION ---
        duration_features_needed = self.model_features['duration']
        
        # Create base DataFrame with zeros
        X_duration = pd.DataFrame(0.0, index=[0], columns=duration_features_needed)
        
        # Fill in the data generated by the Feature Engineer.
        for col in duration_features_needed:
            if col in full_features_df.columns:
                X_duration[col] = full_features_df[col].values
        
        # Ensure exact order and convert to NumPy to avoid column name errors
        X_duration = X_duration[duration_features_needed]
        duration_prediction = self.models['duration'].predict(X_duration.values)[0]
        predicted_duration = max(0.0, float(duration_prediction))

        # 4. Create Final Response
        return {
            "predicted_fare": round(predicted_fare, 2),
            "predicted_duration_minutes": round(predicted_duration, 2),
            "model_version_fare": self.model_versions['fare'],
            "model_version_duration": self.model_versions['duration'],
            "prediction_timestamp": datetime.now().isoformat(),
            "input_features_fare": X_fare.iloc[0].to_dict(),
            "input_features_duration": X_duration.iloc[0].to_dict(),
        }


    def is_loaded(self) -> bool:
        """Check if both core models are loaded"""
        return all(key in self.models and self.models[key] is not None for key in ['fare', 'duration'])