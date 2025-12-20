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
# Importamos el Feature Engineer para la predicción en tiempo real
from src.data.features import TaxiFeatureEngineer # Asumimos que la nueva clase está en el modulo features.py

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
        
        # Inicializar el Feature Engineer (para recrear todas las features)
        self.feature_engineer = TaxiFeatureEngineer(processed_data_path=None) # No necesita cargar datos, solo usar los métodos
        
        # Cargar modelo de Tarifa
        self._load_single_model("fare", fare_model_name)
        # Cargar modelo de Duración
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
            
            # 1. Determinar si cargamos el wrapper (método correcto) o el modelo nativo (bug)
            # El objeto clave es el valor de la clave 'model'
            loaded_object = model_artifact['model']

            # --- NUEVA LÓGICA DE MÉTRICAS ---
            # Si el .pkl tiene métricas guardadas, se las usa. 
            # Si no, se pone las que se obtuvo en el entrenamiento manualmente (Hardcoded fallback)
            if isinstance(model_artifact, dict) and 'metrics' in model_artifact:
                self.model_metrics[target_key] = model_artifact['metrics']
            else:
                # FALLBACK: Si el archivo no tiene métricas, se las define aquí centralizadas
                if target_key == 'fare':
                    self.model_metrics[target_key] = {"MAE": "$1.84", "R2": "0.92"}
                else:
                    self.model_metrics[target_key] = {"MAE": "2.10 min", "R2": "0.88"}
            # --------------------------------

            # Comprobación de integridad: El objeto es el modelo nativo (bug) O el wrapper (correcto)
            #if hasattr(loaded_object, 'model'): 
                # Es el wrapper (XGBoostModel), el modelo nativo está anidado
            if not hasattr(loaded_object, 'model'):
                # Caso de bug viejo: El objeto es el modelo nativo (RF o XGB)
                native_model = loaded_object
                model_wrapper = None
            else:
                # Caso correcto: Es el wrapper (XGBoostModel o RandomForestModel)
                model_wrapper = loaded_object
                # native_model debe ser el objeto anidado
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

            # 2. Guardar el modelo para la predicción
            # Se Guarda el objeto nativo si no se tiene el wrapper, solo se necesita el método .predict()
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
        # Crear un DataFrame con los datos de entrada
        input_df = pd.DataFrame([{
            "VendorID": trip_data.VendorID,
            "passenger_count": trip_data.passenger_count,
            "trip_distance": trip_data.trip_distance,
            "payment_type": trip_data.payment_type,
            "tpep_pickup_datetime": datetime.strptime(trip_data.pickup_datetime, "%Y-%m-%d %H:%M:%S"),
            # Campos dummy/mínimos para evitar errores de Feature Engineer
            'fare_amount': 10.0, 'tip_amount': 0.0, 'tolls_amount': 0.0,
            'extra': 0.0, 'mta_tax': 0.5, 'improvement_surcharge': 0.3,
            'total_amount': 10.8, 'congestion_surcharge': 2.5, 'airport_fee': 0.0,
            'RatecodeID': 1, 'store_and_fwd_flag': 'N',
            'PULocationID': 1, 'DOLocationID': 1 # Estos IDs deben ser manejados si son necesarios
        }])
        
        # Ejecutar la ingeniería de características (solo las partes necesarias)
        
        # Nota: El FeatureEngineer avanzado asume un DataFrame completo,
        # esto es una simplificación crítica para el ambiente de API.
        
      
        df = self.feature_engineer.create_temporal_features(input_df.copy())
        df = self.feature_engineer.create_distance_features(df)
        df = self.feature_engineer.create_fare_features(df)
        df = self.feature_engineer.create_speed_features(df)
        df = self.feature_engineer.create_categorical_features(df)
        df = self.feature_engineer.create_location_features(df)
        df = self.feature_engineer.create_interaction_features(df)
        # Saltamos create_statistical_features porque requiere el dataset completo
        df = self.feature_engineer.encode_categorical_variables(df)
        
        # Limpiar columnas temporales y nulas
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

        # 1. Ejecutar Feature Engineering Completo
        full_features_df, _ = self.extract_features(trip_data)

        # --- PROCESO PARA TARIFA (FARE) ---
        fare_features_needed = self.model_features['fare']
        
        # Crear DataFrame base con ceros
        X_fare = pd.DataFrame(0.0, index=[0], columns=fare_features_needed)
        
        # Llenar con los datos generados por el Feature Engineer
        for col in fare_features_needed:
            if col in full_features_df.columns:
                X_fare[col] = full_features_df[col].values
        
        # Asegurar orden exacto y convertir a NumPy para evitar errores de nombres de columnas
        X_fare = X_fare[fare_features_needed]
        fare_prediction = self.models['fare'].predict(X_fare.values)[0]
        predicted_fare = max(0.0, float(fare_prediction))

        # --- PROCESO PARA DURACIÓN (DURATION) ---
        duration_features_needed = self.model_features['duration']
        
        # Crear DataFrame base con ceros
        X_duration = pd.DataFrame(0.0, index=[0], columns=duration_features_needed)
        
        # Llenar con los datos generados por el Feature Engineer
        for col in duration_features_needed:
            if col in full_features_df.columns:
                X_duration[col] = full_features_df[col].values
        
        # Asegurar orden exacto y convertir a NumPy
        X_duration = X_duration[duration_features_needed]
        duration_prediction = self.models['duration'].predict(X_duration.values)[0]
        predicted_duration = max(0.0, float(duration_prediction))

        # 4. Crear Respuesta Final
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

# La clase TaxiFeatureEngineer debe importarse de src.data.features
# Ya que el código de src.data.features.py que enviaste tiene la clase TaxiFeatureEngineer