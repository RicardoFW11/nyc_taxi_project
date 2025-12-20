"""
Módulo de ingeniería de características para NYC Yellow Taxi Trip Records.
Transforma datos limpios en características listas para modelado.

Esta clase está diseñada para ser usada tanto en el pipeline de entrenamiento (batch)
como en la predicción en tiempo real (una sola fila).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import os
import sys

# La clase requiere las configuraciones, por lo que las importamos (asumiendo que están disponibles)
try:
    # Añadir src al path para la ejecución independiente/local
    if 'src' not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    from src.utils.logging import get_full_logger
    from src.config.paths import LOGGER_NAME, PROCESSED_DATA, FEATURE_DATA
    from src.config.settings import LOG_LEVEL, PREPROCESSING_PARAMS # Requerimos PREPROCESSING_PARAMS del nuevo settings
    
    logger = get_full_logger(name=LOGGER_NAME, log_level=LOG_LEVEL)
except ImportError:
    # Fallback si se ejecuta sin la estructura completa de logging/config
    warnings.warn("Using fallback logging/config structure. Ensure project paths are set.")
    class DummyLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def warning(self, msg): pass
    logger = DummyLogger()

warnings.filterwarnings('ignore')


class TaxiFeatureEngineer:
    """
    Clase para crear características de Machine Learning a partir de datos limpios de NYC Taxi.
    """
    
    def __init__(self, processed_data_path: str = None):
        """
        Inicializa el ingeniero de características.
        """
        self.processed_data_path = processed_data_path or PROCESSED_DATA
        self.df = None
        self.feature_stats = {}
        
        logger.info("TaxiFeatureEngineer initialized")
    
    def load_processed_data(self) -> pd.DataFrame:
        """Carga datos limpios del archivo parquet (usado en pipeline de entrenamiento)."""
        try:
            logger.info(f"Loading processed data from: {self.processed_data_path}")
            self.df = pd.read_parquet(self.processed_data_path)
            logger.info(f"Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características temporales avanzadas.
        """
        self.df = df
        logger.info("Creating temporal features...")
        
        if 'tpep_pickup_datetime' in self.df.columns and not self.df['tpep_pickup_datetime'].empty:
            # Asegura que las columnas de datetime sean tipo datetime
            self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'], errors='coerce')
            
            # Características temporales básicas
            self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
            self.df['pickup_day_of_week'] = self.df['tpep_pickup_datetime'].dt.dayofweek
            
            # Días de la semana
            self.df['is_weekend'] = (self.df['pickup_day_of_week'].isin([5, 6])).astype(int)
            
            # Categorías de tiempo de día
            bins_tod = [-1, 6, 12, 18, 24]
            labels_tod = ['Night', 'Morning', 'Afternoon', 'Evening']
            self.df['time_of_day'] = pd.cut(self.df['pickup_hour'], bins=bins_tod, labels=labels_tod, right=False)
            
            if not self.df['time_of_day'].empty:
                self.df['time_of_day'] = self.df['time_of_day'].cat.set_categories(labels_tod)
            # Horas pico (Rush hours: 7-9 AM, 5-7 PM)
            morning_rush = (self.df['pickup_hour'].between(7, 9))
            evening_rush = (self.df['pickup_hour'].between(17, 19))
            
            self.df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
            self.df['is_morning_rush'] = morning_rush.astype(int)
            self.df['is_evening_rush'] = evening_rush.astype(int)
        
        # Calcular duración si no existe (solo si ambas columnas están presentes)
        if 'trip_duration_minutes' not in self.df.columns and all(col in self.df.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
            self.df['tpep_dropoff_datetime'] = pd.to_datetime(self.df['tpep_dropoff_datetime'], errors='coerce')
            self.df['trip_duration_minutes'] = (
                (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 60
            )
        
        if 'trip_duration_minutes' in self.df.columns:
            # Categorizar duración
            bins_duration = [0, 5, 15, 30, 60, float('inf')]
            labels_duration = ['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            self.df['duration_category'] = pd.cut(self.df['trip_duration_minutes'], bins=bins_duration, labels=labels_duration, right=False)
            
            if not self.df['duration_category'].empty:
                 # Establecer explícitamente las categorías posibles.
                 self.df['duration_category'] = self.df['duration_category'].cat.set_categories(labels_duration)

            self.df['is_very_short_trip'] = (self.df['trip_duration_minutes'] <= 5).astype(int)
            self.df['is_long_trip'] = (self.df['trip_duration_minutes'] >= 30).astype(int)
            
        logger.info("Temporal features created")
        return self.df
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características relacionadas con la distancia.
        """
        self.df = df
        logger.info("Creating distance features...")
        
        if 'trip_distance' in self.df.columns:
            # Categorización de distancia
            bins_distance = [0, 1, 3, 7, 15, float('inf')]
            labels_distance = ['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            self.df['distance_category'] = pd.cut(self.df['trip_distance'], bins=bins_distance, labels=labels_distance, right=False)
            
            self.df['is_short_distance'] = (self.df['trip_distance'] <= 1).astype(int)
            self.df['is_long_distance'] = (self.df['trip_distance'] >= 10).astype(int)
            
            # Distancia logarítmica
            self.df['log_trip_distance'] = np.log1p(self.df['trip_distance'])
            
        logger.info("Distance features created")
        return self.df
    
    def create_fare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características relacionadas con la tarifa y el pago.
        """
        self.df = df
        logger.info("Creating fare features...")
        
        if all(col in self.df.columns for col in ['fare_amount', 'trip_distance']) and (self.df['trip_distance'] > 0).any():
            # Tarifa por milla (protección contra división por cero)
            self.df['fare_per_mile'] = np.where(
                self.df['trip_distance'] > 0,
                self.df['fare_amount'] / self.df['trip_distance'],
                0
            )
        
        if 'tip_amount' in self.df.columns and 'fare_amount' in self.df.columns and (self.df['fare_amount'] > 0).any():
             # Porcentaje de propina
            self.df['tip_percentage'] = np.where(
                self.df['fare_amount'] > 0,
                (self.df['tip_amount'] / self.df['fare_amount']) * 100,
                0
            )

        # Logarithmic transformation of fare
        if 'fare_amount' in self.df.columns:
            self.df['log_fare_amount'] = np.log1p(self.df['fare_amount'])
            
        logger.info("Fare features created")
        return self.df
    
    def create_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de velocidad y eficiencia.
        """
        self.df = df
        logger.info("Creating speed features...")
        
        if all(col in self.df.columns for col in ['trip_distance', 'trip_duration_minutes']) and (self.df['trip_duration_minutes'] > 0).any():
            # Velocidad promedio (MPH)
            self.df['avg_speed_mph'] = np.where(
                self.df['trip_duration_minutes'] > 0,
                (self.df['trip_distance'] / (self.df['trip_duration_minutes'] / 60)),
                0
            )
            
            # Categorizar velocidad
            bins_speed = [0, 5, 15, 25, 40, float('inf')]
            labels_speed = ['Very_Slow', 'Slow', 'Medium', 'Fast', 'Very_Fast']
            self.df['speed_category'] = pd.cut(self.df['avg_speed_mph'], bins=bins_speed, labels=labels_speed, right=False)
            
            # Indicadores de velocidad
            self.df['is_slow_trip'] = (self.df['avg_speed_mph'] <= 10).astype(int)
            self.df['is_fast_trip'] = (self.df['avg_speed_mph'] >= 30).astype(int)
            
            # Eficiencia del viaje (velocidad vs distancia)
            if (self.df['trip_distance'] > 0).any():
                 self.df['trip_efficiency'] = np.where(
                    self.df['trip_distance'] > 0,
                    self.df['avg_speed_mph'] / self.df['trip_distance'],
                    0
                )
            
        logger.info("Speed features created")
        return self.df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea y mejora las características categóricas.
        """
        self.df = df
        logger.info("Creating categorical features...")
        
        ratecode_mapping = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau_Westchester", 5: "Negotiated", 6: "Group_ride", 99: "Unknown"}
        payment_mapping = {0: "Flex_Fare", 1: "Credit_Card", 2: "Cash", 3: "No_Charge", 4: "Dispute", 5: "Unknown", 6: "Voided"}

        RATECODE_CATEGORIES = list(ratecode_mapping.values())
        PAYMENT_CATEGORIES = list(payment_mapping.values())

        if 'PULocationID' in self.df.columns:
            self.df['PULocationID'] = self.df['PULocationID'].astype(str)
        if 'DOLocationID' in self.df.columns:
            self.df['DOLocationID'] = self.df['DOLocationID'].astype(str)

        EXPECTED_LOCATION_CATEGORIES = [str(i) for i in range(1, 266)]
        EXPECTED_LOCATION_CATEGORIES.append('0') # Incluimos '0' para manejar valores fuera de rango/missing
        
        if 'PULocationID' in self.df.columns:
             self.df['PULocationID'] = pd.to_numeric(self.df['PULocationID'], errors='coerce').fillna(0).astype(int).astype(str)
             self.df['PULocationID'] = pd.Categorical(
                self.df['PULocationID'], 
                categories=EXPECTED_LOCATION_CATEGORIES
             )
        if 'DOLocationID' in self.df.columns:
             self.df['DOLocationID'] = pd.to_numeric(self.df['DOLocationID'], errors='coerce').fillna(0).astype(int).astype(str)
             self.df['DOLocationID'] = pd.Categorical(
                self.df['DOLocationID'], 
                categories=EXPECTED_LOCATION_CATEGORIES
             )

        EXPECTED_VENDORS = ['CMT', 'VTS', 'TPEV', 'Unknown'] # Asegura que la lista sea exhaustiva de tu dataset
        if 'vendor_id' in self.df.columns and self.df['vendor_id'].dtype.name in ['object', 'category']:
            # Esto maneja el caso donde el ID es float(NaN) o float(0)
            self.df['vendor_id'] = self.df['vendor_id'].astype(str).str.upper().replace('NAN', 'UNKNOWN')
            
            # 2. Rellenar cualquier NaN/None restante
            self.df['vendor_id'] = self.df['vendor_id'].fillna('Unknown')
            
            # 3. Forzar la conversión a Categorical con las categorías esperadas
            self.df['vendor_id'] = pd.Categorical(self.df['vendor_id'], categories=EXPECTED_VENDORS)

        EXPECTED_FWD_FLAGS = ['Y', 'N'] 
        if 'store_and_fwd_flag' in self.df.columns and self.df['store_and_fwd_flag'].dtype.name in ['object', 'category']:
            # Si se encuentra '0' u otra cosa, se convertirá en NaN y evitará el error 500.
            self.df['store_and_fwd_flag'] = self.df['store_and_fwd_flag'].fillna('N')
            self.df['store_and_fwd_flag'] = pd.Categorical(self.df['store_and_fwd_flag'], categories=EXPECTED_FWD_FLAGS)

        if 'RatecodeID' in self.df.columns:
            self.df['RatecodeID'] = self.df['RatecodeID'].astype(str)
            self.df['RatecodeID'] = pd.to_numeric(self.df['RatecodeID'], errors='coerce').fillna(99).astype(int)
            self.df['ratecode_name'] = self.df['RatecodeID'].map(ratecode_mapping)
            self.df['ratecode_name'] = self.df['ratecode_name'].fillna('Unknown')
            self.df['ratecode_name'] = self.df['ratecode_name'].astype('category').cat.set_categories(RATECODE_CATEGORIES)
            
            self.df['is_standard_rate'] = (self.df['RatecodeID'] == 1).astype(int)
            self.df['is_airport_trip'] = self.df['RatecodeID'].isin([2, 3]).astype(int)
        
        if 'payment_type' in self.df.columns:
            self.df['payment_type'] = pd.to_numeric(self.df['payment_type'], errors='coerce').fillna(99).astype(int)
            self.df['payment_name'] = self.df['payment_type'].map(payment_mapping)
            self.df['payment_name'] = self.df['payment_name'].fillna('Unknown')
            self.df['payment_name'] = self.df['payment_name'].astype('category').cat.set_categories(PAYMENT_CATEGORIES)
            self.df['is_credit_card'] = (self.df['payment_type'] == 1).astype(int)
        
        if 'store_and_fwd_flag' in self.df.columns:
            EXPECTED_FWD_FLAGS = ['Y', 'N']
            self.df['store_and_fwd_flag'] = pd.Categorical(self.df['store_and_fwd_flag'], categories=EXPECTED_FWD_FLAGS)
            self.df['is_store_forward'] = (self.df['store_and_fwd_flag'] == 'Y').astype(int)
            
        logger.info("Categorical features created")
        return self.df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características relacionadas con la ubicación.
        """
        self.df = df
        logger.info("Creating location features...")
        
        if all(col in self.df.columns for col in ['PULocationID', 'DOLocationID']):
            # Viaje circular (Misma recogida y destino)
            self.df['is_round_trip'] = (
                self.df['PULocationID'] == self.df['DOLocationID']
            ).astype(int)
            
            # Nota: Las características de popularidad/agregación (e.g., hourly_fare_amount_mean)
            # se omiten en el contexto de inferencia de la API porque requieren estadísticas del dataset completo.
            
        logger.info("Location features created")
        return self.df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de interacción entre variables.
        """
        self.df = df
        logger.info("Creating interaction features...")
        
        # Interacción temporal (asumiendo que las features existen del paso 1)
        if all(col in self.df.columns for col in ['is_weekend', 'is_rush_hour']):
            self.df['weekend_rush'] = (
                self.df['is_weekend'] & self.df['is_rush_hour']
            ).astype(int)
        
        # Interacción distancia/tiempo
        if all(col in self.df.columns for col in ['is_long_distance', 'is_long_trip']):
            self.df['long_distance_long_time'] = (
                self.df['is_long_distance'] & self.df['is_long_trip']
            ).astype(int)
        
        logger.info("Interaction features created")
        return self.df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características estadísticas agregadas (usadas solo en entrenamiento).
        
        Nota: Este método debe usarse solo durante el entrenamiento, ya que
        requiere estadísticas de todo el dataset (o un diccionario precalculado).
        En la API, se debe omitir o usar valores por defecto.
        """
        self.df = df
        logger.warning("Skipping real-time statistical feature creation (requires batch context).")
        return self.df # Devuelve el DF sin cambios en la API
        
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica codificación one-hot y ordinal a variables categóricas.
        
        Nota: Esto debe ser compatible con las features que esperan los modelos.
        """
        self.df = df
        logger.info("Applying encoding to categorical variables...")

        # Definiciones para la codificación ordinal (si aplica)
        ordinal_mappings = {
            'distance_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'duration_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'speed_category': {'Very_Slow': 1, 'Slow': 2, 'Medium': 3, 'Fast': 4, 'Very_Fast': 5},
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in self.df.columns:
                # Usar codificación ordinal si el campo existe
                temp_col = self.df[col].astype(object)
                self.df[f'{col}_encoded'] = temp_col.map(mapping).fillna(0).astype(int)
                
        # One-hot encoding para variables nominales
        nominal_columns = [
            'time_of_day', 'ratecode_name', 'payment_name', 
            'PULocationID', 'DOLocationID', 'vendor_id', 'store_and_fwd_flag'
        ]
        
        # Aplicamos one-hot y luego eliminamos las categorías base/no codificadas
        for col in nominal_columns:
            if col in self.df.columns:
                #and self.df[col].dtype.name in ['category', 'object']
                #dummies = pd.get_dummies(self.df[col], prefix=col, dummy_na=False)
                # Concatenamos y eliminamos el campo original
                #self.df = pd.concat([self.df, dummies], axis=1).drop(columns=[col])
                self.df[col] = self.df[col].astype(object).fillna('Unknown')
                dummies = pd.get_dummies(self.df[col], prefix=col, dummy_na=False)
                self.df = pd.concat([self.df, dummies], axis=1).drop(columns=[col])

        logger.info("Categorical variable encoding completed")
        return self.df
    
    # El resto de los métodos (create_feature_summary, feature_engineering_pipeline, save_feature_data) 
    # se usan solo en el pipeline de entrenamiento batch y no en la API.

    def feature_engineering_pipeline(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline.
        
        Returns:
            pd.DataFrame: Dataset with features for ML
        """
        
        logger.info("=== STARTING FEATURE ENGINEERING PIPELINE ===")
        self.df = df
        # 1. Load processed data
        if df is None:
            df = self.load_processed_data()
        
        # 2. Create temporal features
        df = self.create_temporal_features(df)
        
        # 3. Create distance features
        df = self.create_distance_features(df)
        
        # 4. Create fare features
        df = self.create_fare_features(df)
        
        # 5. Create speed features
        df = self.create_speed_features(df)
        
        # 6. Create categorical features
        df = self.create_categorical_features(df)
        
        # 7. Create location features
        df = self.create_location_features(df)
        
        # 8. Create interaction features
        df = self.create_interaction_features(df)
        
        # 9. Create statistical features
        df = self.create_statistical_features(df)
        
        # 10. Encode categorical variables
        df = self.encode_categorical_variables(df)
        
        # Final summary
        summary = self.create_feature_summary()
        logger.info("=== FEATURE ENGINEERING COMPLETED ===")
        logger.info(f"Total features: {summary['total_columns']}")
        logger.info(f"Numeric variables: {summary['numeric_columns']}")
        logger.info(f"Categorical variables: {summary['categorical_columns']}")
        logger.info(f"Binary indicators: {summary['binary_indicators']}")
        
        self.feature_stats = summary
        return df
    
    def create_feature_summary(self):
        # Implementación mínima temporal para evitar el AttributeError
        return {
            'total_columns': len(self.df.columns) if self.df is not None else 0,
            'numeric_columns': 0, # Implementar lógica real si es necesario
            'categorical_columns': 0,
            'binary_indicators': 0,
            'data_shape': self.df.shape if self.df is not None else (0, 0),
        }
    
    def save_feature_data(self, output_path: str = None) -> str:
        """
        Save dataset with features to a parquet file.
        
        Args:
            output_path: Output path. If None, uses FEATURE_DATA from config.
            
        Returns:
            str: Path of the saved file
        """
        if self.df is None:
            raise ValueError("No feature data to save.")
        
        output_path = output_path or FEATURE_DATA
        
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Feature dataset saved at: {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path

# --- Función main para el pipeline de batch (si se ejecuta localmente) ---
def main():
    """Función principal para el feature engineering en modo batch."""
    try:
        feature_engineer = TaxiFeatureEngineer()
        
        # Ejecutar pipeline completo (solo se ejecuta en el script build_dataset.py)
        feature_data = feature_engineer.feature_engineering_pipeline() 
        
        # Guardar feature data
        output_path = feature_engineer.save_feature_data()
        
# Show summary
        summary = feature_engineer.feature_stats
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total features: {summary['total_columns']}")
        print(f"Numeric variables: {summary['numeric_columns']}")
        print(f"Categorical variables: {summary['categorical_columns']}")
        print(f"Binary indicators: {summary['binary_indicators']}")
        print(f"Dataset size: {summary['data_shape']}")
        print(f"Saved file: {output_path}")
        
        return feature_data, summary
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise        

if __name__ == "__main__":
    # Solo se ejecuta si se llama directamente (para testing/batch)
    # En el entorno Docker, se llama desde build_dataset.py
    main()