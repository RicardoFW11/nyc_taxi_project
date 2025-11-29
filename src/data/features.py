"""
Módulo de ingeniería de características para NYC Yellow Taxi Trip Records.
Transforma datos limpios en características listas para modelado.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

from utils.logging import get_full_logger
from config.paths import LOGGER_NAME, PROCESSED_DATA, FEATURE_DATA

warnings.filterwarnings('ignore')
logger = get_full_logger(name=LOGGER_NAME, log_level="INFO")

class TaxiFeatureEngineer:
    """
    Clase para crear características de machine learning a partir de datos limpios de NYC Taxi.
    
    Responsabilidades:
    - Crear variables temporales
    - Generar ratios y métricas derivadas
    - Categorizar variables continuas
    - Crear variables dummy/one-hot encoding
    - Normalizar/escalar características
    """
    
    def __init__(self, processed_data_path: str = None):
        """
        Inicializar el ingeniero de características.
        
        Args:
            processed_data_path: Ruta al archivo de datos procesados/limpios
        """
        self.processed_data_path = processed_data_path or PROCESSED_DATA
        self.df = None
        self.feature_stats = {}
        
        logger.info("TaxiFeatureEngineer inicializado")
    
    def load_processed_data(self) -> pd.DataFrame:
        """Cargar datos procesados/limpios desde archivo parquet."""
        try:
            logger.info(f"Cargando datos procesados desde: {self.processed_data_path}")
            self.df = pd.read_parquet(self.processed_data_path)
            logger.info(f"Datos cargados: {self.df.shape[0]:,} filas, {self.df.shape[1]} columnas")
            return self.df
        except Exception as e:
            logger.error(f"Error cargando datos procesados: {e}")
            raise
    
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Crear características temporales avanzadas.
        """
        logger.info("Creando características temporales...")
        
        if 'tpep_pickup_datetime' in self.df.columns:
            # Features temporales básicas
            self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
            self.df['pickup_day_of_week'] = self.df['tpep_pickup_datetime'].dt.dayofweek
            self.df['pickup_day'] = self.df['tpep_pickup_datetime'].dt.day
            self.df['pickup_month'] = self.df['tpep_pickup_datetime'].dt.month
            
            # Características de día de la semana
            self.df['is_weekend'] = (self.df['pickup_day_of_week'].isin([5, 6])).astype(int)
            self.df['is_monday'] = (self.df['pickup_day_of_week'] == 0).astype(int)
            self.df['is_friday'] = (self.df['pickup_day_of_week'] == 4).astype(int)
            
            # Categorías de tiempo del día
            self.df['time_of_day'] = pd.cut(
                self.df['pickup_hour'],
                bins=[-1, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
            
            # Horas pico detalladas
            morning_rush = (self.df['pickup_hour'].between(7, 9))
            evening_rush = (self.df['pickup_hour'].between(17, 19))
            
            self.df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
            self.df['is_morning_rush'] = morning_rush.astype(int)
            self.df['is_evening_rush'] = evening_rush.astype(int)
            
            # Categorías horarias más específicas
            conditions = [
                self.df['pickup_hour'].between(0, 5),   # Late night
                self.df['pickup_hour'].between(6, 11),  # Morning
                self.df['pickup_hour'].between(12, 16), # Afternoon
                self.df['pickup_hour'].between(17, 20), # Evening
                self.df['pickup_hour'].between(21, 23)  # Night
            ]
            choices = ['Late_Night', 'Morning', 'Afternoon', 'Evening', 'Night']
            self.df['detailed_time_category'] = np.select(conditions, choices, default='Unknown')
        
        if not 'trip_duration_minutes' in self.df.columns:
            if all(col in self.df.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
                self.df['trip_duration_minutes'] = (
                    (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 60
                )
        
        # Duración del viaje si existe
        if 'trip_duration_minutes' in self.df.columns:
            # Categorizar duración
            self.df['duration_category'] = pd.cut(
                self.df['trip_duration_minutes'],
                bins=[0, 5, 15, 30, 60, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            
            # Indicadores de duración
            self.df['is_very_short_trip'] = (self.df['trip_duration_minutes'] <= 5).astype(int)
            self.df['is_long_trip'] = (self.df['trip_duration_minutes'] >= 30).astype(int)
        
        logger.info("Características temporales creadas")
        return self.df
    
    def create_distance_features(self) -> pd.DataFrame:
        """
        Crear características relacionadas con distancia.
        """
        logger.info("Creando características de distancia...")
        
        if 'trip_distance' in self.df.columns:
            # Categorización de distancia
            self.df['distance_category'] = pd.cut(
                self.df['trip_distance'],
                bins=[0, 1, 3, 7, 15, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            
            # Indicadores de distancia
            self.df['is_short_distance'] = (self.df['trip_distance'] <= 1).astype(int)
            self.df['is_medium_distance'] = (self.df['trip_distance'].between(1, 5)).astype(int)
            self.df['is_long_distance'] = (self.df['trip_distance'] >= 10).astype(int)
            
            # Distancia logarítmica para normalizar distribución
            self.df['log_trip_distance'] = np.log1p(self.df['trip_distance'])
        
        logger.info("Características de distancia creadas")
        return self.df
    
    def create_fare_features(self) -> pd.DataFrame:
        """
        Crear características relacionadas con tarifas y pagos.
        """
        logger.info("Creando características de tarifas...")
        
        # Ratios y métricas derivadas
        if all(col in self.df.columns for col in ['tip_amount', 'fare_amount']):
            # Porcentaje de propina
            self.df['tip_percentage'] = np.where(
                self.df['fare_amount'] > 0,
                (self.df['tip_amount'] / self.df['fare_amount']) * 100,
                0
            )
            
            # Categorizar propinas
            self.df['tip_category'] = pd.cut(
                self.df['tip_percentage'],
                bins=[-0.1, 0, 10, 20, 30, float('inf')],
                labels=['No_Tip', 'Low_Tip', 'Medium_Tip', 'High_Tip', 'Very_High_Tip']
            )
            
            # Indicadores de propina
            self.df['has_tip'] = (self.df['tip_amount'] > 0).astype(int)
            self.df['generous_tipper'] = (self.df['tip_percentage'] >= 20).astype(int)
        
        if all(col in self.df.columns for col in ['fare_amount', 'trip_distance']):
            # Tarifa por milla
            self.df['fare_per_mile'] = np.where(
                self.df['trip_distance'] > 0,
                self.df['fare_amount'] / self.df['trip_distance'],
                0
            )
            
            # Categorizar tarifa por milla
            self.df['fare_per_mile_category'] = pd.cut(
                self.df['fare_per_mile'],
                bins=[0, 3, 6, 10, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
        
        if 'total_amount' in self.df.columns:
            # Categorizar monto total
            self.df['total_amount_category'] = pd.cut(
                self.df['total_amount'],
                bins=[0, 10, 25, 50, 100, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High', 'Extreme']
            )
            
            # Transformación logarítmica para normalizar
            self.df['log_total_amount'] = np.log1p(self.df['total_amount'])
        
        if 'fare_amount' in self.df.columns:
            self.df['log_fare_amount'] = np.log1p(self.df['fare_amount'])
        
        logger.info("Características de tarifas creadas")
        return self.df
    
    def create_speed_features(self) -> pd.DataFrame:
        """
        Crear características de velocidad y eficiencia.
        """
        logger.info("Creando características de velocidad...")
        
        if all(col in self.df.columns for col in ['trip_distance', 'trip_duration_minutes']):
            # Velocidad promedio
            self.df['avg_speed_mph'] = np.where(
                self.df['trip_duration_minutes'] > 0,
                (self.df['trip_distance'] / (self.df['trip_duration_minutes'] / 60)),
                0
            )
            
            # Categorizar velocidad
            self.df['speed_category'] = pd.cut(
                self.df['avg_speed_mph'],
                bins=[0, 5, 15, 25, 40, float('inf')],
                labels=['Very_Slow', 'Slow', 'Medium', 'Fast', 'Very_Fast']
            )
            
            # Indicadores de velocidad
            self.df['is_slow_trip'] = (self.df['avg_speed_mph'] <= 10).astype(int)
            self.df['is_fast_trip'] = (self.df['avg_speed_mph'] >= 30).astype(int)
            
            # Eficiencia del viaje (velocidad vs distancia)
            self.df['trip_efficiency'] = np.where(
                self.df['trip_distance'] > 0,
                self.df['avg_speed_mph'] / self.df['trip_distance'],
                0
            )
        
        logger.info("Características de velocidad creadas")
        return self.df
    
    def create_categorical_features(self) -> pd.DataFrame:
        """
        Crear y mejorar características categóricas.
        """
        logger.info("Creando características categóricas...")
        
        # Mapeos para interpretabilidad
        vendor_mapping = {
            1: "Creative_Mobile", 2: "Curb_Mobility", 
            6: "Myle_Technologies", 7: "Helix"
        }
        
        ratecode_mapping = {
            1: "Standard", 2: "JFK", 3: "Newark",
            4: "Nassau_Westchester", 5: "Negotiated", 
            6: "Group_ride", 99: "Unknown"
        }
        
        payment_mapping = {
            0: "Flex_Fare", 1: "Credit_Card", 2: "Cash", 
            3: "No_Charge", 4: "Dispute", 5: "Unknown", 6: "Voided"
        }
        
        # Aplicar mapeos
        if 'VendorID' in self.df.columns:
            self.df['vendor_name'] = self.df['VendorID'].map(vendor_mapping)
            
        if 'RatecodeID' in self.df.columns:
            self.df['ratecode_name'] = self.df['RatecodeID'].map(ratecode_mapping)
            
            # Indicadores específicos
            self.df['is_airport_trip'] = self.df['RatecodeID'].isin([2, 3]).astype(int)
            self.df['is_jfk_trip'] = (self.df['RatecodeID'] == 2).astype(int)
            self.df['is_newark_trip'] = (self.df['RatecodeID'] == 3).astype(int)
            self.df['is_standard_rate'] = (self.df['RatecodeID'] == 1).astype(int)
        
        if 'payment_type' in self.df.columns:
            self.df['payment_name'] = self.df['payment_type'].map(payment_mapping)
            
            # Indicadores de pago
            self.df['is_credit_card'] = (self.df['payment_type'] == 1).astype(int)
            self.df['is_cash_payment'] = (self.df['payment_type'] == 2).astype(int)
            self.df['is_no_charge'] = (self.df['payment_type'] == 3).astype(int)
        
        if 'passenger_count' in self.df.columns:
            # Categorizar número de pasajeros
            self.df['passenger_category'] = pd.cut(
                self.df['passenger_count'],
                bins=[0, 1, 2, 4, float('inf')],
                labels=['Single', 'Couple', 'Small_Group', 'Large_Group']
            )
            
            self.df['is_single_passenger'] = (self.df['passenger_count'] == 1).astype(int)
            self.df['is_group_trip'] = (self.df['passenger_count'] >= 3).astype(int)
        
        if 'store_and_fwd_flag' in self.df.columns:
            self.df['is_store_forward'] = (self.df['store_and_fwd_flag'] == 'Y').astype(int)
        
        logger.info("Características categóricas creadas")
        return self.df
    
    def create_location_features(self) -> pd.DataFrame:
        """
        Crear características relacionadas con ubicaciones.
        """
        logger.info("Creando características de ubicación...")
        
        if all(col in self.df.columns for col in ['PULocationID', 'DOLocationID']):
            # Indicador de viaje circular (mismo pickup y dropoff)
            self.df['is_round_trip'] = (
                self.df['PULocationID'] == self.df['DOLocationID']
            ).astype(int)
            
            # Crear identificador único de ruta
            self.df['route_id'] = (
                self.df['PULocationID'].astype(str) + '_to_' + 
                self.df['DOLocationID'].astype(str)
            )
            
            # Popularidad de ubicaciones (frecuencia)
            pickup_counts = self.df['PULocationID'].value_counts()
            dropoff_counts = self.df['DOLocationID'].value_counts()
            
            self.df['pickup_popularity'] = self.df['PULocationID'].map(pickup_counts)
            self.df['dropoff_popularity'] = self.df['DOLocationID'].map(dropoff_counts)
            
            # Categorizar popularidad
            self.df['pickup_popularity_category'] = pd.cut(
                self.df['pickup_popularity'],
                bins=[0, 100, 500, 2000, float('inf')],
                labels=['Rare', 'Uncommon', 'Common', 'Very_Common']
            )
        
        logger.info("Características de ubicación creadas")
        return self.df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Crear características de interacción entre variables.
        """
        logger.info("Creando características de interacción...")
        
        # Interacciones temporales
        if all(col in self.df.columns for col in ['is_weekend', 'is_rush_hour']):
            self.df['weekend_rush'] = (
                self.df['is_weekend'] & self.df['is_rush_hour']
            ).astype(int)
        
        # Interacciones de pago y propina
        if all(col in self.df.columns for col in ['is_credit_card', 'has_tip']):
            self.df['credit_card_with_tip'] = (
                self.df['is_credit_card'] & self.df['has_tip']
            ).astype(int)
        
        # Interacciones de distancia y tiempo
        if all(col in self.df.columns for col in ['is_long_distance', 'is_long_trip']):
            self.df['long_distance_long_time'] = (
                self.df['is_long_distance'] & self.df['is_long_trip']
            ).astype(int)
        
        # Interacciones de aeropuerto y tiempo
        if all(col in self.df.columns for col in ['is_airport_trip', 'time_of_day']):
            self.df['airport_morning'] = (
                self.df['is_airport_trip'] & (self.df['time_of_day'] == 'Morning')
            ).astype(int)
            
            self.df['airport_evening'] = (
                self.df['is_airport_trip'] & (self.df['time_of_day'] == 'Evening')
            ).astype(int)
        
        logger.info("Características de interacción creadas")
        return self.df
    
    def create_statistical_features(self) -> pd.DataFrame:
        """
        Crear características estadísticas agregadas.
        """
        logger.info("Creando características estadísticas...")
        
        # Agregaciones por hora del día
        if 'pickup_hour' in self.df.columns:
            hourly_stats = self.df.groupby('pickup_hour').agg({
                'fare_amount': ['mean', 'std'],
                'trip_distance': ['mean', 'std'],
                'tip_percentage': 'mean'
            }).round(2)
            
            # Flatten column names
            hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]
            
            # Merge back to main dataframe
            for col in hourly_stats.columns:
                self.df[f'hourly_{col}'] = self.df['pickup_hour'].map(
                    hourly_stats[col]
                )
        
        # Agregaciones por día de la semana
        if 'pickup_day_of_week' in self.df.columns:
            daily_stats = self.df.groupby('pickup_day_of_week').agg({
                'fare_amount': 'mean',
                'trip_distance': 'mean',
                'avg_speed_mph': 'mean'
            }).round(2)
            
            for col in daily_stats.columns:
                self.df[f'daily_avg_{col}'] = self.df['pickup_day_of_week'].map(
                    daily_stats[col]
                )
        
        logger.info("Características estadísticas creadas")
        return self.df
    
    def encode_categorical_variables(self) -> pd.DataFrame:
        """
        Aplicar encoding a variables categóricas para ML.
        """
        logger.info("Aplicando encoding a variables categóricas...")
        
        # Variables categóricas ordinales (mantener como numéricas)
        ordinal_mappings = {
            'distance_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'duration_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'speed_category': {'Very_Slow': 1, 'Slow': 2, 'Medium': 3, 'Fast': 4, 'Very_Fast': 5},
            'tip_category': {'No_Tip': 0, 'Low_Tip': 1, 'Medium_Tip': 2, 'High_Tip': 3, 'Very_High_Tip': 4},
            'total_amount_category': {'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4, 'Extreme': 5}
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in self.df.columns:
                self.df[f'{col}_encoded'] = self.df[col].map(mapping)
        
        # One-hot encoding para variables categóricas nominales
        nominal_columns = [
            'time_of_day', 'detailed_time_category', 'vendor_name', 
            'ratecode_name', 'payment_name', 'passenger_category'
        ]
        
        for col in nominal_columns:
            if col in self.df.columns:
                # Crear dummies
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
        
        logger.info("Encoding de variables categóricas completado")
        return self.df
    
    def create_feature_summary(self) -> dict:
        """
        Crear resumen de las características generadas.
        """
        if self.df is None:
            return {}
        
        # Categorizar tipos de columnas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Binary/indicator columns (solo 0s y 1s)
        binary_cols = []
        for col in numeric_cols:
            if set(self.df[col].dropna().unique()).issubset({0, 1}):
                binary_cols.append(col)
        
        summary = {
            'total_columns': len(self.df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'binary_indicators': len(binary_cols),
            'column_types': {
                'numeric': numeric_cols,
                'categorical': categorical_cols,
                'datetime': datetime_cols,
                'binary_indicators': binary_cols
            },
            'data_shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return summary
    
    def feature_engineering_pipeline(self) -> pd.DataFrame:
        """
        Ejecutar pipeline completo de ingeniería de características.
        
        Returns:
            pd.DataFrame: Dataset con características para ML
        """
        logger.info("=== INICIANDO PIPELINE DE INGENIERÍA DE CARACTERÍSTICAS ===")
        
        # 1. Cargar datos procesados
        self.load_processed_data()
        
        # 2. Crear características temporales
        self.create_temporal_features()
        
        # 3. Crear características de distancia
        self.create_distance_features()
        
        # 4. Crear características de tarifas
        self.create_fare_features()
        
        # 5. Crear características de velocidad
        self.create_speed_features()
        
        # 6. Crear características categóricas
        self.create_categorical_features()
        
        # 7. Crear características de ubicación
        self.create_location_features()
        
        # 8. Crear características de interacción
        self.create_interaction_features()
        
        # 9. Crear características estadísticas
        self.create_statistical_features()
        
        # 10. Encoding de variables categóricas
        self.encode_categorical_variables()
        
        # Resumen final
        summary = self.create_feature_summary()
        logger.info("=== INGENIERÍA DE CARACTERÍSTICAS COMPLETADA ===")
        logger.info(f"Características totales: {summary['total_columns']}")
        logger.info(f"Variables numéricas: {summary['numeric_columns']}")
        logger.info(f"Variables categóricas: {summary['categorical_columns']}")
        logger.info(f"Indicadores binarios: {summary['binary_indicators']}")
        
        self.feature_stats = summary
        return self.df
    
    def save_feature_data(self, output_path: str = None) -> str:
        """
        Guardar dataset con características en archivo parquet.
        
        Args:
            output_path: Ruta de salida. Si None, usa FEATURE_DATA de config.
            
        Returns:
            str: Ruta del archivo guardado
        """
        if self.df is None:
            raise ValueError("No hay datos con características para guardar.")
        
        output_path = output_path or FEATURE_DATA
        
        # Crear directorio si no existe
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar datos
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Dataset con características guardado en: {output_path}")
        logger.info(f"Tamaño del archivo: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path


def main():
    """Función principal para ejecutar la ingeniería de características."""
    try:
        # Inicializar ingeniero de características
        feature_engineer = TaxiFeatureEngineer()
        
        # Ejecutar pipeline completo
        feature_data = feature_engineer.feature_engineering_pipeline()
        
        # Guardar datos con características
        output_path = feature_engineer.save_feature_data()
        
        # Mostrar resumen
        summary = feature_engineer.feature_stats
        print("\n" + "="*60)
        print("RESUMEN DE INGENIERÍA DE CARACTERÍSTICAS")
        print("="*60)
        print(f"Total de características: {summary['total_columns']}")
        print(f"Variables numéricas: {summary['numeric_columns']}")
        print(f"Variables categóricas: {summary['categorical_columns']}")
        print(f"Indicadores binarios: {summary['binary_indicators']}")
        print(f"Tamaño del dataset: {summary['data_shape']}")
        print(f"Archivo guardado: {output_path}")
        
        return feature_data, summary
        
    except Exception as e:
        logger.error(f"Error en ingeniería de características: {e}")
        raise


if __name__ == "__main__":
    main()
