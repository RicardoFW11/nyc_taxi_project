"""
Módulo de Ingeniería de Características (Feature Engineering) para NYC Taxi Data.

Este módulo centraliza la transformación de datos crudos limpios en vectores de características
optimizados para el modelado predictivo. Implementa estrategias de transformación temporal,
geoespacial y categórica, asegurando consistencia tanto en el entrenamiento (batch) como
en la inferencia en tiempo real.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import os
import sys

# Configuración del entorno de ejecución y logging
try:
    # Ajuste dinámico del path para permitir ejecución como script independiente
    if 'src' not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    from src.utils.logging import get_full_logger
    from src.config.paths import LOGGER_NAME, PROCESSED_DATA, FEATURE_DATA
    from src.config.settings import LOG_LEVEL, PREPROCESSING_PARAMS
    
    logger = get_full_logger(name=LOGGER_NAME, log_level=LOG_LEVEL)
except ImportError:
    # Mecanismo de fallback para entornos donde la estructura del proyecto no esté completa
    warnings.warn("Using fallback logging/config structure. Ensure project paths are set.")
    class DummyLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def warning(self, msg): pass
    logger = DummyLogger()

warnings.filterwarnings('ignore')


class TaxiFeatureEngineer:
    """
    Orquestador del proceso de ingeniería de características.
    
    Responsable de aplicar transformaciones deterministas sobre el dataset de taxis.
    Encapsula la lógica de negocio para derivar nuevas variables predictivas a partir
    de los datos transaccionales básicos.
    """
    
    def __init__(self, processed_data_path: str = None):
        """
        Inicializa el procesador de características.

        Args:
            processed_data_path (str, optional): Ruta al archivo de datos preprocesados.
                                                 Por defecto utiliza la ruta definida en configuración.
        """
        self.processed_data_path = processed_data_path or PROCESSED_DATA
        self.df = None
        self.feature_stats = {}
        
        logger.info("TaxiFeatureEngineer initialized")
    
    def load_processed_data(self) -> pd.DataFrame:
        """
        Recupera el dataset limpio desde el almacenamiento persistente (Parquet).
        Utilizado principalmente en el flujo de entrenamiento batch.
        """
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
        Genera variables derivadas del componente temporal del viaje.
        
        Extrae patrones cíclicos (hora, día de la semana) y categóricos (fin de semana, hora pico)
        esenciales para capturar la estacionalidad de la demanda y el tráfico.
        """
        self.df = df
        logger.info("Creating temporal features...")
        
        if 'tpep_pickup_datetime' in self.df.columns and not self.df['tpep_pickup_datetime'].empty:
            # Normalización del tipo de dato datetime
            self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'], errors='coerce')
            
            # Extracción de componentes temporales básicos
            self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
            self.df['pickup_day_of_week'] = self.df['tpep_pickup_datetime'].dt.dayofweek
            
            # Indicador binario para fines de semana (Sábado=5, Domingo=6)
            self.df['is_weekend'] = (self.df['pickup_day_of_week'].isin([5, 6])).astype(int)
            
            # Segmentación del día en franjas horarias operativas
            bins_tod = [-1, 6, 12, 18, 24]
            labels_tod = ['Night', 'Morning', 'Afternoon', 'Evening']
            self.df['time_of_day'] = pd.cut(self.df['pickup_hour'], bins=bins_tod, labels=labels_tod, right=False)
            
            if not self.df['time_of_day'].empty:
                self.df['time_of_day'] = self.df['time_of_day'].cat.set_categories(labels_tod)
            
            # Identificación de horas pico (Rush Hour) basada en patrones de tráfico de NYC
            morning_rush = (self.df['pickup_hour'].between(7, 9))
            evening_rush = (self.df['pickup_hour'].between(17, 19))
            
            self.df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
            self.df['is_morning_rush'] = morning_rush.astype(int)
            self.df['is_evening_rush'] = evening_rush.astype(int)
        
        # Cálculo de la duración del viaje para entrenamiento (Target Calculation)
        if 'trip_duration_minutes' not in self.df.columns and all(col in self.df.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
            self.df['tpep_dropoff_datetime'] = pd.to_datetime(self.df['tpep_dropoff_datetime'], errors='coerce')
            self.df['trip_duration_minutes'] = (
                (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 60
            )
        
        # Categorización de la duración para análisis estratificado
        if 'trip_duration_minutes' in self.df.columns:
            bins_duration = [0, 5, 15, 30, 60, float('inf')]
            labels_duration = ['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            self.df['duration_category'] = pd.cut(self.df['trip_duration_minutes'], bins=bins_duration, labels=labels_duration, right=False)
            
            if not self.df['duration_category'].empty:
                 self.df['duration_category'] = self.df['duration_category'].cat.set_categories(labels_duration)

            self.df['is_very_short_trip'] = (self.df['trip_duration_minutes'] <= 5).astype(int)
            self.df['is_long_trip'] = (self.df['trip_duration_minutes'] >= 30).astype(int)
            
        logger.info("Temporal features created")
        return self.df
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece el dataset con métricas derivadas de la distancia del viaje.
        Aplica transformaciones logarítmicas para normalizar la distribución de distancias
        y crea categorías de rango para capturar comportamientos no lineales.
        """
        self.df = df
        logger.info("Creating distance features...")
        
        if 'trip_distance' in self.df.columns:
            # Segmentación de distancias
            bins_distance = [0, 1, 3, 7, 15, float('inf')]
            labels_distance = ['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            self.df['distance_category'] = pd.cut(self.df['trip_distance'], bins=bins_distance, labels=labels_distance, right=False)
            
            self.df['is_short_distance'] = (self.df['trip_distance'] <= 1).astype(int)
            self.df['is_long_distance'] = (self.df['trip_distance'] >= 10).astype(int)
            
            # Transformación logarítmica (Log1p) para reducir el sesgo de la distribución
            self.df['log_trip_distance'] = np.log1p(self.df['trip_distance'])
            
        logger.info("Distance features created")
        return self.df
    
    def create_fare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas económicas derivadas.
        Nota: Estas características (especialmente las que usan fare_amount) deben manejarse
        con cuidado para evitar 'Data Leakage' si se usan como predictores del target.
        """
        self.df = df
        logger.info("Creating fare features...")
        
        # Cálculo de eficiencia económica (Tarifa por milla)
        if all(col in self.df.columns for col in ['fare_amount', 'trip_distance']) and (self.df['trip_distance'] > 0).any():
            self.df['fare_per_mile'] = np.where(
                self.df['trip_distance'] > 0,
                self.df['fare_amount'] / self.df['trip_distance'],
                0
            )
        
        # Análisis del comportamiento de propinas
        if 'tip_amount' in self.df.columns and 'fare_amount' in self.df.columns and (self.df['fare_amount'] > 0).any():
            self.df['tip_percentage'] = np.where(
                self.df['fare_amount'] > 0,
                (self.df['tip_amount'] / self.df['fare_amount']) * 100,
                0
            )

        # Transformación logarítmica del objetivo (útil para normalización en regresión)
        if 'fare_amount' in self.df.columns:
            self.df['log_fare_amount'] = np.log1p(self.df['fare_amount'])
            
        logger.info("Fare features created")
        return self.df
    
    def create_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deriva métricas de velocidad y eficiencia operativa.
        Estas métricas son fundamentales para entender la congestión y el flujo del tráfico.
        """
        self.df = df
        logger.info("Creating speed features...")
        
        if all(col in self.df.columns for col in ['trip_distance', 'trip_duration_minutes']) and (self.df['trip_duration_minutes'] > 0).any():
            # Cálculo de velocidad media del viaje (Millas por hora)
            self.df['avg_speed_mph'] = np.where(
                self.df['trip_duration_minutes'] > 0,
                (self.df['trip_distance'] / (self.df['trip_duration_minutes'] / 60)),
                0
            )
            
            # Categorización semántica de la velocidad
            bins_speed = [0, 5, 15, 25, 40, float('inf')]
            labels_speed = ['Very_Slow', 'Slow', 'Medium', 'Fast', 'Very_Fast']
            self.df['speed_category'] = pd.cut(self.df['avg_speed_mph'], bins=bins_speed, labels=labels_speed, right=False)
            
            # Indicadores de extremos de velocidad (tráfico pesado vs autopista libre)
            self.df['is_slow_trip'] = (self.df['avg_speed_mph'] <= 10).astype(int)
            self.df['is_fast_trip'] = (self.df['avg_speed_mph'] >= 30).astype(int)
            
            # Eficiencia de desplazamiento (inversa de la congestión)
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
        Procesa y normaliza variables categóricas nominales.
        Asegura que los IDs de ubicación y códigos de tarifa sean tratados como categorías
        y maneja la imputación de valores desconocidos o nulos.
        """
        self.df = df
        logger.info("Creating categorical features...")
        
        # Mapeos estándar del diccionario de datos de TLC
        ratecode_mapping = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau_Westchester", 5: "Negotiated", 6: "Group_ride", 99: "Unknown"}
        payment_mapping = {0: "Flex_Fare", 1: "Credit_Card", 2: "Cash", 3: "No_Charge", 4: "Dispute", 5: "Unknown", 6: "Voided"}

        RATECODE_CATEGORIES = list(ratecode_mapping.values())
        PAYMENT_CATEGORIES = list(payment_mapping.values())

        # Normalización de IDs de ubicación (PULocationID, DOLocationID)
        if 'PULocationID' in self.df.columns:
            self.df['PULocationID'] = self.df['PULocationID'].astype(str)
        if 'DOLocationID' in self.df.columns:
            self.df['DOLocationID'] = self.df['DOLocationID'].astype(str)

        EXPECTED_LOCATION_CATEGORIES = [str(i) for i in range(1, 266)]
        EXPECTED_LOCATION_CATEGORIES.append('0') # Categoría para valores fuera de rango o desconocidos
        
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

        # Normalización de Proveedores (VendorID)
        EXPECTED_VENDORS = ['CMT', 'VTS', 'TPEV', 'Unknown'] 
        if 'vendor_id' in self.df.columns and self.df['vendor_id'].dtype.name in ['object', 'category']:
            self.df['vendor_id'] = self.df['vendor_id'].astype(str).str.upper().replace('NAN', 'UNKNOWN')
            self.df['vendor_id'] = self.df['vendor_id'].fillna('Unknown')
            self.df['vendor_id'] = pd.Categorical(self.df['vendor_id'], categories=EXPECTED_VENDORS)

        # Flag de almacenamiento y reenvío (Store and Forward)
        EXPECTED_FWD_FLAGS = ['Y', 'N'] 
        if 'store_and_fwd_flag' in self.df.columns and self.df['store_and_fwd_flag'].dtype.name in ['object', 'category']:
            self.df['store_and_fwd_flag'] = self.df['store_and_fwd_flag'].fillna('N')
            self.df['store_and_fwd_flag'] = pd.Categorical(self.df['store_and_fwd_flag'], categories=EXPECTED_FWD_FLAGS)

        # Mapeo de códigos de tarifa (RateCodeID)
        if 'RatecodeID' in self.df.columns:
            self.df['RatecodeID'] = self.df['RatecodeID'].astype(str)
            self.df['RatecodeID'] = pd.to_numeric(self.df['RatecodeID'], errors='coerce').fillna(99).astype(int)
            self.df['ratecode_name'] = self.df['RatecodeID'].map(ratecode_mapping)
            self.df['ratecode_name'] = self.df['ratecode_name'].fillna('Unknown')
            self.df['ratecode_name'] = self.df['ratecode_name'].astype('category').cat.set_categories(RATECODE_CATEGORIES)
            
            self.df['is_standard_rate'] = (self.df['RatecodeID'] == 1).astype(int)
            self.df['is_airport_trip'] = self.df['RatecodeID'].isin([2, 3]).astype(int)
        
        # Mapeo de tipos de pago
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
        Genera características topológicas basadas en la geografía del viaje.
        """
        self.df = df
        logger.info("Creating location features...")
        
        if all(col in self.df.columns for col in ['PULocationID', 'DOLocationID']):
            # Detección de viajes locales/circulares (mismo punto de inicio y fin)
            self.df['is_round_trip'] = (
                self.df['PULocationID'] == self.df['DOLocationID']
            ).astype(int)
            
        logger.info("Location features created")
        return self.df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de interacción para capturar efectos combinados.
        Ejemplo: Un viaje largo durante hora pico tiene un comportamiento distinto a uno largo en horario valle.
        """
        self.df = df
        logger.info("Creating interaction features...")
        
        # Interacción Temporal: Fin de semana + Hora Pico
        if all(col in self.df.columns for col in ['is_weekend', 'is_rush_hour']):
            self.df['weekend_rush'] = (
                self.df['is_weekend'] & self.df['is_rush_hour']
            ).astype(int)
        
        # Interacción Espacial-Temporal: Larga distancia + Larga duración
        if all(col in self.df.columns for col in ['is_long_distance', 'is_long_trip']):
            self.df['long_distance_long_time'] = (
                self.df['is_long_distance'] & self.df['is_long_trip']
            ).astype(int)
        
        logger.info("Interaction features created")
        return self.df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder para la creación de características estadísticas agregadas.
        
        Advertencia: Este método está diseñado para el contexto de entrenamiento por lotes (Batch),
        donde se tiene acceso al dataset completo. En inferencia en tiempo real, estas métricas
        deben inyectarse pre-calculadas o utilizar valores por defecto para evitar latencia.
        """
        self.df = df
        logger.warning("Skipping real-time statistical feature creation (requires batch context).")
        return self.df
        
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma variables categóricas en representaciones numéricas aptas para modelos de ML.
        Aplica codificación ordinal para variables con jerarquía y One-Hot Encoding para nominales.
        """
        self.df = df
        logger.info("Applying encoding to categorical variables...")

        # Mapeos para codificación ordinal (preservando el orden de magnitud)
        ordinal_mappings = {
            'distance_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'duration_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'speed_category': {'Very_Slow': 1, 'Slow': 2, 'Medium': 3, 'Fast': 4, 'Very_Fast': 5},
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in self.df.columns:
                temp_col = self.df[col].astype(object)
                self.df[f'{col}_encoded'] = temp_col.map(mapping).fillna(0).astype(int)
                
        # Variables nominales para One-Hot Encoding
        nominal_columns = [
            'time_of_day', 'ratecode_name', 'payment_name', 
            'PULocationID', 'DOLocationID', 'vendor_id', 'store_and_fwd_flag'
        ]
        
        # Aplicación de One-Hot Encoding y eliminación de columnas originales
        for col in nominal_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(object).fillna('Unknown')
                dummies = pd.get_dummies(self.df[col], prefix=col, dummy_na=False)
                self.df = pd.concat([self.df, dummies], axis=1).drop(columns=[col])

        logger.info("Categorical variable encoding completed")
        return self.df
    
    def feature_engineering_pipeline(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Ejecuta el flujo completo de ingeniería de características.
        
        Secuencia de ejecución:
        1. Carga de datos (si no se proporcionan).
        2. Generación de features temporales, espaciales y de tarifa.
        3. Creación de interacciones y transformaciones estadísticas.
        4. Codificación final de variables categóricas.
        
        Args:
            df (pd.DataFrame, optional): DataFrame inicial. Si es None, carga desde disco.
            
        Returns:
            pd.DataFrame: Dataset enriquecido listo para entrenamiento o inferencia.
        """
        
        logger.info("=== STARTING FEATURE ENGINEERING PIPELINE ===")
        self.df = df
        
        # 1. Carga de datos base
        if df is None:
            df = self.load_processed_data()
        
        # 2. Generación de características base
        df = self.create_temporal_features(df)
        df = self.create_distance_features(df)
        df = self.create_fare_features(df)
        df = self.create_speed_features(df)
        
        # 3. Tratamiento de variables categóricas y ubicación
        df = self.create_categorical_features(df)
        df = self.create_location_features(df)
        
        # 4. Características complejas y estadísticas
        df = self.create_interaction_features(df)
        df = self.create_statistical_features(df)
        
        # 5. Codificación final
        df = self.encode_categorical_variables(df)
        
        # Generación de resumen de ejecución
        summary = self.create_feature_summary()
        logger.info("=== FEATURE ENGINEERING COMPLETED ===")
        logger.info(f"Total features: {summary['total_columns']}")
        
        self.feature_stats = summary
        return df
    
    def create_feature_summary(self):
        """
        Genera un resumen estadístico básico de las características creadas.
        Útil para validación y registro de metadatos del pipeline.
        """
        return {
            'total_columns': len(self.df.columns) if self.df is not None else 0,
            'numeric_columns': 0, # Placeholder para implementación futura de conteo detallado
            'categorical_columns': 0,
            'binary_indicators': 0,
            'data_shape': self.df.shape if self.df is not None else (0, 0),
        }
    
    def save_feature_data(self, output_path: str = None) -> str:
        """
        Persiste el dataset procesado en formato Parquet.
        
        Args:
            output_path (str, optional): Ruta de destino. Por defecto usa FEATURE_DATA.
            
        Returns:
            str: Ruta absoluta del archivo generado.
        """
        if self.df is None:
            raise ValueError("No feature data to save.")
        
        output_path = output_path or FEATURE_DATA
        
        # Garantiza la existencia del directorio de salida
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Escritura optimizada con compresión gzip
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Feature dataset saved at: {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path

# --- Punto de entrada para ejecución Batch ---
def main():
    """
    Función principal para la ejecución del pipeline en modo Batch.
    Invocada típicamente por orquestadores de datos (ej. Airflow) o scripts de build.
    """
    try:
        feature_engineer = TaxiFeatureEngineer()
        
        # Ejecución del pipeline de transformación
        feature_data = feature_engineer.feature_engineering_pipeline() 
        
        # Persistencia de resultados
        output_path = feature_engineer.save_feature_data()
        
        # Reporte final en consola
        summary = feature_engineer.feature_stats
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total features: {summary['total_columns']}")
        print(f"Dataset size: {summary['data_shape']}")
        print(f"Saved file: {output_path}")
        
        return feature_data, summary
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise        

if __name__ == "__main__":
    # Bloque de ejecución directa para pruebas o procesos batch independientes.
    main()