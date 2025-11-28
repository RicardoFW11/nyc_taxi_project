"""
Configuración de paths y constantes para el proyecto NYC Taxi.
"""

from pathlib import Path
import os

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# URLs y archivos de datos
DOWNLOAD_URL_TAXI_DATA = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-05.parquet"
RAW_DATA = str(RAW_DIR / "yellow_tripdata_2022-05.parquet")
PROCESSED_DATA = str(PROCESSED_DIR / "processed_data.parquet")

# Archivos de salida del preprocesamiento
TRAIN_DATA = str(PROCESSED_DIR / "train_data.parquet")
TEST_DATA = str(PROCESSED_DIR / "test_data.parquet")

# Configuración de logging
LOGGER_NAME = "nyc_taxi_logger"

# Parámetros de preprocesamiento basados en hallazgos del EDA
PREPROCESSING_PARAMS = {
    # Filtros temporales
    'target_year': 2022,
    'target_month': 5,
    'min_trip_duration_minutes': 1,
    'max_trip_duration_minutes': 180,  # 3 horas
    
    # Filtros de pasajeros
    'min_passengers': 1,
    'max_passengers': 6,
    
    # Filtros monetarios
    'min_fare_amount': 0.01,
    'min_trip_distance': 0.01,
    
    # Outliers (basado en hallazgos EDA: 10-12% outliers por IQR)
    'outlier_method': 'iqr',
    'outlier_factor': 1.5,
    'remove_extreme_outliers': True,
    
    # Validación de campos categóricos
    'valid_vendor_ids': [1, 2, 6, 7],  # Según diccionario TLC
    'valid_ratecode_ids': [1, 2, 3, 4, 5, 6, 99],
    'valid_payment_types': [0, 1, 2, 3, 4, 5, 6],
    'valid_store_flags': ['Y', 'N'],
    
    # Límites para campos específicos (basado en percentiles del EDA)
    'max_fare_amount_percentile': 0.99,  # Filtrar 1% superior
    'max_total_amount_percentile': 0.99,
    'max_trip_distance_percentile': 0.99,
    
    # Features a crear
    'create_temporal_features': True,
    'create_ratio_features': True,
    'create_categorical_indicators': True,
    'create_distance_categories': True
}

# Crear directorios si no existen
def ensure_directories():
    """Crear directorios necesarios si no existen."""
    for directory in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
