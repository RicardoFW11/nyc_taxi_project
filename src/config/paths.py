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
FEATURE_DATA = str(PROCESSED_DIR / "feature_data.parquet")

# Archivos de salida del preprocesamiento
TRAIN_DATA = str(PROCESSED_DIR / "train_data.parquet")
TEST_DATA = str(PROCESSED_DIR / "test_data.parquet")

# Configuración de logging
LOGGER_NAME = "nyc_taxi_logger"

# Crear directorios si no existen
def ensure_directories():
    """Crear directorios necesarios si no existen."""
    for directory in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
