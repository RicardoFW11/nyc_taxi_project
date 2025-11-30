"""
Path configuration and constants for the NYC Taxi project.
"""

from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# URLs and data files
DOWNLOAD_URL_TAXI_DATA = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-05.parquet"
RAW_DATA = str(RAW_DIR / "yellow_tripdata_2022-05.parquet")
PROCESSED_DATA = str(PROCESSED_DIR / "processed_data.parquet")
FEATURE_DATA = str(PROCESSED_DIR / "feature_data.parquet")

# Output files from preprocessing
TRAIN_DATA = str(PROCESSED_DIR / "train_data.parquet")
TEST_DATA = str(PROCESSED_DIR / "test_data.parquet")
VAL_DATA = str(PROCESSED_DIR / "val_data.parquet")

# Logging configuration
LOGGER_NAME = "nyc_taxi_logger"

# Create directories if they do not exist
def ensure_directories():
    """Create necessary directories if they do not exist."""
    for directory in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
