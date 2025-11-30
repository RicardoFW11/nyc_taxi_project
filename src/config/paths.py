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
TRAIN_DATA_FARE = str(PROCESSED_DIR / "train_data_fare.parquet")
TEST_DATA_FARE = str(PROCESSED_DIR / "test_data_fare.parquet")
VAL_DATA_FARE = str(PROCESSED_DIR / "val_data_fare.parquet")

TRAIN_DATA_DURATION = str(PROCESSED_DIR / "train_data_duration.parquet")
TEST_DATA_DURATION = str(PROCESSED_DIR / "test_data_duration.parquet")
VAL_DATA_DURATION = str(PROCESSED_DIR / "val_data_duration.parquet")

FARE_MODEL_DATA_FILE = str(PROCESSED_DIR / "fare_model_data.pkl")
DURATION_MODEL_DATA_FILE = str(PROCESSED_DIR / "duration_model_data.pkl")
FARE_FEATURE_SCORES_FILE = str(PROCESSED_DIR / "fare_feature_scores.csv")
DURATION_FEATURE_SCORES_FILE = str(PROCESSED_DIR / "duration_feature_scores.csv")

# Logging configuration
LOGGER_NAME = "nyc_taxi_logger"

# Trained models output path
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline"
ADVANCED_MODEL_PATH = PROJECT_ROOT / "models" / "advanced"

# Create directories if they do not exist
def ensure_directories():
    """Create necessary directories if they do not exist."""
    for directory in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
