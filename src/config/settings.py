"""
Configuration settings for NYC Taxi project.
Single source of truth for all project parameters.
"""

from pathlib import Path
from typing import List

# ============================================
# PROJECT PATHS
# ============================================

# Root directory of the project
PROJECT_ROOT=Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
ADVANCED_MODELS_DIR = MODELS_DIR / "advanced"

# Ensure directories exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    BASELINE_MODELS_DIR,
    ADVANCED_MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# DATA CONFIGURATION
# ============================================

# Dataset files
RAW_DATA_FILE = "yellow_tripdata_2022-05.parquet"
PROCESSED_TRAIN_FILE = "train_data.parquet"
PROCESSED_TEST_FILE = "test_data.parquet"

# Sampling for MVP (set to None to use full dataset)
SAMPLE_SIZE = 100_000  # Use 100k rows for quick iteration
RANDOM_STATE = 42

# ============================================
# FEATURE COLUMNS
# ============================================

# Original columns to keep

FEATURE_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
]

# Target variables

TARGET_FARE = "fare_amount"
TARGET_TRIP_DURATION = "trip_duration"

# Datetime columns

DATETIME_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]

# Features to use from model training (after feature engineering)
MODEL_FEATURES = [
    'VendorID',
    'passenger_count',
    'trip_distance',
    'payment_type',
    'pickup_hour',
    'pickup_day_of_week',
    'pickup_month',
    'distance_euclidean',
]


# ============================================
# MODEL TRAINING CONFIGURATION
# ============================================

TEST_SIZE = 0.2  # Train/test split


FARE_MIN = 1
FARE_MAX = 10000