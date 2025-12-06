"""
Configuration settings for NYC Taxi project.
Unified: Baseline + Advanced.
"""

from pathlib import Path

# ============================================
# PROJECT PATHS
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
ADVANCED_MODELS_DIR = MODELS_DIR / "advanced"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BASELINE_MODELS_DIR, ADVANCED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# DATA CONFIGURATION
# ============================================
RAW_DATA_FILE = "yellow_tripdata_2022-05.parquet"
PROCESSED_TRAIN_FILE = "train_data.parquet"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# ============================================
# FEATURES & TARGET
# ============================================
TARGET_FARE = "fare_amount"
TARGET_DURATION = "trip_duration"

# Unified Feature List
MODEL_FEATURES = [
    'VendorID',
    'passenger_count',
    'trip_distance',
    'payment_type',
    'pickup_hour',
    'pickup_day_of_week',
    'pickup_month',
    'distance_euclidean'
]

# ============================================
# MODEL HYPERPARAMETERS
# ============================================
TEST_SIZE = 0.2

# XGBoost Params
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_jobs": -1
}