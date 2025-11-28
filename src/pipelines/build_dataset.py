"""
Data preparation pipeline - MVP version
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config.settings import (
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_FILE,
    RANDOM_STATE,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    SAMPLE_SIZE,
)
from src.data.features import engineer_features
from src.data.preprocess import clean_data


def main():
    """Build preprocessed dataset"""
    print("Building dataset...")

    # 1. Load raw data
    print("Loading raw data...")
    raw_data_path = RAW_DATA_DIR / RAW_DATA_FILE
    df = pd.read_parquet(raw_data_path)
    print(f"Loaded {df.shape[0]} rows")

    # 2. Sample data
    if SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} rows...")
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
        print(f"Sampled: {df.shape[0]} rows")

    # 3. Clean data
    print("Cleaning data...")
    df = clean_data(df)

    # 4. Engineer features
    print("Engineering features...")
    df = engineer_features(df)

    # 5. Save processed data
    print("Saving processed data...")
    processed_data_path = PROCESSED_DATA_DIR / PROCESSED_TRAIN_FILE
    df.to_parquet(processed_data_path, index=False)
    print(f"Saved {df.shape[0]} rows to {processed_data_path}")


if __name__ == "__main__":
    main()
