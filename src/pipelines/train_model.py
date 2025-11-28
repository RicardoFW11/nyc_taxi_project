import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import (
    BASELINE_MODELS_DIR,
    MODEL_FEATURES,
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_FILE,
    RANDOM_STATE,
    TARGET_FARE,
    TEST_SIZE,
)
from src.evaluation.metrics import calculate_metrics, print_metrics
from src.models.baseline import BaselineModel


def main():
    """Train baseline models for fare prediction"""
    print("Training baseline models for fare prediction...")

    # 1. Load processed data
    print("Loading processed data...")
    data_path = PROCESSED_DATA_DIR / PROCESSED_TRAIN_FILE
    df = pd.read_parquet(data_path)
    print(f"Loaded: {df.shape}")

    # 2. Prepare features and target
    print("Preparing features and target...")
    X = df[MODEL_FEATURES]
    y = df[TARGET_FARE]

    # 3. Train/test split
    print(f"Splitting data into (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 4. Train models
    print("Training models...")
    model = BaselineModel(model_type="linear")
    model.fit(X_train, y_train)

    # 5. Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, title="Linear Regression - Fare Prediction")

    # 6. Save model
    print("Saving model...")
    model_path = BASELINE_MODELS_DIR / "linear_fare.pkl"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
