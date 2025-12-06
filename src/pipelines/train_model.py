import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import (
    BASELINE_MODELS_DIR,
    ADVANCED_MODELS_DIR,
    MODEL_FEATURES,
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_FILE,
    RANDOM_STATE,
    TARGET_FARE,
    TEST_SIZE,
)
from src.evaluation.metrics import calculate_metrics, print_metrics
from src.models.baseline import BaselineModel
from src.models.advanced import XGBoostModel

def main(model_type):
    """Train model based on type (linear or xgboost)"""
    print(f"🚀 Starting training pipeline for: {model_type.upper()}")

    # 1. Load data
    data_path = PROCESSED_DATA_DIR / PROCESSED_TRAIN_FILE
    if not data_path.exists():
        raise FileNotFoundError("Processed data not found. Run build_dataset.py first.")
    
    df = pd.read_parquet(data_path)
    
    # 2. Prepare features
    X = df[MODEL_FEATURES]
    y = df[TARGET_FARE]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 4. Select and Train Model
    if model_type == "linear":
        model = BaselineModel()
        save_dir = BASELINE_MODELS_DIR
        filename = "linear_fare.pkl"
    elif model_type == "xgboost":
        model = XGBoostModel()
        save_dir = ADVANCED_MODELS_DIR
        filename = "xgboost_fare.pkl"
    else:
        raise ValueError("Unknown model type")

    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, title=f"{model_type.upper()} Performance")

    # 6. Save
    save_path = save_dir / filename
    model.save(save_path)
    print(f"✅ Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "xgboost"], 
                        help="Choose model to train: 'linear' or 'xgboost'")
    args = parser.parse_args()
    
    main(args.model)