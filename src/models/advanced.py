import joblib
import xgboost as xgb
from src.config.settings import XGB_PARAMS

class XGBoostModel:
    """Wrapper for XGBoost model (Week 2 Goal)"""

    def __init__(self):
        self.model = xgb.XGBRegressor(**XGB_PARAMS)
        self.model_type = "xgboost"
        self.is_fitted = False

    def fit(self, X, y):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        self.model.fit(X, y)
        self.is_fitted = True
        print("Model trained successfully")
        return self

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self.model.predict(X)

    def save(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"Model saved successfully to {path}")

    def load(self, path: str):
        """Load the model"""
        self.model = joblib.load(path)
        self.is_fitted = True
        return self.model