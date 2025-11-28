import joblib
from sklearn.linear_model import LinearRegression


class BaselineModel:
    """Simple wrapper for baseline models"""

    def __init__(self, model_type: str = "linear"):
        if model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        self.model_type = model_type
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
        print(f"Making predictions with {self.model_type} model...")
        return self.model.predict(X)

    def save(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"Model saved successfully to {path}")

    def load(self, path: str):
        """Load the model"""
        print(f"Loading {self.model_type} model from {path}...")
        model = joblib.load(path)
        print(f"Model loaded successfully from {path}")
        return model
