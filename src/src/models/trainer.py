import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.configs.paths import PROCESSED_DATA_DIR, MODELS_DIR
from src.configs.settings import XGB_PARAMS

class TaxiTrainer:
    def __init__(self):
        self.features = ['PULocationID', 'DOLocationID', 'trip_distance', 
                         'passenger_count', 'pickup_hour', 'pickup_dayofweek']
        self.model_fare = xgb.XGBRegressor(**XGB_PARAMS)
        self.model_duration = xgb.XGBRegressor(**XGB_PARAMS)

    def load_data(self):
        path = PROCESSED_DATA_DIR / "train_data.parquet"
        return pd.read_parquet(path)

    def train(self):
        df = self.load_data()
        X = df[self.features]
        y_fare = df['fare_amount']
        y_duration = df['trip_duration']

        # Train Fare Model
        print("Training Fare Model...")
        self.model_fare.fit(X, y_fare)
        
        # Train Duration Model
        print("Training Duration Model...")
        self.model_duration.fit(X, y_duration)

        # Evaluation (Simple holdout for demo)
        print(f"Fare MAE: {mean_absolute_error(y_fare, self.model_fare.predict(X)):.2f}")
        print(f"Duration MAE: {mean_absolute_error(y_duration, self.model_duration.predict(X)):.2f}")

        self.save_models()

    def save_models(self):
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.model_fare, MODELS_DIR / "xgboost_fare.pkl")
        joblib.dump(self.model_duration, MODELS_DIR / "xgboost_duration.pkl")
        print("Models saved successfully.")

if __name__ == "__main__":
    trainer = TaxiTrainer()
    trainer.train()
