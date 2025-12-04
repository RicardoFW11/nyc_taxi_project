# Feature Configuration
CATEGORICAL_COLS = ["PULocationID", "DOLocationID"]
NUMERICAL_COLS = ["passenger_count", "trip_distance"]
TARGET_FARE = "fare_amount"
TARGET_DURATION = "trip_duration"

# Model Hyperparameters (Baseline XGBoost)
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_jobs": -1
}
