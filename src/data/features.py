import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the model.
    """
    print("Engineering features...")

    # Extract basic time features
    df["pickup_hour"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.hour
    df["pickup_day_of_week"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    ).dt.dayofweek
    df["pickup_month"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.month
    
    df["distance_euclidean"] = df["trip_distance"]

    print("Engineered features successfully")
    return df
