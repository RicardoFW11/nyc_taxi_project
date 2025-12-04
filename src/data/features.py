import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features to the dataframe:
    1. trip_duration_min
    2. hour_of_day
    3. day_of_week
    """
    # 1. Calculate Duration in Minutes
    df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # 2. Filter Impossible Durations (Based on your EDA)
    # We remove trips < 1 min (cancelled?) and > 300 mins (5 hours - likely errors)
    df = df[df['trip_duration_min'].between(1, 300)]
    
    # 3. Extract Time Features (Very useful for ML models later)
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    
    return df