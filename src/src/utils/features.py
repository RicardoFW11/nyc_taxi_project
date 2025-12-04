import numpy as np
import pandas as pd
from typing import Tuple

def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates trip duration in minutes from timestamps."""
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes outliers and invalid data points."""
    # Filter duration between 1 minute and 60 minutes
    mask_duration = (df['trip_duration'] >= 1) & (df['trip_duration'] <= 60)
    
    # Filter reasonable fare amounts
    mask_fare = (df['fare_amount'] > 0) & (df['fare_amount'] < 300)
    
    # Filter valid distances
    mask_distance = (df['trip_distance'] > 0) & (df['trip_distance'] < 100)
    
    return df[mask_duration & mask_fare & mask_distance].copy()

def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts time-based features."""
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
    return df
