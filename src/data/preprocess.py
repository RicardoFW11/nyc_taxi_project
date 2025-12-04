
import pandas as pd
import numpy as np

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning:
    1. Handles non-numeric fares
    2. Removes negative/zero fares
    3. Casts timestamps to datetime
    """
    # 1. Force fare_amount to numeric (handles 'Unknown', '$12', etc.)
    # errors='coerce' turns bad data into NaN
    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    
    # 2. Drop rows with invalid fares (NaN, negative, or zero)
    # We keep only positive fares
    df = df[df['fare_amount'] > 0].copy()
    
    # 3. Convert timestamps
    # This is critical for calculating duration later
    if 'tpep_pickup_datetime' in df.columns:
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        
    # 4. Filter Passenger Count (Standard taxi rules: 1-6 passengers)
    # If the column exists, we clean it
    if 'passenger_count' in df.columns:
        df = df[df['passenger_count'].between(1, 6)]
        
    return df