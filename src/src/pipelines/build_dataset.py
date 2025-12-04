import pandas as pd
from src.configs.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.features import calculate_duration, clean_data, engineered_features

def main():
    # Load raw data (Example: May 2022)
    input_path = RAW_DATA_DIR / "yellow_tripdata_2022-05.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Data not found at {input_path}. Run download.py first.")
    
    print("Loading raw data...")
    df = pd.read_parquet(input_path)
    
    print("Processing data...")
    df = calculate_duration(df)
    df = clean_data(df)
    df = engineered_features(df)
    
    # Select only columns needed for training
    cols = ['PULocationID', 'DOLocationID', 'trip_distance', 'passenger_count', 
            'pickup_hour', 'pickup_dayofweek', 'fare_amount', 'trip_duration']
    
    df = df[cols].dropna()
    
    output_path = PROCESSED_DATA_DIR / "train_data.parquet"
    df.to_parquet(output_path)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    main()
