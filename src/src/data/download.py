import os
import requests
from src.configs.paths import RAW_DATA_DIR

def download_taxi_data(year: int, month: int):
    """Downloads NYC Yellow Taxi data (Parquet format)"""
    month_str = f"{month:02d}"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month_str}.parquet"
    filename = RAW_DATA_DIR / f"yellow_tripdata_{year}-{month_str}.parquet"
    
    if filename.exists():
        print(f"File {filename} already exists. Skipping.")
        return

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_taxi_data(2022, 5) # Default to May 2022
