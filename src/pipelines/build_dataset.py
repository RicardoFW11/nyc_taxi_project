import pandas as pd
import os
import sys

# Add the project root to system path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess import clean_raw_data
from src.data.features import add_features

# CONSTANTS
RAW_DATA_PATH = 'data/raw/taxi_data.csv'
PROCESSED_DATA_PATH = 'data/processed/training_data.parquet'
CHUNK_SIZE = 100000  # Process 100k rows at a time to save RAM

def run_etl_pipeline():
    print(f"🚀 Starting ETL Pipeline...")
    print(f"Reading from: {RAW_DATA_PATH}")
    
    # Initialize a list to hold processed chunks
    processed_chunks = []
    
    # Read the file in chunks
    try:
        chunk_iterator = pd.read_csv(RAW_DATA_PATH, chunksize=CHUNK_SIZE)
        
        for i, chunk in enumerate(chunk_iterator):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            # 1. Clean
            cleaned_chunk = clean_raw_data(chunk)
            
            # 2. Add Features
            featured_chunk = add_features(cleaned_chunk)
            
            # 3. Optimize Types (to save memory)
            # Convert float64 to float32 where possible
            cols = featured_chunk.select_dtypes(include=['float64']).columns
            featured_chunk[cols] = featured_chunk[cols].astype('float32')
            
            # Add to list
            processed_chunks.append(featured_chunk)
            
    except FileNotFoundError:
        print(f"\nError: File not found at {RAW_DATA_PATH}")
        return

    print(f"\nAll chunks processed. Merging...")
    
    # Combine all chunks into one DataFrame
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"💾 Saving processed data ({len(final_df)} rows)...")
    
    # Save as Parquet (This will be much smaller than 5GB, likely ~500MB)
    final_df.to_parquet(PROCESSED_DATA_PATH, index=False)
    
    print(f"✨ Done! Saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    run_etl_pipeline()