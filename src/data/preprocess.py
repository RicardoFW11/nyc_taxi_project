import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dummy cleaning function for MVP.
    Just removes obvious nulls - will be enhanced later.
    """
    print(f"Processing data: Input shape: {df.shape}")
    
    # Drop nulls
    df = df.dropna()
    
    # drop duplicates
    df = df.drop_duplicates()
    
    print(f"Processed data: Output shape: {df.shape}")
    return df