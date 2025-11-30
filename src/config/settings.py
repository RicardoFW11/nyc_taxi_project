#Preprocesing Parameters based on EDA findings
PREPROCESSING_PARAMS = {
    # Temporal filters
    'target_year': 2022,
    'target_month': 5,
    'min_trip_duration_minutes': 1,
    'max_trip_duration_minutes': 180,  # 3 hours
    
    # Passenger filters
    'min_passengers': 1,
    'max_passengers': 6,
    
    # Monetary filters
    'min_fare_amount': 0.01,
    'min_trip_distance': 0.01,
    
    # Outliers (based on EDA findings: 10-12% outliers by IQR)
    'outlier_method': 'iqr',
    'outlier_factor': 1.5,
    'remove_extreme_outliers': True,
    
    # Validation of categorical fields
    'valid_vendor_ids': [1, 2, 6, 7],  # According to TLC dictionary
    'valid_ratecode_ids': [1, 2, 3, 4, 5, 6, 99],
    'valid_payment_types': [0, 1, 2, 3, 4, 5, 6],
    'valid_store_flags': ['Y', 'N'],
    
    # LLimits for specific fields (based on EDA percentiles)
    'max_fare_amount_percentile': 0.99,  # Filter top 1%
    'max_total_amount_percentile': 0.99,
    'max_trip_distance_percentile': 0.99,
    
    # Features to create
    'create_temporal_features': True,
    'create_ratio_features': True,
    'create_categorical_indicators': True,
    'create_distance_categories': True,
    
    # Year and month of the data
    'data_year': 2022,
    'data_month': 5,
}

# General parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
TRAIN_SIZE = 0.7
LOG_LEVEL = "DEBUG"