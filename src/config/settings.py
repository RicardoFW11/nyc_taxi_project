# Parámetros de preprocesamiento basados en hallazgos del EDA
PREPROCESSING_PARAMS = {
    # Filtros temporales
    'target_year': 2022,
    'target_month': 5,
    'min_trip_duration_minutes': 1,
    'max_trip_duration_minutes': 180,  # 3 horas
    
    # Filtros de pasajeros
    'min_passengers': 1,
    'max_passengers': 6,
    
    # Filtros monetarios
    'min_fare_amount': 0.01,
    'min_trip_distance': 0.01,
    
    # Outliers (basado en hallazgos EDA: 10-12% outliers por IQR)
    'outlier_method': 'iqr',
    'outlier_factor': 1.5,
    'remove_extreme_outliers': True,
    
    # Validación de campos categóricos
    'valid_vendor_ids': [1, 2, 6, 7],  # Según diccionario TLC
    'valid_ratecode_ids': [1, 2, 3, 4, 5, 6, 99],
    'valid_payment_types': [0, 1, 2, 3, 4, 5, 6],
    'valid_store_flags': ['Y', 'N'],
    
    # Límites para campos específicos (basado en percentiles del EDA)
    'max_fare_amount_percentile': 0.99,  # Filtrar 1% superior
    'max_total_amount_percentile': 0.99,
    'max_trip_distance_percentile': 0.99,
    
    # Features a crear
    'create_temporal_features': True,
    'create_ratio_features': True,
    'create_categorical_indicators': True,
    'create_distance_categories': True,
    
    # Año y mes de los datos
    'data_year': 2022,
    'data_month': 5,
}