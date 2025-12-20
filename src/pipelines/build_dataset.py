"""
Complete Data preparation pipeline - Advanced Version
Executes: download -> preprocess -> feature engineering -> data splitting (for fare and duration models)
"""
import os
import sys
import logging
import pandas as pd
import joblib
from pathlib import Path

# Agregar el directorio src al path para importaciones
#sys.path.append(str(Path(__file__).parent.parent))
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except Exception:
    pass    
# Importar módulos avanzados
from src.data.download import download_data
from src.data.preprocess import TaxiDataPreprocessor
from src.data.features import TaxiFeatureEngineer
from src.data.data_splitter import DataSplitter
from src.config.paths import (
    FARE_MODEL_DATA_FILE, 
    DURATION_MODEL_DATA_FILE,
    FARE_FEATURE_SCORES_FILE, 
    DURATION_FEATURE_SCORES_FILE, 
    LOGGER_NAME,
    RAW_DIR,
    PROCESSED_DIR
)
from src.utils.logging import LoggerFactory

# Configurar logging
logger = LoggerFactory.create_logger(
    name=LOGGER_NAME,
    log_level='DEBUG',
    console_output=True,
    file_output=False
)

def build_complete_dataset():
    """
    Complete pipeline to build the NYC taxi dataset.
    Executes: download -> preprocess -> feature engineering -> split & save PKL
    """
    try:
        logger.info("Starting complete dataset build pipeline...")
        
        # Step 1: Setup directories (Ya se hace en paths.py, pero lo aseguramos)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Download data
        logger.info("Step 2: Downloading raw data...")
        download_success = download_data()
        if download_success > 0:
            logger.error("Data download failed. Stopping pipeline.")
            return False
        
        # Step 3: Preprocess data (Usando la clase avanzada)
        logger.info("Step 3: Preprocessing data...")
        preprocessor = TaxiDataPreprocessor()
        cleaned_data = preprocessor.preprocess_full_pipeline(
            remove_outliers=True,
            outlier_method='iqr',
            outlier_factor=1.5
        )
        if cleaned_data is None or cleaned_data.empty:
            logger.error("Data preprocessing failed. Stopping pipeline.")
            return False
        
        # Guardar datos limpios intermedios
        processed_path = preprocessor.save_processed_data()
        
        # Step 4: Feature engineering (Usando la clase avanzada)
        logger.info("Step 4: Creating features...")
        feature_engineer = TaxiFeatureEngineer(processed_data_path=processed_path)
        feature_data = feature_engineer.feature_engineering_pipeline()
        
        # Guardar datos con características
        output_path = feature_engineer.save_feature_data()
        
        # Step 5: Data splitting for models (CRÍTICO: Crea los archivos PKL para el entrenamiento)
        logger.info("Step 5: Splitting data for model training...")
        
        data_splitter = DataSplitter()
        data_splits = data_splitter.split_data_for_both_models(feature_data)
        
        # Guardar datos para fare model (ARCHIVO PKL ESPERADO POR train_model.py)
        joblib.dump(data_splits['fare_model'], FARE_MODEL_DATA_FILE)
        logger.info(f"💾 ✅ Fare model data saved: {FARE_MODEL_DATA_FILE}")
        
        # Guardar datos para duration model (ARCHIVO PKL ESPERADO POR train_model.py)
        joblib.dump(data_splits['duration_model'], DURATION_MODEL_DATA_FILE)
        logger.info(f"💾 ✅ Duration model data saved: {DURATION_MODEL_DATA_FILE}")
        
        # Guardar feature scores
        data_splits['fare_model']['feature_scores'].to_csv(FARE_FEATURE_SCORES_FILE, index=False)
        data_splits['duration_model']['feature_scores'].to_csv(DURATION_FEATURE_SCORES_FILE, index=False)
        
        logger.info("Complete dataset build pipeline finished successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        # Añadimos una excepción más específica para el debug
        print(f"FATAL ETL ERROR: {str(e)}")
        return False

def main():
    """Main execution function"""
    success = build_complete_dataset()
    if success:
        logger.info("✅ Dataset build completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Dataset build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()