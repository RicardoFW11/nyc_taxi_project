import os
import sys
import logging
import pandas as pd
import joblib

from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.download import download_data
from data.preprocess import TaxiDataPreprocessor
from data.features import TaxiFeatureEngineer
from data.data_splitter import DataSplitter
from config.paths import FARE_MODEL_DATA_FILE, DURATION_MODEL_DATA_FILE, \
    FARE_FEATURE_SCORES_FILE, DURATION_FEATURE_SCORES_FILE, LOGGER_NAME
    
from utils.logging import LoggerFactory

# Configure logging

logger = LoggerFactory.create_logger(
            name=LOGGER_NAME,
            log_level='DEBUG',
            console_output=True,
            file_output=False
        )

def build_complete_dataset():
    """
    Complete pipeline to build the NYC taxi dataset.
    Executes: download -> preprocess -> feature engineering
    """
    try:
        logger.info("Starting complete dataset build pipeline...")
        
        # step 1: Setup directories
        
        project_root = Path(__file__).parent.parent.parent
        raw_data_dir = project_root / "data" / "raw"
        processed_data_dir = project_root / "data" / "processed"
        
        # Ensure directories exist
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Download data
        logger.info("Step 2: Downloading raw data...")
        download_success = download_data()
        if download_success > 0:
            logger.error("Data download failed. Stopping pipeline.")
            return False
        
        # Step 3: Preprocess data
        logger.info("Step 3: Preprocessing data...")
        # Ejecutar pipeline de limpieza
        preprocessor = TaxiDataPreprocessor()
        cleaned_data = preprocessor.preprocess_full_pipeline(
            remove_outliers=True,
            outlier_method='iqr',
            outlier_factor=1.5
        )
        if cleaned_data is None or cleaned_data.empty:
            logger.error("Data preprocessing failed. Stopping pipeline.")
            return False
        
        # Guardar datos limpios
        _ = preprocessor.save_processed_data()
        
        # Step 4: Feature engineering
        logger.info("Step 4: Creating features...")
        # Inicializar ingeniero de características
        feature_engineer = TaxiFeatureEngineer()
        
        # Ejecutar pipeline completo
        _ = feature_engineer.feature_engineering_pipeline()
        
        # Guardar datos con características
        output_path = feature_engineer.save_feature_data()
        
        # Step 5: Data splitting for models
        logger.info("Step 5: Splitting data for model training...")
        print("🚀 PREPARING DATA FOR MODEL TRAINING")
        print("="*50)
    
        # Cargar dataset con features engineered
        print("📥 Loading feature-engineered dataset...")
        df = pd.read_parquet(output_path)
        print(f"   Dataset shape: {df.shape}")
        data_splitter = DataSplitter()
        data_splits = data_splitter.split_data_for_both_models(df)
        
        # Mostrar información del split
        split_info = data_splits['split_info']
        print(f"\n📊 DATA SPLIT INFORMATION:")
        print(f"   Total samples: {split_info['total_samples']:,}")
        print(f"   Training: {split_info['train_samples']:,} ({split_info['train_ratio']:.1%})")
        print(f"   Validation: {split_info['val_samples']:,} ({split_info['val_ratio']:.1%})")
        print(f"   Test: {split_info['test_samples']:,} ({split_info['test_ratio']:.1%})")
        
        # Información sobre features seleccionadas
        print(f"\n🎯 FEATURE SELECTION RESULTS:")
        
        # Fare model features
        fare_features = data_splits['fare_model']['features']
        print(f"\n💰 FARE AMOUNT MODEL:")
        print(f"   Selected features: {len(fare_features)}")
        print(f"   Top 10 features:")
        for i, feature in enumerate(fare_features[:10], 1):
            print(f"      {i:2d}. {feature}")
        
        # Duration model features
        duration_features = data_splits['duration_model']['features']
        print(f"\n⏱️ TRIP DURATION MODEL:")
        print(f"   Selected features: {len(duration_features)}")
        print(f"   Top 10 features:")
        for i, feature in enumerate(duration_features[:10], 1):
            print(f"      {i:2d}. {feature}")
        
        # Features en común
        common_features = set(fare_features) & set(duration_features)
        print(f"\n🔗 SHARED FEATURES:")
        print(f"   Common features: {len(common_features)}")
        print(f"   Examples: {list(common_features)[:5]}")
        
        print(f"\n💾 SAVING PREPARED DATA:")
        # Guardar datos para fare model
        joblib.dump(data_splits['fare_model'], FARE_MODEL_DATA_FILE)
        print(f"   ✅ Fare model data: {FARE_MODEL_DATA_FILE}")
        
        # Guardar datos para duration model
        joblib.dump(data_splits['duration_model'], DURATION_MODEL_DATA_FILE)
        print(f"   ✅ Duration model data: {DURATION_MODEL_DATA_FILE}")
        
        # Guardar feature scores para análisis
        fare_scores = data_splits['fare_model']['feature_scores']
        duration_scores = data_splits['duration_model']['feature_scores']
        
        fare_scores.to_csv(FARE_FEATURE_SCORES_FILE, index=False)
        duration_scores.to_csv(DURATION_FEATURE_SCORES_FILE, index=False)
        
        print(f"   ✅ Feature importance scores saved")
        
        logger.info("Complete dataset build pipeline finished successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        return False

def main():
    """Main execution function"""
    success = build_complete_dataset()
    if success:
        print("✅ Dataset build completed successfully!")
        sys.exit(0)
    else:
        print("❌ Dataset build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()