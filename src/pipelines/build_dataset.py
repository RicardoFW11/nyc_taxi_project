import os
import sys
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.download import download_data
from data.preprocess import TaxiDataPreprocessor
from data.features import TaxiFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        output_path = preprocessor.save_processed_data()
        
        # Mostrar resumen
        # summary = preprocessor.get_preprocessing_summary()
        # print("\n" + "="*50)
        # print("RESUMEN DE LIMPIEZA DE DATOS")
        # print("="*50)
        # print(f"Registros originales: {summary['original_shape'][0]:,}")
        # print(f"Registros limpios: {summary['final_shape'][0]:,}")
        # print(f"Registros removidos: {summary['total_removed']:,} ({summary['removal_percentage']:.2f}%)")
        # print(f"Archivo de datos limpios: {output_path}")
        # print("\n📋 SIGUIENTE PASO:")
        # print("   Ejecutar features.py para ingeniería de características")
        
        # Step 4: Feature engineering
        logger.info("Step 4: Creating features...")
        # Inicializar ingeniero de características
        feature_engineer = TaxiFeatureEngineer()
        
        # Ejecutar pipeline completo
        _ = feature_engineer.feature_engineering_pipeline()
        
        # Guardar datos con características
        output_path = feature_engineer.save_feature_data()
        
        # Mostrar resumen
        # summary = feature_engineer.feature_stats
        # print("\n" + "="*60)
        # print("RESUMEN DE INGENIERÍA DE CARACTERÍSTICAS")
        # print("="*60)
        # print(f"Total de características: {summary['total_columns']}")
        # print(f"Variables numéricas: {summary['numeric_columns']}")
        # print(f"Variables categóricas: {summary['categorical_columns']}")
        # print(f"Indicadores binarios: {summary['binary_indicators']}")
        # print(f"Tamaño del dataset: {summary['data_shape']}")
        # print(f"Archivo guardado: {output_path}")
        
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