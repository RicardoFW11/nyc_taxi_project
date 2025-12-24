"""
Pipeline de Orquestación de Datos ETL (Extraction, Transformation, Loading).

Este script define y ejecuta el flujo de trabajo completo para la preparación del dataset.
Integra los módulos de descarga, preprocesamiento, ingeniería de características y particionado,
generando los artefactos binarios (.pkl) necesarios para el entrenamiento de modelos.

Flujo de ejecución:
1.  Ingesta de datos crudos (Download).
2.  Limpieza y validación (Preprocessing).
3.  Generación de variables predictivas (Feature Engineering).
4.  División estratificada y serialización (Data Splitting & Persistencia).
"""

import os
import sys
import logging
import pandas as pd
import joblib
from pathlib import Path

# Configuración del entorno de ejecución
try:
    # Asegura que el directorio raíz del proyecto esté en el PYTHONPATH
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except Exception:
    pass    

# Importación de componentes del pipeline
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

# Inicialización del sistema de logging centralizado
logger = LoggerFactory.create_logger(
    name=LOGGER_NAME,
    log_level='DEBUG',
    console_output=True,
    file_output=False
)

def build_complete_dataset():
    """
    Ejecuta el pipeline ETL completo para construir el dataset de entrenamiento.
    
    Proceso:
    1. Prepara la estructura de directorios.
    2. Descarga los datos fuente si no existen localmente.
    3. Aplica limpieza y validación de reglas de negocio.
    4. Realiza un muestreo estratégico para optimizar el uso de memoria en entornos contenerizados.
    5. Genera características avanzadas (temporales, espaciales).
    6. Divide los datos en conjuntos de entrenamiento/prueba y serializa los objetos resultantes.
    
    Returns:
        bool: True si el pipeline se completó exitosamente, False en caso contrario.
    """
    try:
        logger.info("Starting complete dataset build pipeline...")
        
        # Paso 1: Inicialización del entorno
        # Garantiza la existencia de los directorios de trabajo
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Paso 2: Ingesta de datos (Extracción)
        logger.info("Step 2: Downloading raw data...")
        download_success = download_data()
        if download_success > 0:
            logger.error("Data download failed. Stopping pipeline.")
            return False
        
        # Paso 3: Preprocesamiento (Limpieza y Validación)
        logger.info("Step 3: Preprocessing data...")
        preprocessor = TaxiDataPreprocessor()
        cleaned_data = preprocessor.preprocess_full_pipeline(
            remove_outliers=True,
            outlier_method='iqr',
            outlier_factor=1.5
        )

        # Gestión de Recursos: Muestreo para estabilidad en Docker
        # Se limita el dataset a 300k registros para evitar desbordamientos de memoria (OOM)
        # durante la fase de entrenamiento en contenedores con recursos limitados.
        logger.warning("Applying data sampling to ensure memory stability within Docker container.")
        if len(cleaned_data) > 300000:
            cleaned_data = cleaned_data.sample(n=300000, random_state=42)
            logger.info(f"Dataset downsampled to: {len(cleaned_data)} records")

        if cleaned_data is None or cleaned_data.empty:
            logger.error("Data preprocessing failed (empty dataset). Stopping pipeline.")
            return False
        
        # Persistencia intermedia: Sobrescritura con dataset optimizado
        processed_path = preprocessor.save_processed_data()

        logger.info(f"Overwriting intermediate processed file with {len(cleaned_data)} records...")
        cleaned_data.to_parquet(processed_path) 
        logger.info("Intermediate file update completed.")

        # Paso 4: Ingeniería de Características (Transformación)
        logger.info("Step 4: Creating features...")
        feature_engineer = TaxiFeatureEngineer(processed_data_path=processed_path)
        feature_data = feature_engineer.feature_engineering_pipeline()
        
        # Persistencia de dataset enriquecido
        output_path = feature_engineer.save_feature_data()
        
        # Paso 5: Particionamiento y Serialización (Carga)
        # Prepara los diccionarios de datos específicos para cada modelo (Tarifa y Duración)
        logger.info("Step 5: Splitting data for model training...")
        
        data_splitter = DataSplitter()
        data_splits = data_splitter.split_data_for_both_models(feature_data)
        
        # Serialización de datos para el modelo de Tarifa
        joblib.dump(data_splits['fare_model'], FARE_MODEL_DATA_FILE)
        logger.info(f"Fare model training data serialized at: {FARE_MODEL_DATA_FILE}")
        
        # Serialización de datos para el modelo de Duración
        joblib.dump(data_splits['duration_model'], DURATION_MODEL_DATA_FILE)
        logger.info(f"Duration model training data serialized at: {DURATION_MODEL_DATA_FILE}")
        
        # Exportación de metadatos de importancia de características
        data_splits['fare_model']['feature_scores'].to_csv(FARE_FEATURE_SCORES_FILE, index=False)
        data_splits['duration_model']['feature_scores'].to_csv(DURATION_FEATURE_SCORES_FILE, index=False)
        
        logger.info("Complete dataset build pipeline finished successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with critical error: {str(e)}")
        print(f"FATAL ETL ERROR: {str(e)}")
        return False

def main():
    """Función de punto de entrada para la ejecución del script."""
    success = build_complete_dataset()
    if success:
        logger.info("Dataset build completed successfully!")
        sys.exit(0)
    else:
        logger.error("Dataset build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()