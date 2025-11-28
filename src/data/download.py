"""
Módulo para descarga de datos de taxi de NYC.
"""

import requests
import os
import time
from pathlib import Path

from src.utils.logging import get_full_logger
from src.config.paths import LOGGER_NAME, DOWNLOAD_URL_TAXI_DATA, RAW_DATA

logger = get_full_logger(name=LOGGER_NAME, log_level="INFO")

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def download_taxi_data(url: str, save_path: str) -> None:
    """
    Download taxi data from the specified URL and save it to the given path.
    
    Args:
        url (str): The URL to download the taxi data from.
        save_path (str): The file path to save the downloaded data.
    """
    
    logger.info(f"Iniciando descarga desde: {url}")
    
    # Verificar si el archivo ya existe
    if os.path.exists(save_path):
        existing_size = get_file_size_mb(save_path)
        logger.info(f"Archivo ya existe ({existing_size:.2f} MB). Saltando descarga.")
        return
    
    # Crear directorio de destino
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Descargar archivo
    start_time = time.time()
    logger.info("Descargando archivo...")
    
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        # Estadísticas básicas
        download_duration = time.time() - start_time
        final_size = get_file_size_mb(save_path)
        speed = final_size / download_duration if download_duration > 0 else 0
        
        logger.info(f"Descarga completada: {final_size:.2f} MB en {download_duration:.1f}s ({speed:.2f} MB/s)")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error durante la descarga: {e}")
        # Limpiar archivo parcial
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    
def main() -> None:
    """Función principal para ejecutar la descarga de datos de taxi NYC."""
    
    logger.info("Iniciando descarga de datos NYC Taxi")
    
    try:
        # Verificar configuración
        if not DOWNLOAD_URL_TAXI_DATA or not RAW_DATA:
            raise ValueError("URLs de configuración no definidas")
        
        logger.info(f"URL: {DOWNLOAD_URL_TAXI_DATA}")
        logger.info(f"Destino: {RAW_DATA}")
        
        # Ejecutar descarga
        download_taxi_data(DOWNLOAD_URL_TAXI_DATA, RAW_DATA)
        
        # Verificar resultado
        if os.path.exists(RAW_DATA):
            size = get_file_size_mb(RAW_DATA)
            logger.info(f"✅ Descarga exitosa: {size:.2f} MB")
        else:
            logger.error("❌ Error: Archivo no fue creado")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error durante la descarga: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())