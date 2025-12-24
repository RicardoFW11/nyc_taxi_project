"""
Módulo de Ingestión de Datos de Taxis de NYC.

Este módulo orquesta la adquisición de datos brutos desde fuentes remotas.
Gestiona la transferencia HTTP, asegura la persistencia en el sistema de archivos local
y registra métricas de rendimiento sobre el proceso de descarga.
"""

import requests
import os
import time
from pathlib import Path

from src.utils.logging import get_full_logger
from src.config.paths import LOGGER_NAME, DOWNLOAD_URL_TAXI_DATA, RAW_DATA

logger = get_full_logger(name=LOGGER_NAME, log_level="INFO")

def get_file_size_mb(file_path: str) -> float:
    """
    Calcula el tamaño de un archivo en Megabytes (MB).
    Utilizado principalmente para la generación de reportes y métricas de logs.
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def download_taxi_data(url: str, save_path: str) -> None:
    """
    Ejecuta la transferencia del dataset desde el origen remoto hacia el almacenamiento local.
    
    Implementa una verificación de caché para evitar descargas redundantes si el archivo
    ya existe en el destino.

    Args:
        url (str): Endpoint HTTP fuente del dataset.
        save_path (str): Ruta absoluta local donde se persistirá el archivo.
    """
    
    logger.info(f"Starting download from: {url}")
    
    # Verifica la existencia previa del archivo para optimizar ancho de banda y tiempo.
    if os.path.exists(save_path):
        existing_size = get_file_size_mb(save_path)
        logger.info(f"File already exists ({existing_size:.2f} MB). Skipping download.")
        return
    
    # Garantiza la existencia de la jerarquía de directorios antes de la escritura.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Inicia el cronómetro y el proceso de transferencia.
    start_time = time.time()
    logger.info("Downloading file...")
    
    try:
        # Establece la conexión HTTP con un tiempo de espera definido para evitar bloqueos.
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Escribe el flujo de datos binarios directamente al disco.
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        # Calcula métricas de rendimiento de la transferencia (throughput).
        download_duration = time.time() - start_time
        final_size = get_file_size_mb(save_path)
        speed = final_size / download_duration if download_duration > 0 else 0
        
        logger.info(f"Download completed: {final_size:.2f} MB in {download_duration:.1f}s ({speed:.2f} MB/s)")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during download: {e}")
        # Realiza una limpieza de artefactos corruptos o incompletos tras un fallo de red.
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    
def download_data() -> int:
    """
    Función orquestadora principal para la descarga de datos.
    Valida la configuración del entorno antes de delegar la ejecución técnica.
    
    Returns:
        int: Código de estado de salida (0 para éxito, 1 para error).
    """
    
    logger.info("Starting NYC Taxi data download")
    
    try:
        # Valida que las variables de configuración críticas estén definidas en el entorno.
        if not DOWNLOAD_URL_TAXI_DATA or not RAW_DATA:
            raise ValueError("Configuration URLs not defined")
        
        logger.info(f"URL: {DOWNLOAD_URL_TAXI_DATA}")
        logger.info(f"Destination: {RAW_DATA}")
        
        # Invoca la lógica de transferencia con los parámetros validados.
        download_taxi_data(DOWNLOAD_URL_TAXI_DATA, RAW_DATA)
        
        # Verifica la integridad del resultado en el sistema de archivos.
        if os.path.exists(RAW_DATA):
            size = get_file_size_mb(RAW_DATA)
            logger.info(f"✅ Download successful: {size:.2f} MB")
        else:
            logger.error("❌ Error: File was not created")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during download: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(download_data())