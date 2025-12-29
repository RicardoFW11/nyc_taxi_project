"""
NYC Taxi Data Ingestion Module.

This module orchestrates the acquisition of raw data from remote sources.
It manages HTTP transfer, ensures persistence in the local file system,
and logs performance metrics on the download process.
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
    Calculates the size of a file in megabytes (MB).
    Mainly used for generating reports and log metrics.
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def download_taxi_data(url: str, save_path: str) -> None:
    """
    Executes the transfer of the dataset from the remote source to local storage.
    
    Implements a cache check to avoid redundant downloads if the file
    already exists at the destination.

    Args:
        url (str): Source HTTP endpoint of the dataset.
        save_path (str): Local absolute path where the file will be persisted.
    """
    
    logger.info(f"Starting download from: {url}")
    
    # Check for the existence of the file beforehand to optimize bandwidth and time.
    if os.path.exists(save_path):
        existing_size = get_file_size_mb(save_path)
        logger.info(f"File already exists ({existing_size:.2f} MB). Skipping download.")
        return
    
    # Ensures the existence of the directory hierarchy before writing.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Start the timer and the transfer process.
    start_time = time.time()
    logger.info("Downloading file...")
    
    try:
        # Establishes the HTTP connection with a defined timeout to prevent deadlocks.
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Writes the binary data stream directly to the disk.
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        # Calculates transfer throughput metrics.
        download_duration = time.time() - start_time
        final_size = get_file_size_mb(save_path)
        speed = final_size / download_duration if download_duration > 0 else 0
        
        logger.info(f"Download completed: {final_size:.2f} MB in {download_duration:.1f}s ({speed:.2f} MB/s)")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during download: {e}")
        # Perform a cleanup of corrupt or incomplete devices after a network failure.
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    
def download_data() -> int:
    """
    Primary orchestration function for data download.
    Validates the environment configuration before delegating technical execution.
    
    Returns:
        int: Exit status code (0 for success, 1 for error).
    """
    
    logger.info("Starting NYC Taxi data download")
    
    try:
        # Validate that critical configuration variables are defined in the environment.
        if not DOWNLOAD_URL_TAXI_DATA or not RAW_DATA:
            raise ValueError("Configuration URLs not defined")
        
        logger.info(f"URL: {DOWNLOAD_URL_TAXI_DATA}")
        logger.info(f"Destination: {RAW_DATA}")
        
        # Invoke the transfer logic with the validated parameters.
        download_taxi_data(DOWNLOAD_URL_TAXI_DATA, RAW_DATA)
        
        # Verify the integrity of the result in the file system.
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