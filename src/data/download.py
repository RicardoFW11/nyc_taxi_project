"""
Module for downloading NYC taxi data.

This module handles downloading the NYC taxi dataset from a specified URL,
saving it to a designated path, and logging relevant information about the download process.
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
    
    logger.info(f"Starting download from: {url}")
    
    # Check if file already exists
    if os.path.exists(save_path):
        existing_size = get_file_size_mb(save_path)
        logger.info(f"File already exists ({existing_size:.2f} MB). Skipping download.")
        return
    
    # Create destination directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download file
    start_time = time.time()
    logger.info("Downloading file...")
    
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        # Basic statistics
        download_duration = time.time() - start_time
        final_size = get_file_size_mb(save_path)
        speed = final_size / download_duration if download_duration > 0 else 0
        
        logger.info(f"Download completed: {final_size:.2f} MB in {download_duration:.1f}s ({speed:.2f} MB/s)")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during download: {e}")
        # Clean up partial file
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    
def download_data() -> int:
    """Main function to execute NYC taxi data download."""
    
    logger.info("Starting NYC Taxi data download")
    
    try:
        # Check configuration
        if not DOWNLOAD_URL_TAXI_DATA or not RAW_DATA:
            raise ValueError("Configuration URLs not defined")
        
        logger.info(f"URL: {DOWNLOAD_URL_TAXI_DATA}")
        logger.info(f"Destination: {RAW_DATA}")
        
        # Execute download
        download_taxi_data(DOWNLOAD_URL_TAXI_DATA, RAW_DATA)
        
        # Verify result
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