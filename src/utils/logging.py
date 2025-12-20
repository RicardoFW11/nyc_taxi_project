import logging
from datetime import datetime
from pathlib import Path


class LoggerFactory:
    """
    Factory class to create and configure loggers with both console and file handlers.
    """
    
    @staticmethod
    def create_logger(
        name: str,
        log_level: str = "INFO",
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = False,
        log_format: str = None
    ) -> logging.Logger:
        """
        Create a logger with console and/or file handlers.
        
        Args:
            name (str): Name of the logger
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir (str): Directory to store log files
            console_output (bool): Enable console output
            file_output (bool): Enable file output
            log_format (str): Custom log format string
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(name)
        
        # Avoid adding handlers multiple times
        if logger.handlers:
            return logger
        
        # Set log level
        log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(log_level_obj)
        
        # Default log format
        if log_format is None:
            log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(log_format)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level_obj)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            # Create logs directory if it doesn't exist
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True)
            
            # Create log filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{name}_{timestamp}.log"
            log_filepath = log_path / log_filename
            
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(log_level_obj)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


# Convenience functions for quick logger creation
def get_console_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Get a logger that only outputs to console."""
    return LoggerFactory.create_logger(
        name=name,
        log_level=log_level,
        console_output=True,
        file_output=False
    )


def get_file_logger(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Get a logger that only outputs to file."""
    return LoggerFactory.create_logger(
        name=name,
        log_level=log_level,
        log_dir=log_dir,
        console_output=False,
        file_output=True
    )


def get_full_logger(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Get a logger that outputs to both console and file."""
    return LoggerFactory.create_logger(
        name=name,
        log_level=log_level,
        log_dir=log_dir,
        console_output=True,
        file_output=True
    )
