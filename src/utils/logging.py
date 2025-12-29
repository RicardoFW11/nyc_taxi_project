"""
Módulo de Utilidades de Logging (Registro de Eventos).

Implementa el patrón de diseño 'Factory' para estandarizar la creación y configuración
de objetos Logger en toda la aplicación. Centraliza la gestión de niveles de log,
formatos de salida y destinos (consola vs archivo), asegurando consistencia en la
trazabilidad del sistema.
"""

import logging
from datetime import datetime
from pathlib import Path


class LoggerFactory:
    """
    Fábrica de Loggers configurable.
    
    Proporciona métodos estáticos para instanciar loggers con configuraciones predefinidas
    o personalizadas, manejando automáticamente la creación de directorios y la
    prevención de duplicidad de handlers.
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
        Crea e inicializa una instancia de Logger.

        Si el logger ya existe y tiene handlers configurados, retorna la instancia existente
        para evitar la duplicación de mensajes en la salida.

        Args:
            name (str): Identificador único del logger (generalmente __name__).
            log_level (str): Nivel de severidad mínimo a registrar ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            log_dir (str): Directorio destino para los archivos de log (si file_output=True).
            console_output (bool): Habilita el flujo de salida estándar (stdout).
            file_output (bool): Habilita la persistencia en archivos de texto rotativos.
            log_format (str, optional): Plantilla de formato para los mensajes.

        Returns:
            logging.Logger: Objeto logger configurado y listo para usar.
        """
        # Recuperación o creación de la instancia base
        logger = logging.getLogger(name)
        
        # Verificación de idempotencia: Si ya tiene handlers, no se reconfigura.
        if logger.handlers:
            return logger
        
        # Configuración del nivel de severidad global
        log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(log_level_obj)
        
        # Definición del formato estándar si no se provee uno personalizado
        if log_format is None:
            log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(log_format)
        
        # Configuración del Handler de Consola (StreamHandler)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level_obj)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Configuración del Handler de Archivo (FileHandler)
        if file_output:
            # Garantiza la existencia del directorio de logs
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True)
            
            # Generación de nombre de archivo único basado en timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{name}_{timestamp}.log"
            log_filepath = log_path / log_filename
            
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(log_level_obj)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


# --- Funciones de conveniencia (Helpers) ---

def get_console_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Retorna un logger configurado exclusivamente para salida por consola.
    Ideal para scripts interactivos o entornos de desarrollo efímeros.
    """
    return LoggerFactory.create_logger(
        name=name,
        log_level=log_level,
        console_output=True,
        file_output=False
    )


def get_file_logger(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Retorna un logger configurado exclusivamente para persistencia en archivo.
    Útil para tareas en segundo plano (background jobs) donde no hay stdout visible.
    """
    return LoggerFactory.create_logger(
        name=name,
        log_level=log_level,
        log_dir=log_dir,
        console_output=False,
        file_output=True
    )


def get_full_logger(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Retorna un logger completo (Consola + Archivo).
    Configuración recomendada para entornos de producción donde se requiere monitoreo
    en tiempo real y auditoría histórica.
    """
    return LoggerFactory.create_logger(
        name=name,
        log_level=log_level,
        log_dir=log_dir,
        console_output=True,
        file_output=True
    )