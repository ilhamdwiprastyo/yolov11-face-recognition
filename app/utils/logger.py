import logging
import sys
from pathlib import Path
from loguru import logger
from config.settings import settings

def setup_logging():
    """Setup logging configuration"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler
    log_file = Path(settings.log_dir) / "api.log"
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    return logger

def get_logger(name: str):
    """Get logger for specific module"""
    return logger.bind(name=name)

# Setup logging when module is imported
setup_logging()