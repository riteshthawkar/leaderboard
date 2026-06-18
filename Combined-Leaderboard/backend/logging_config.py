"""
Comprehensive logging configuration for production deployment.
"""

import logging
import logging.handlers
from pathlib import Path
from pythonjsonlogger import jsonlogger
import os


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger("leaderboard")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Formatter for text logs
    text_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - for development and monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(text_formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler - main application log
    app_log_path = log_path / "app.log"
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(text_formatter)
    logger.addHandler(app_handler)
    
    # JSON file handler - for structured logging (ELK stack compatible)
    json_log_path = log_path / "app-json.log"
    json_handler = logging.handlers.RotatingFileHandler(
        json_log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    json_handler.setLevel(logging.DEBUG)
    json_formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s %(filename)s %(lineno)d'
    )
    json_handler.setFormatter(json_formatter)
    logger.addHandler(json_handler)
    
    # Error file handler - for errors only
    error_log_path = log_path / "error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(text_formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"Logging configured. Level: {log_level}, Directory: {log_path}")
    
    return logger


# Create module-level logger instance
logger = setup_logging(
    log_dir=os.getenv("LEADERBOARD_LOG_DIR", "logs"),
    log_level=os.getenv("LEADERBOARD_LOG_LEVEL", "INFO")
)
