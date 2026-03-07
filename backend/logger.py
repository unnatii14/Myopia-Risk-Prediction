"""
Structured logging for Flask API
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(name: str = 'api', log_file: str = 'logs/api.log', level: str = 'INFO'):
    """
    Configure structured logging with file rotation
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class RequestLogger:
    """Middleware for logging API requests"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def log_request(self, request):
        """Log incoming request"""
        self.logger.info(
            f"Request: {request.method} {request.path} | "
            f"IP: {request.remote_addr} | "
            f"Content-Type: {request.content_type}"
        )
    
    def log_response(self, response, duration_ms: float):
        """Log outgoing response"""
        self.logger.info(
            f"Response: {response.status_code} | "
            f"Duration: {duration_ms:.2f}ms | "
            f"Size: {len(response.data)} bytes"
        )
    
    def log_prediction(self, input_data: dict, output: dict, duration_ms: float):
        """Log ML prediction event"""
        self.logger.info(
            f"Prediction: Age={input_data.get('age')} | "
            f"Risk={output.get('risk_level')} ({output.get('risk_score', 0):.1f}%) | "
            f"Duration: {duration_ms:.2f}ms"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(
            f"Error in {context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )
