"""
Configuration management for Flask API
"""
import os
from pathlib import Path
from typing import List

class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-me')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5174').split(',')
    
    # Models
    MODELS_DIR = Path(os.getenv('MODELS_DIR', '../models'))
    MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')
    
    # API
    API_PORT = int(os.getenv('API_PORT', 5001))
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 1048576))
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 30))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/api.log')
    
    # Monitoring
    ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'false').lower() == 'true'
    SENTRY_DSN = os.getenv('SENTRY_DSN', '')


class DevelopmentConfig(Config):
    """Development environment config"""
    DEBUG = True


class ProductionConfig(Config):
    """Production environment config"""
    DEBUG = False
    RATE_LIMIT_ENABLED = True


class TestingConfig(Config):
    """Testing environment config"""
    TESTING = True
    DEBUG = True


def get_config():
    """Get configuration based on FLASK_ENV"""
    env = os.getenv('FLASK_ENV', 'development')
    
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
    }
    
    return configs.get(env, DevelopmentConfig)()
