"""
Configuration management for Flask API
"""
import os
from pathlib import Path
from typing import List


DEFAULT_SECRET_KEY = 'dev-secret-key-change-me'
DEFAULT_JWT_SECRET = 'myopia_dev_secret_key_2024'

class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', DEFAULT_SECRET_KEY)
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


def get_cors_origins() -> List[str]:
    """Return normalized non-empty CORS origins from environment config."""
    raw = os.getenv('CORS_ORIGINS', 'http://localhost:5174')
    return [origin.strip() for origin in raw.split(',') if origin.strip()]


def validate_production_config() -> List[str]:
    """Return a list of fatal production configuration issues."""
    errors: List[str] = []
    env = os.getenv('FLASK_ENV', 'development').lower()
    if env != 'production':
        return errors

    secret_key = os.getenv('SECRET_KEY', DEFAULT_SECRET_KEY)
    jwt_secret = os.getenv('JWT_SECRET', DEFAULT_JWT_SECRET)
    google_client_id = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    cors_origins = get_cors_origins()

    if not secret_key or secret_key == DEFAULT_SECRET_KEY:
        errors.append('SECRET_KEY must be set to a strong non-default value in production.')
    if not jwt_secret or jwt_secret == DEFAULT_JWT_SECRET:
        errors.append('JWT_SECRET must be set to a strong non-default value in production.')
    if not google_client_id:
        errors.append('GOOGLE_CLIENT_ID must be configured in production.')
    if not cors_origins:
        errors.append('CORS_ORIGINS must contain at least one trusted origin in production.')
    if any('localhost' in origin.lower() for origin in cors_origins):
        errors.append('CORS_ORIGINS must not include localhost entries in production.')

    return errors
