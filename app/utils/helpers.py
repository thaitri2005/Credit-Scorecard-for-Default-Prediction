import os
import logging
from typing import Optional
from app.models.model import CreditRiskModel

logger = logging.getLogger(__name__)

# Global model instance
_model_instance: Optional[CreditRiskModel] = None


def get_model_instance() -> CreditRiskModel:
    """
    Get or create the global model instance

    Returns:
        CreditRiskModel instance
    """
    global _model_instance

    if _model_instance is None:
        logger.info("Creating new model instance")
        _model_instance = CreditRiskModel()

        # Try to load model from file if it exists
        model_path = os.getenv("MODEL_PATH", "models/trained_model.pkl")
        if os.path.exists(model_path):
            _model_instance.load_model(model_path)
        else:
            logger.warning(
                f"Model file {model_path} not found. Using default mappings."
            )

    return _model_instance


def reload_model(model_path: str) -> bool:
    """
    Reload the model from a new file

    Args:
        model_path: Path to the new model file

    Returns:
        True if successful, False otherwise
    """
    global _model_instance

    try:
        if _model_instance is None:
            _model_instance = CreditRiskModel()

        _model_instance.load_model(model_path)
        logger.info(f"Model reloaded from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}")
        return False


def validate_environment() -> bool:
    """
    Validate that the environment is properly configured

    Returns:
        True if environment is valid, False otherwise
    """
    try:
        # Check if required environment variables are set
        from typing import List
        required_vars: List[str] = []

        missing_vars: List[str] = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.warning(f"Missing required environment variables: {missing_vars}")
            return False

        # Check if model can be loaded
        model = get_model_instance()
        if not model.woe_mappings:
            logger.warning("Model mappings not available")
            return False

        logger.info("Environment validation passed")
        return True

    except Exception as e:
        logger.error(f"Environment validation failed: {str(e)}")
        return False


def get_logging_config() -> dict:
    """
    Get logging configuration

    Returns:
        Dictionary with logging configuration
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(funcName)s:%(lineno)d - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "app": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {"level": log_level, "handlers": ["console"]},
    }


def setup_logging():
    """Setup logging configuration"""
    import logging.config

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Apply logging configuration
    logging.config.dictConfig(get_logging_config())

    logger.info("Logging configured successfully")


def get_api_metadata() -> dict:
    """
    Get API metadata

    Returns:
        Dictionary with API metadata
    """
    return {
        "name": "Credit Risk Scorecard API",
        "version": os.getenv("API_VERSION", "1.0.0"),
        "description": "API for predicting credit risk and generating credit scores",
        "author": "Credit Risk Team",
        "license": "MIT",
        "repository": "https://github.com/your-org/credit-risk-api",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
    }
