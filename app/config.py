"""Service-wide settings: version, logging and service metadata."""
import logging
import os

API_VERSION = "1.1.0"


def setup_logging():
    """Configure logging format and level"""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def get_api_metadata():
    """Return API metadata"""
    return {
        "service": "Credit Risk Scorecard API",
        "version": API_VERSION,
        "description": (
            "Predict loan default probability and credit score using "
            "logistic regression + WOE pipeline"
        ),
        "endpoints": [
            "/api/v1/health",
            "/api/v1/predict",
            "/api/v1/predict/batch",
            "/api/v1/model/info",
        ],
    }
