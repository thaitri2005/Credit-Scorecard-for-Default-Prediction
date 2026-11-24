import logging

from app.config import API_VERSION, get_api_metadata, setup_logging


def test_get_api_metadata():
    meta = get_api_metadata()
    assert meta["service"] == "Credit Risk Scorecard API"
    assert meta["version"] == API_VERSION
    assert "/api/v1/predict" in meta["endpoints"]


def test_setup_logging_returns_module_logger():
    assert isinstance(setup_logging(), logging.Logger)
