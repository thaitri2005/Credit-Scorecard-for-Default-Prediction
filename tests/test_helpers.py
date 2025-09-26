from app.utils.helpers import get_api_metadata, validate_environment, setup_logging


def test_get_api_metadata():
    meta = get_api_metadata()
    assert meta["service"] == "Credit Risk Scorecard API"
    assert "endpoints" in meta and "/api/v1/predict" in meta["endpoints"]


def test_validate_environment_ok():
    assert validate_environment() is True


def test_setup_logging_idempotent():
    logger1 = setup_logging()
    logger2 = setup_logging()
    assert logger1.name == logger2.name
