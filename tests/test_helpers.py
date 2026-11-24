import logging

import pytest

import app.utils.helpers as helpers
from app.utils.helpers import API_VERSION, get_api_metadata, setup_logging


def test_get_api_metadata():
    meta = get_api_metadata()
    assert meta["service"] == "Credit Risk Scorecard API"
    assert meta["version"] == API_VERSION
    assert "/api/v1/predict" in meta["endpoints"]


def test_setup_logging_returns_module_logger():
    assert isinstance(setup_logging(), logging.Logger)


def test_failed_model_load_is_remembered_not_retried(monkeypatch):
    calls = []

    class Broken:
        def __init__(self):
            calls.append(1)
            raise RuntimeError("artifact missing")

    monkeypatch.setattr(helpers, "CreditRiskService", Broken)
    monkeypatch.setattr(helpers, "_model_instance", None)
    monkeypatch.setattr(helpers, "_model_error", None)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="artifact missing"):
            helpers.get_model_instance()
    assert len(calls) == 1
