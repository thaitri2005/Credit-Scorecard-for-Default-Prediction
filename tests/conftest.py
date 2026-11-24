import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_model as routes_get_model
from app.main import app as fastapi_app

FEATURES = [
    "int_rate_woe",
    "total_rev_hi_lim_woe",
    "tot_cur_bal_woe",
    "annual_inc_woe",
    "purpose_woe",
    "loan_burden_woe",
    "credit_history_length_woe",
    "revol_util_woe",
    "verification_status_woe",
]

VALID_PAYLOAD = {
    "annual_inc": 75000,
    "int_rate": 12.5,
    "credit_history_length": 5.5,
    "purpose": "debt_consolidation",
    "verification_status": "Verified",
    "loan_amount": 15000,
}

FULL_PAYLOAD = {
    **VALID_PAYLOAD,
    "total_rev_hi_lim": 30000,
    "tot_cur_bal": 120000,
    "revol_util": 40,
}


class FakeCreditRiskService:
    """Deterministic stand-in used to test the HTTP layer in isolation."""

    def __init__(self):
        self.metadata = {"version": "test-1.0.0", "training_date": "2024-01-01"}
        self.features = FEATURES
        self.scoring_params = {"PDO": 20.0, "BaseScore": 600.0, "BaseOdds": 50.0}
        self.lowest_rating = {"rating": "C", "description": "Highest default risk"}
        self.rating_bands = [
            {"min_score": 590, "rating": "A", "description": "Low default risk"}
        ]

    def predict(self, input_dict: dict) -> dict:
        return self.predict_many([input_dict])[0]

    def predict_many(self, inputs: list) -> list:
        if any(item.get("int_rate") == 13.37 for item in inputs):
            raise ValueError("Synthetic failure with secret internals")
        return [
            {
                "credit_score": 600.0,
                "default_probability": 0.02,
                "risk_level": "A",
                "risk_description": "Low default risk",
                "log_odds": -3.8918,
                "missing_fields": ["tot_cur_bal"],
            }
            for _ in inputs
        ]


@pytest.fixture()
def fake_service() -> FakeCreditRiskService:
    return FakeCreditRiskService()


@pytest.fixture()
def client(fake_service: FakeCreditRiskService):
    """Client whose model dependency is replaced by the fake service."""
    fastapi_app.dependency_overrides[routes_get_model] = lambda: fake_service
    yield TestClient(fastapi_app, raise_server_exceptions=False)
    fastapi_app.dependency_overrides.clear()


@pytest.fixture()
def real_client():
    """Client running against the real model artifacts in models/."""
    with TestClient(fastapi_app, raise_server_exceptions=False) as test_client:
        yield test_client
