import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.api.routes import get_model as routes_get_model


class FakeCreditRiskService:
    def __init__(self):
        self.metadata = {
            "version": "test-1.0.0",
            "features": [
                "int_rate_woe",
                "total_rev_hi_lim_woe",
                "tot_cur_bal_woe",
                "annual_inc_woe",
                "purpose_woe",
                "loan_burden_woe",
                "credit_history_length_woe",
                "revol_util_woe",
                "verification_status_woe",
            ],
            "PDO": 20,
            "BaseScore": 600,
            "BaseOdds": 50,
            "training_date": "2024-01-01",
            "risk_prob_thresholds": {"low": 0.05, "medium": 0.15, "high": 0.3},
        }

    def predict(self, input_dict: dict) -> dict:
        if input_dict.get("int_rate") == 13.37:
            raise ValueError("Synthetic failure")
        # Return deterministic values for assertions
        return {
            "credit_score": 650.0,
            "default_probability": 0.1,
            "risk_level": "Medium Risk",
            "log_odds": -2.1972,
        }


@pytest.fixture()
def fake_service() -> FakeCreditRiskService:
    return FakeCreditRiskService()


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch, fake_service: FakeCreditRiskService
) -> TestClient:
    # Monkeypatch for functions that call routes_get_model directly (e.g., /health)
    monkeypatch.setattr("app.api.routes.get_model", lambda: fake_service)

    # Dependency override for routes using Depends(get_model)
    fastapi_app.dependency_overrides[routes_get_model] = lambda: fake_service

    return TestClient(fastapi_app)


@pytest.fixture(autouse=True)
def clear_overrides():
    # Ensure dependency overrides are cleared between tests
    yield
    fastapi_app.dependency_overrides.clear()
