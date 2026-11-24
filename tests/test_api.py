"""HTTP-layer tests against a fake model service."""
from tests.conftest import VALID_PAYLOAD


def test_health_ok(client, monkeypatch, fake_service):
    monkeypatch.setattr("app.api.routes.get_model_instance", lambda: fake_service)
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["version"] == "test-1.0.0"


def test_health_returns_503_when_model_cannot_load(client, monkeypatch):
    def broken():
        raise RuntimeError("artifact missing")

    monkeypatch.setattr("app.api.routes.get_model_instance", broken)
    res = client.get("/api/v1/health")
    assert res.status_code == 503
    assert res.json()["status"] == "unhealthy"
    assert res.json()["model_loaded"] is False


def test_predict_returns_503_when_model_cannot_load(client, monkeypatch):
    from app.main import app

    app.dependency_overrides.clear()

    def broken():
        raise RuntimeError("artifact missing")

    monkeypatch.setattr("app.api.routes.get_model_instance", broken)
    res = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert res.status_code == 503
    assert "artifact missing" not in res.text


def test_predict_ok(client):
    res = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    data = res.json()
    assert data["credit_score"] == 600.0
    assert data["default_probability"] == 0.02
    assert data["risk_level"] == "A"
    assert data["risk_description"] == "Low default risk"
    assert data["missing_fields"] == ["tot_cur_bal"]
    assert "tot_cur_bal" in data["message"]


def test_predict_validation_error(client):
    res = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "annual_inc": -100})
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail[0]["loc"] == ["body", "annual_inc"]
    # The offending input is never echoed back (it may not be JSON-serialisable).
    assert set(detail[0]) == {"loc", "msg", "type"}


def test_predict_rejects_nan_with_422(client):
    # FastAPI's default 422 handler crashes on NaN; ours must not.
    body = (
        '{"annual_inc": NaN, "int_rate": 12.5, "credit_history_length": 5.5, '
        '"purpose": "car", "verification_status": "Verified", "loan_amount": 15000}'
    )
    res = client.post(
        "/api/v1/predict", content=body, headers={"content-type": "application/json"}
    )
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["body", "annual_inc"]


def test_predict_rejects_infinity_on_unbounded_field(client):
    body = (
        '{"annual_inc": 75000, "int_rate": 12.5, "credit_history_length": 5.5, '
        '"purpose": "car", "verification_status": "Verified", "loan_amount": 15000, '
        '"tot_cur_bal": Infinity}'
    )
    res = client.post(
        "/api/v1/predict", content=body, headers={"content-type": "application/json"}
    )
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["body", "tot_cur_bal"]


def test_predict_rejects_loan_amount_and_loan_burden_together(client):
    res = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "loan_burden": 5})
    assert res.status_code == 422


def test_predict_rejects_unknown_fields(client):
    res = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "loan_amnt": 5000})
    assert res.status_code == 422


def test_predict_internal_error_does_not_leak_details(client):
    res = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "int_rate": 13.37})
    assert res.status_code == 500
    assert res.json() == {"detail": "Prediction failed"}


def test_predict_batch_ok(client):
    payload = {"applications": [VALID_PAYLOAD, {**VALID_PAYLOAD, "purpose": "car"}]}
    res = client.post("/api/v1/predict/batch", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["total_applications"] == 2
    assert [p["risk_level"] for p in body["predictions"]] == ["A", "A"]


def test_predict_batch_failure_is_an_error_not_a_fake_score(client):
    payload = {"applications": [VALID_PAYLOAD, {**VALID_PAYLOAD, "int_rate": 13.37}]}
    res = client.post("/api/v1/predict/batch", json=payload)
    assert res.status_code == 500
    assert res.json() == {"detail": "Batch prediction failed"}


def test_predict_batch_rejects_empty_and_oversized(client):
    assert (
        client.post("/api/v1/predict/batch", json={"applications": []}).status_code
        == 422
    )
    too_many = {"applications": [VALID_PAYLOAD] * 1001}
    assert client.post("/api/v1/predict/batch", json=too_many).status_code == 422
    extra_key = {"applications": [VALID_PAYLOAD], "include_details": True}
    assert client.post("/api/v1/predict/batch", json=extra_key).status_code == 422


def test_model_info(client):
    res = client.get("/api/v1/model/info")
    assert res.status_code == 200
    info = res.json()
    assert info["model_type"].startswith("Logistic Regression")
    assert len(info["features_used"]) == 9
    assert info["scoring_params"] == {"PDO": 20.0, "BaseScore": 600.0, "BaseOdds": 50.0}
    assert info["rating_bands"][0]["rating"] == "A"


def test_unhandled_exception_returns_json_500(client, monkeypatch):
    def boom():
        raise RuntimeError("secret internals")

    monkeypatch.setattr("app.main.get_api_metadata", boom)
    monkeypatch.delenv("DEBUG", raising=False)
    res = client.get("/info")
    assert res.status_code == 500
    assert res.json() == {
        "error": "Internal server error",
        "detail": "An error occurred",
    }


def test_ui_and_info_are_served(client):
    ui = client.get("/")
    assert ui.status_code == 200
    assert "Credit Risk Scorecard" in ui.text
    assert client.get("/static/js/app.js").status_code == 200
    info = client.get("/info").json()
    assert "/api/v1/predict" in info["endpoints"]


def test_cors_never_allows_credentials(client):
    res = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "https://example.org",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.status_code == 200
    assert "access-control-allow-credentials" not in res.headers


def test_allowed_hosts_env_keeps_localhost_for_healthcheck(monkeypatch):
    import importlib

    import app.main as main_module
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ALLOWED_HOSTS", "api.example.com")
    try:
        restricted = importlib.reload(main_module).app
        local = TestClient(restricted, base_url="http://localhost")
        assert local.get("/info").status_code == 200
        named = TestClient(restricted, base_url="http://api.example.com")
        assert named.get("/info").status_code == 200
        other = TestClient(restricted, base_url="http://evil.example")
        assert other.get("/info").status_code == 400
    finally:
        monkeypatch.delenv("ALLOWED_HOSTS")
        importlib.reload(main_module)
