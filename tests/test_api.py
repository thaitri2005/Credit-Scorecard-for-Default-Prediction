def test_health_ok(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert "version" in body


def test_predict_ok(client):
    payload = {
        "annual_inc": 75000,
        "int_rate": 12.5,
        "credit_history_length": 5.5,
        "purpose": "debt_consolidation",
        "verification_status": "Verified",
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["credit_score"] == 650.0
    assert data["default_probability"] == 0.1
    assert data["risk_level"] == "Medium Risk"
    assert "message" in data


def test_predict_validation_error(client):
    # invalid: negative income
    payload = {
        "annual_inc": -100,
        "int_rate": 12.5,
        "credit_history_length": 5.5,
        "purpose": "debt_consolidation",
        "verification_status": "Verified",
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 422


def test_predict_internal_error_from_service(client):
    # use valid payload but special int_rate to trigger fake service error
    payload = {
        "annual_inc": 50000,
        "int_rate": 13.37,
        "credit_history_length": 5.5,
        "purpose": "debt_consolidation",
        "verification_status": "Verified",
    }
    res = client.post("/api/v1/predict", json=payload)
    # routes.py catches and maps to 500
    assert res.status_code == 500


def test_predict_batch_mixed_results(client):
    payload = {
        "applications": [
            {
                "annual_inc": 70000,
                "int_rate": 10.0,
                "credit_history_length": 4.0,
                "purpose": "car",
                "verification_status": "Verified",
            },
            {
                # this one will fail inside fake service
                "annual_inc": 70000,
                "int_rate": 13.37,
                "credit_history_length": 8.0,
                "purpose": "credit_card",
                "verification_status": "Verified",
            },
        ]
    }
    res = client.post("/api/v1/predict/batch", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["total_applications"] == 2
    preds = body["predictions"]
    assert len(preds) == 2
    assert preds[0]["risk_level"] == "Medium Risk"
    assert preds[1]["risk_level"] == "Error"


def test_model_info(client):
    res = client.get("/api/v1/model/info")
    assert res.status_code == 200
    info = res.json()
    assert info["model_type"].startswith("Logistic Regression")
    assert isinstance(info["features_used"], list)
    assert set(["PDO", "BaseScore", "BaseOdds"]).issubset(info["scoring_params"].keys())
