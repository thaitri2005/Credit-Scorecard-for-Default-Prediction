"""Tests against the real model artifacts shipped in models/.

These guard the things a fake service cannot: that the pickles load with the
pinned library versions, and that scoring behaves sensibly end to end.
"""
import itertools
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from app.api.schemas import LoanPurpose, VerificationStatus
from app.utils.helpers import (
    DEFAULT_MODEL_DIR,
    CreditRiskService,
)
from tests.conftest import FULL_PAYLOAD, VALID_PAYLOAD


@pytest.fixture(scope="module")
def service() -> CreditRiskService:
    return CreditRiskService()


def score(service, **overrides) -> float:
    return service.predict({**FULL_PAYLOAD, **overrides})["credit_score"]


def test_artifacts_load_and_predict_golden_values(service):
    result = service.predict(dict(FULL_PAYLOAD))
    assert result["credit_score"] == 569.55
    assert result["default_probability"] == 0.0543
    assert result["risk_level"] == "BB"
    assert result["risk_description"] == "Average default risk"
    assert result["log_odds"] == -2.8568
    assert result["missing_fields"] == []


def test_points_breakdown_adds_up_to_the_score(service):
    result = service.predict(dict(FULL_PAYLOAD))
    assert result["base_points"] == 562.01
    assert [item["feature"] for item in result["points"]] == [
        feature[: -len("_woe")] for feature in service.features
    ]
    total = result["base_points"] + sum(item["points"] for item in result["points"])
    assert total == pytest.approx(result["credit_score"], abs=0.05)
    by_feature = {item["feature"]: item["points"] for item in result["points"]}
    assert by_feature["int_rate"] == 7.89
    assert by_feature["verification_status"] < 0


ARTIFACTS = ("credit_risk_model.joblib", "preprocessing_pipeline.joblib")


def copy_models(tmp_path):
    target = tmp_path / "models"
    shutil.copytree(DEFAULT_MODEL_DIR, target)
    return target


def edit_metadata(model_dir, change):
    path = model_dir / "model_metadata.json"
    metadata = json.loads(path.read_text())
    change(metadata)
    path.write_text(json.dumps(metadata))


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_tampered_artifact_refuses_to_load(tmp_path, artifact):
    model_dir = copy_models(tmp_path)
    (model_dir / artifact).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match=f"Checksum mismatch.*{artifact}"):
        CreditRiskService(model_dir=model_dir)


def test_checksum_verification_fails_closed(tmp_path):
    model_dir = copy_models(tmp_path)
    (model_dir / ARTIFACTS[0]).write_bytes(b"tampered")

    edit_metadata(model_dir, lambda m: m["artifact_sha256"].pop(ARTIFACTS[0]))
    with pytest.raises(RuntimeError, match="No checksum recorded"):
        CreditRiskService(model_dir=model_dir)

    edit_metadata(model_dir, lambda m: m.pop("artifact_sha256"))
    with pytest.raises(RuntimeError, match="no artifact_sha256"):
        CreditRiskService(model_dir=model_dir)

    (model_dir / "model_metadata.json").unlink()
    with pytest.raises(RuntimeError, match="Cannot read"):
        CreditRiskService(model_dir=model_dir)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(BaseOdds=0),
        lambda m: m.update(PDO=None),
        lambda m: m["rating_bands"][0].pop("min_score"),
        lambda m: m.update(rating_bands=[{"min_score": "high", "rating": "A"}]),
    ],
)
def test_malformed_scoring_metadata_is_rejected(tmp_path, change):
    model_dir = copy_models(tmp_path)
    edit_metadata(model_dir, change)
    with pytest.raises(RuntimeError, match="metadata"):
        CreditRiskService(model_dir=model_dir)


def test_service_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert CreditRiskService().predict(dict(FULL_PAYLOAD))["credit_score"] == 569.55


def test_every_purpose_and_verification_status_scores(service):
    for purpose, status in itertools.product(LoanPurpose, VerificationStatus):
        result = service.predict(
            {
                **VALID_PAYLOAD,
                "purpose": purpose.value,
                "verification_status": status.value,
            }
        )
        assert 0 < result["default_probability"] < 1


def test_score_matches_scorecard_points(service):
    """API score == base points + sum of the points in models/scorecard.csv."""
    card = pd.read_csv(DEFAULT_MODEL_DIR / "scorecard.csv")
    applicant = {
        "annual_inc": 90000,
        "int_rate": 8.0,
        "credit_history_length": 16.0,
        "purpose": "credit_card",
        "verification_status": "Not Verified",
        "total_rev_hi_lim": 45000,
        "tot_cur_bal": 250000,
        "revol_util": 20,
        "loan_burden": 0.1,
    }
    bins = {
        "annual_inc": "(80046.219, 100129.0]",
        "int_rate": "(7.275, 9.995]",
        "credit_history_length": "(13.593, 19.092]",
        "purpose": "credit_card",
        "verification_status": "Not Verified",
        "total_rev_hi_lim": "(39694.5, 55668.5]",
        "tot_cur_bal": "(223634.5, 386197.5]",
        "revol_util": "(-inf, 38.15]",
        "loan_burden": "(-inf, 0.164]",
    }
    points = sum(
        card[(card.feature == feature) & (card.bin == label)].points.iloc[0]
        for feature, label in bins.items()
    )
    expected = service.metadata["base_points"] + points
    assert service.predict(applicant)["credit_score"] == pytest.approx(
        expected, abs=0.1
    )


def test_riskier_inputs_score_lower(service):
    assert (
        score(service, int_rate=7)
        > score(service, int_rate=13)
        > score(service, int_rate=25)
    )
    assert score(service, annual_inc=150000) > score(service, annual_inc=30000)
    assert score(service, revol_util=10) > score(service, revol_util=95)
    assert score(service, loan_amount=5000) > score(service, loan_amount=60000)


def without(payload, *keys):
    return {k: v for k, v in payload.items() if k not in keys}


def test_loan_amount_drives_loan_burden(service):
    # loan_burden = loan_amount / (annual_inc + 1), exactly as in training.
    # With an income of 1 the "+ 1" halves the ratio: 0.4 / 2 = 0.2.
    base = without({**FULL_PAYLOAD, "annual_inc": 1}, "loan_amount")
    derived = service.predict({**base, "loan_amount": 0.4})
    assert derived == service.predict({**base, "loan_burden": 0.2})
    assert derived != service.predict({**base, "loan_burden": 0.4})
    assert derived["missing_fields"] == []


def test_missing_optional_field_never_beats_disclosing_it(service):
    result = service.predict(dict(VALID_PAYLOAD))
    assert result["missing_fields"] == ["total_rev_hi_lim", "tot_cur_bal", "revol_util"]

    worst_values = {"total_rev_hi_lim": 0, "tot_cur_bal": 0, "revol_util": 100}
    for feature, worst in worst_values.items():
        blank = service.predict(without(FULL_PAYLOAD, feature))["credit_score"]
        assert blank == score(service, **{feature: worst})
    no_burden = service.predict(without(FULL_PAYLOAD, "loan_amount"))
    assert no_burden["missing_fields"] == ["loan_burden"]
    assert no_burden["credit_score"] == score(service, loan_amount=10_000_000)


@pytest.mark.parametrize(
    "feature,low,high",
    [("tot_cur_bal", 80597.5, 80623.5), ("total_rev_hi_lim", 23682.5, 23774.5)],
)
def test_values_near_the_training_median_are_not_treated_as_missing(
    service, feature, low, high
):
    # "Impute the median, then bin" created a narrow penalty bin (low, high]
    # around the training median. Real values there score like the bin below;
    # just above it the next real bin starts.
    below = score(service, **{feature: low - 1})
    assert score(service, **{feature: low}) == below
    assert score(service, **{feature: (low + high) / 2}) == below
    assert score(service, **{feature: high}) == below
    assert score(service, **{feature: high + 1}) > below


def achievable_score_range(service):
    """Best and worst score over every bin a request can actually land in."""
    woe = dict(service.pipeline.steps)["woe_transformer"]
    coefs = dict(zip(service.features, service.model.coef_[0]))
    low = high = service.model.intercept_[0]
    for feature_woe in service.features:
        feature = feature_woe[: -len("_woe")]
        artefact = service._imputation_bins.get(feature)
        contributions = [
            coefs[feature_woe] * w
            for interval, w in woe.woe_mappings[feature].items()
            if artefact is None or not np.isclose(interval.left, artefact[0])
        ]
        low += min(contributions)
        high += max(contributions)
    return service.offset - service.factor * high, service.offset - service.factor * low


def test_rating_bands_cover_the_achievable_score_range(service):
    worst, best = achievable_score_range(service)
    assert service.metadata["score_range"]["min"] == pytest.approx(worst, abs=0.01)
    assert service.metadata["score_range"]["max"] == pytest.approx(best, abs=0.01)
    # Every grade can be reached: the best case is the top grade, the worst
    # case is the bottom one, and bands are contiguous in between.
    assert service.rate(best)["rating"] == "AAA"
    assert service.rate(worst)["rating"] == "C"

    ratings = [band["rating"] for band in service.rating_bands]
    assert ratings == ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC"]
    assert service.rate(620)["rating"] == "AAA"
    assert service.rate(619.99)["rating"] == "AA"
    assert service.rate(515)["rating"] == "CC"
    assert service.rate(514.99)["rating"] == "C"


def test_extreme_applicants_hit_the_documented_range(service):
    best = service.predict(
        {
            "annual_inc": 200000,
            "int_rate": 6,
            "credit_history_length": 30,
            "purpose": "credit_card",
            "verification_status": "Not Verified",
            "total_rev_hi_lim": 100000,
            "tot_cur_bal": 500000,
            "revol_util": 5,
            "loan_amount": 5000,
        }
    )
    worst = service.predict(
        {
            "annual_inc": 20000,
            "int_rate": 25,
            "credit_history_length": 2,
            "purpose": "educational",
            "verification_status": "Verified",
            "loan_amount": 35000,
        }
    )
    assert (best["credit_score"], best["risk_level"]) == (645.76, "AAA")
    assert (worst["credit_score"], worst["risk_level"]) == (511.08, "C")


def test_predict_many_matches_predict(service):
    inputs = [
        dict(FULL_PAYLOAD),
        dict(VALID_PAYLOAD),
        {**VALID_PAYLOAD, "tot_cur_bal": 5000, "purpose": "small_business"},
        {**without(FULL_PAYLOAD, "loan_amount"), "loan_burden": 0.4},
    ]
    assert service.predict_many(inputs) == [service.predict(i) for i in inputs]
    assert service.predict_many([]) == []


def test_metadata_documents_scoring(service):
    with open(DEFAULT_MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    for key in ("PDO", "BaseScore", "BaseOdds", "rating_bands", "artifact_sha256"):
        assert key in metadata
    base_points = service.offset - service.factor * service.model.intercept_[0]
    assert metadata["base_points"] == pytest.approx(base_points, abs=0.01)
    assert np.isclose(service.factor, 20 / np.log(2))


def test_real_api_end_to_end(real_client):
    assert real_client.get("/api/v1/health").json()["status"] == "healthy"

    res = real_client.post("/api/v1/predict", json=FULL_PAYLOAD)
    assert res.status_code == 200
    assert res.json()["credit_score"] == 569.55
    assert res.json()["risk_level"] == "BB"

    batch = real_client.post(
        "/api/v1/predict/batch", json={"applications": [FULL_PAYLOAD, VALID_PAYLOAD]}
    )
    assert batch.status_code == 200
    predictions = batch.json()["predictions"]
    assert [p["credit_score"] for p in predictions] == [569.55, 562.45]
    assert predictions[1]["missing_fields"] == [
        "total_rev_hi_lim",
        "tot_cur_bal",
        "revol_util",
    ]

    info = real_client.get("/api/v1/model/info").json()
    assert [band["rating"] for band in info["rating_bands"]][-1] == "CC"


def test_startup_preloads_the_model(monkeypatch):
    import app.main as main_module
    from fastapi.testclient import TestClient

    loaded = []
    monkeypatch.setattr(main_module, "get_model_instance", lambda: loaded.append(1))
    with TestClient(main_module.app):
        assert loaded == [1]
