import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CreditScoreResponse,
    HealthResponse,
    LoanApplicationRequest,
    ModelInfoResponse,
    RatingBand,
)
from app.utils.helpers import CreditRiskService, get_model_instance

logger = logging.getLogger(__name__)
router = APIRouter()

# Handlers are plain ``def`` on purpose: scoring is CPU-bound pandas/sklearn
# work, so FastAPI runs it in its threadpool instead of blocking the event loop.


def get_model() -> CreditRiskService:
    try:
        return get_model_instance()
    except Exception:
        raise HTTPException(status_code=503, detail="Model is not available")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get(
    "/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}
)
def health_check():
    try:
        model = get_model()
    except Exception:
        body = HealthResponse(
            status="unhealthy", model_loaded=False, version="unknown", timestamp=_now()
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version=str(model.metadata.get("version", "unknown")),
        timestamp=_now(),
    )


def _describe(result: dict) -> str:
    message = result.get("risk_description") or "Prediction completed"
    missing = result.get("missing_fields") or []
    if missing:
        message += f". Not provided; assigned the least favorable attribute: {', '.join(missing)}"
    return message


@router.post("/predict", response_model=CreditScoreResponse)
def predict_credit_score(
    request: LoanApplicationRequest, model: CreditRiskService = Depends(get_model)
):
    try:
        result = model.predict(request.model_dump(mode="json"))
    except Exception:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")
    return CreditScoreResponse(**result, message=_describe(result))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest, model: CreditRiskService = Depends(get_model)
):
    start = time.perf_counter()
    try:
        results = model.predict_many(
            [app.model_dump(mode="json") for app in request.applications]
        )
    except Exception:
        logger.error("Batch prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction failed")
    return BatchPredictionResponse(
        predictions=[
            CreditScoreResponse(**result, message=_describe(result))
            for result in results
        ],
        total_applications=len(results),
        processing_time=round(time.perf_counter() - start, 4),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(model: CreditRiskService = Depends(get_model)):
    return ModelInfoResponse(
        model_type="Logistic Regression with WOE pipeline",
        features_used=model.features,
        scoring_params=model.scoring_params,
        rating_bands=[RatingBand(**band) for band in model.rating_bands],
        lowest_rating=model.lowest_rating["rating"],
        score_range=model.metadata.get("score_range", {}),
        training_date=model.metadata.get("training_date"),
    )
