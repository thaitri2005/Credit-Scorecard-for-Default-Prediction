from fastapi import APIRouter, HTTPException, Depends
import logging, time
from datetime import datetime
from app.api.schemas import LoanApplicationRequest, CreditScoreResponse, HealthResponse, ModelInfoResponse, BatchPredictionRequest, BatchPredictionResponse
from app.utils.helpers import get_model_instance, CreditRiskService

logger = logging.getLogger(__name__)
router = APIRouter()

def get_model() -> CreditRiskService:
    return get_model_instance()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            version=model.metadata.get("version", "1.0.0"),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version="unknown",
            timestamp=datetime.now().isoformat(),
        )

@router.post("/predict", response_model=CreditScoreResponse)
async def predict_credit_score(
    request: LoanApplicationRequest, model: CreditRiskService = Depends(get_model)
):
    try:
        start = time.time()
        result = model.predict(request.dict())
        return CreditScoreResponse(**result, message=f"Prediction completed in {time.time()-start:.3f}s")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest, model: CreditRiskService = Depends(get_model)
):
    results = []
    for app in request.applications:
        try:
            results.append(CreditScoreResponse(**model.predict(app.dict()), message="Batch prediction completed"))
        except Exception as e:
            results.append(CreditScoreResponse(
                credit_score=0, default_probability=1, risk_level="Error", log_odds=None, message=str(e)
            ))
    return BatchPredictionResponse(predictions=results, total_applications=len(request.applications))
    
@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(model: CreditRiskService = Depends(get_model)):
    return ModelInfoResponse(
        model_type="Logistic Regression with WOE pipeline",
        features_used=model.metadata.get("features", []),
        scoring_params={
            "PDO": model.metadata.get("PDO", 20),
            "BaseScore": model.metadata.get("BaseScore", 600),
            "BaseOdds": model.metadata.get("BaseOdds", 50),
        },
        training_date=model.metadata.get("training_date"),
    )
