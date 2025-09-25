from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging
import time
from datetime import datetime

from app.api.schemas import (
    LoanApplicationRequest,
    CreditScoreResponse,
    HealthResponse,
    FeatureImportanceResponse,
    ModelInfoResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from app.models.model import CreditRiskModel
from app.utils.helpers import get_model_instance

logger = logging.getLogger(__name__)
router = APIRouter()


# Dependency to get model instance
def get_model() -> CreditRiskModel:
    """Get the model instance"""
    return get_model_instance()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=model.model is not None or len(model.woe_mappings) > 0,
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
        )


@router.post("/predict", response_model=CreditScoreResponse)
async def predict_credit_score(
    request: LoanApplicationRequest, model: CreditRiskModel = Depends(get_model)
):
    """Predict credit score for a single loan application"""
    try:
        start_time = time.time()

        # Convert request to dict
        input_data = request.dict()

        # Validate input
        is_valid, errors = model.validate_input(input_data)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Validation errors: {', '.join(errors)}"
            )

        # Get prediction
        result = model.predict_score(input_data)

        processing_time = time.time() - start_time

        return CreditScoreResponse(
            credit_score=result["credit_score"],
            default_probability=result["default_probability"],
            risk_level=result["risk_level"],
            log_odds=result.get("log_odds"),
            message=f"Prediction completed in {processing_time:.3f} seconds",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_credit_scores(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model: CreditRiskModel = Depends(get_model),
):
    """Predict credit scores for multiple loan applications"""
    try:
        start_time = time.time()

        if len(request.applications) > 100:  # Limit batch size
            raise HTTPException(
                status_code=400, detail="Batch size cannot exceed 100 applications"
            )

        predictions = []

        for i, application in enumerate(request.applications):
            try:
                # Convert to dict and validate
                input_data = application.dict()
                is_valid, errors = model.validate_input(input_data)

                if not is_valid:
                    # Add error prediction for invalid applications
                    predictions.append(
                        CreditScoreResponse(
                            credit_score=0.0,
                            default_probability=1.0,
                            risk_level="Invalid Input",
                            log_odds=None,
                            message=f"Validation errors: {', '.join(errors)}",
                        )
                    )
                    continue

                # Get prediction
                result = model.predict_score(input_data)
                predictions.append(
                    CreditScoreResponse(
                        credit_score=result["credit_score"],
                        default_probability=result["default_probability"],
                        risk_level=result["risk_level"],
                        log_odds=result.get("log_odds"),
                        message="Batch prediction completed",
                    )
                )

            except Exception as e:
                logger.error(f"Error processing application {i}: {str(e)}")
                predictions.append(
                    CreditScoreResponse(
                        credit_score=0.0,
                        default_probability=1.0,
                        risk_level="Processing Error",
                        log_odds=None,
                        message=f"Error: {str(e)}",
                    )
                )

        processing_time = time.time() - start_time

        return BatchPredictionResponse(
            predictions=predictions,
            total_applications=len(request.applications),
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(model: CreditRiskModel = Depends(get_model)):
    """Get information about the current model"""
    try:
        return ModelInfoResponse(
            model_type="Logistic Regression with WOE Transformations",
            features_used=model.final_woe_features,
            scoring_params=model.scoring_params,
            training_date="2024-01-01",  # This would be loaded from model metadata
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/model/features/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(model: CreditRiskModel = Depends(get_model)):
    """Get feature importance scores"""
    try:
        importance = model.get_feature_importance()
        return FeatureImportanceResponse(
            feature_importance=importance, total_features=len(importance)
        )
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get feature importance: {str(e)}"
        )


@router.get("/model/features")
async def get_available_features():
    """Get list of available features for prediction"""
    try:
        model = get_model()
        return {
            "features": model.final_woe_features,
            "required_features": [
                "annual_inc",
                "int_rate",
                "purpose",
                "verification_status",
            ],
            "optional_features": [
                "total_rev_hi_lim",
                "tot_cur_bal",
                "loan_burden",
                "revol_util",
                "credit_history_length",
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get features: {str(e)}")


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Risk Scorecard API",
        "version": "1.0.0",
        "description": "API for predicting credit risk and generating credit scores",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "model_info": "/api/v1/model/info",
            "feature_importance": "/api/v1/model/features/importance",
            "features": "/api/v1/model/features",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "documentation": {
            "swagger_ui": "/docs",
            "reDoc": "/redoc",
            "openapi_schema": "/openapi.json",
        },
    }


# Error handlers are moved to main.py since they need to be on the FastAPI app instance
