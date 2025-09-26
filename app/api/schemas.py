from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum


class LoanPurpose(str, Enum):
    """Enum for loan purposes"""

    CREDIT_CARD = "credit_card"
    DEBT_CONSOLIDATION = "debt_consolidation"
    HOME_IMPROVEMENT = "home_improvement"
    CAR = "car"
    EDUCATIONAL = "educational"
    OTHER = "other"


class VerificationStatus(str, Enum):
    """Enum for verification status (matches training data labels)"""

    VERIFIED = "Verified"
    SOURCE_VERIFIED = "Source Verified"
    NOT_VERIFIED = "Not Verified"


class LoanApplicationRequest(BaseModel):
    """Request model for loan application"""

    annual_inc: float = Field(..., description="Annual income in USD", gt=0)
    int_rate: float = Field(..., description="Interest rate percentage", ge=0, le=50)
    credit_history_length: float = Field(
        ..., description="Credit history length in years", ge=0
    )
    total_rev_hi_lim: Optional[float] = Field(
        None, description="Total revolving high limit", ge=0
    )
    tot_cur_bal: Optional[float] = Field(
        None, description="Total current balance", ge=0
    )
    purpose: LoanPurpose = Field(..., description="Loan purpose")
    loan_burden: Optional[float] = Field(None, description="Loan burden ratio", ge=0)
    revol_util: Optional[float] = Field(
        None, description="Revolving utilization percentage", ge=0, le=100
    )
    verification_status: VerificationStatus = Field(
        ..., description="Verification status"
    )
    loan_amount: Optional[float] = Field(None, description="Loan amount in USD", gt=0)

    @field_validator("annual_inc")
    def validate_annual_income(cls, v):
        if v <= 0:
            raise ValueError("Annual income must be positive")
        if v > 10000000:  # 10 million
            raise ValueError("Annual income seems unreasonably high")
        return v

    @field_validator("int_rate")
    def validate_interest_rate(cls, v):
        if v < 0 or v > 50:
            raise ValueError("Interest rate must be between 0 and 50 percent")
        return v

    @field_validator("credit_history_length")
    def validate_credit_history(cls, v):
        if v < 0:
            raise ValueError("Credit history length cannot be negative")
        if v > 100:  # 100 years seems unreasonable
            raise ValueError("Credit history length seems unreasonably high")
        return v


class CreditScoreResponse(BaseModel):
    """Response model for credit score prediction"""

    credit_score: float = Field(..., description="Calculated credit score")
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_level: str = Field(..., description="Risk level assessment")
    log_odds: Optional[float] = Field(None, description="Log odds value")
    message: Optional[str] = Field(None, description="Additional information")


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    timestamp: Optional[str] = Field(None, description="Response timestamp")


class FeatureImportanceResponse(BaseModel):
    """Feature importance response model"""

    feature_importance: Dict[str, float] = Field(
        ..., description="Feature importance scores"
    )
    total_features: int = Field(..., description="Total number of features")


class ModelInfoResponse(BaseModel):
    """Model information response model"""

    model_type: str = Field(..., description="Type of model")
    features_used: List[str] = Field(..., description="List of features used")
    scoring_params: Dict[str, float] = Field(..., description="Scoring parameters")
    training_date: Optional[str] = Field(None, description="Model training date")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    applications: List[LoanApplicationRequest] = Field(
        ..., description="List of loan applications"
    )
    include_details: bool = Field(
        False, description="Whether to include detailed results"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[CreditScoreResponse] = Field(
        ..., description="List of predictions"
    )
    total_applications: int = Field(
        ..., description="Total number of applications processed"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
