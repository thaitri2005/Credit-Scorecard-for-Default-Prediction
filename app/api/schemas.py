from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BATCH_SIZE = 1000


class LoanPurpose(str, Enum):
    """Loan purposes the model was trained on"""

    CREDIT_CARD = "credit_card"
    DEBT_CONSOLIDATION = "debt_consolidation"
    HOME_IMPROVEMENT = "home_improvement"
    CAR = "car"
    EDUCATIONAL = "educational"
    HOUSE = "house"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    MOVING = "moving"
    RENEWABLE_ENERGY = "renewable_energy"
    SMALL_BUSINESS = "small_business"
    VACATION = "vacation"
    WEDDING = "wedding"
    OTHER = "other"


class VerificationStatus(str, Enum):
    """Enum for verification status (matches training data labels)"""

    VERIFIED = "Verified"
    SOURCE_VERIFIED = "Source Verified"
    NOT_VERIFIED = "Not Verified"


class LoanApplicationRequest(BaseModel):
    """Request model for loan application"""

    # Reject unknown fields (catches typos such as "loan_amnt") and NaN/Infinity.
    # Numeric fields are strict: booleans and numeric strings are not accepted.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    annual_inc: float = Field(
        ..., strict=True, description="Annual income in USD", gt=0, le=10_000_000
    )
    int_rate: float = Field(
        ..., strict=True, description="Loan interest rate in percent", ge=0, le=50
    )
    credit_history_length: float = Field(
        ..., strict=True, description="Credit history length in years", ge=0, le=100
    )
    purpose: LoanPurpose = Field(..., description="Loan purpose")
    verification_status: VerificationStatus = Field(
        ..., description="Income verification status"
    )
    total_rev_hi_lim: Optional[float] = Field(
        None, strict=True, description="Total revolving credit limit in USD", ge=0
    )
    tot_cur_bal: Optional[float] = Field(
        None,
        strict=True,
        description="Total current balance of all accounts in USD",
        ge=0,
    )
    revol_util: Optional[float] = Field(
        None, strict=True, description="Revolving utilization in percent", ge=0, le=100
    )
    loan_amount: Optional[float] = Field(
        None,
        strict=True,
        description="Loan amount in USD. Send this or loan_burden.",
        gt=0,
    )
    loan_burden: Optional[float] = Field(
        None,
        strict=True,
        description=(
            "Loan amount divided by annual income, for callers that already "
            "have the ratio. Send this or loan_amount, not both."
        ),
        ge=0,
    )

    @model_validator(mode="after")
    def require_exactly_one_loan_size(self):
        if (self.loan_amount is None) == (self.loan_burden is None):
            raise ValueError("Provide exactly one of loan_amount or loan_burden")
        return self


class FeaturePoints(BaseModel):
    """Points one input contributed to the credit score"""

    feature: str
    points: float


class CreditScoreResponse(BaseModel):
    """Response model for credit score prediction"""

    credit_score: float = Field(..., description="Calculated credit score")
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_level: str = Field(..., description="Internal risk grade (AAA to C)")
    risk_description: Optional[str] = Field(
        None, description="Plain-language meaning of the risk grade"
    )
    log_odds: Optional[float] = Field(None, description="Log odds of default")
    base_points: Optional[float] = Field(
        None, description="Score of an applicant with average evidence on every input"
    )
    points: List[FeaturePoints] = Field(
        default_factory=list,
        description="Points each input adds to base_points; together they give the score",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description=(
            "Optional inputs that were not supplied. Each is scored like the "
            "least favorable value of that input."
        ),
    )
    message: Optional[str] = Field(None, description="Additional information")


class HealthResponse(BaseModel):
    """Health check response model"""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Model version")
    timestamp: Optional[str] = Field(None, description="Response timestamp")


class RatingBand(BaseModel):
    """One risk grade and the minimum score that earns it"""

    min_score: float
    rating: str
    description: str


class ModelInfoResponse(BaseModel):
    """Model information response model"""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(..., description="Type of model")
    features_used: List[str] = Field(..., description="List of features used")
    scoring_params: Dict[str, float] = Field(..., description="Scoring parameters")
    rating_bands: List[RatingBand] = Field(
        default_factory=list, description="Score thresholds for each risk grade"
    )
    lowest_rating: Optional[str] = Field(
        None, description="Grade given to scores below every threshold"
    )
    score_range: Dict[str, float] = Field(
        default_factory=dict, description="Lowest and highest score the model can give"
    )
    training_date: Optional[str] = Field(None, description="Model training date")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    model_config = ConfigDict(extra="forbid")

    applications: List[LoanApplicationRequest] = Field(
        ...,
        description="List of loan applications",
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[CreditScoreResponse] = Field(
        ..., description="Predictions, in the same order as the applications"
    )
    total_applications: int = Field(
        ..., description="Total number of applications processed"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
