import pytest
from pydantic import ValidationError

from app.api.schemas import LoanApplicationRequest, LoanPurpose, VerificationStatus


def make_valid_request(**overrides):
    base = {
        "annual_inc": 75000,
        "int_rate": 12.5,
        "credit_history_length": 5.5,
        "purpose": LoanPurpose.DEBT_CONSOLIDATION.value,
        "verification_status": VerificationStatus.VERIFIED.value,
    }
    base.update(overrides)
    return base


def test_schema_valid_minimal():
    data = make_valid_request()
    obj = LoanApplicationRequest(**data)
    assert obj.annual_inc == 75000
    assert obj.int_rate == 12.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("annual_inc", 0),
        ("annual_inc", -1),
        ("int_rate", -0.1),
        ("int_rate", 55.0),
        ("credit_history_length", -0.5),
        ("credit_history_length", 150.0),
    ],
)
def test_schema_invalid_values(field, value):
    data = make_valid_request(**{field: value})
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**data)


def test_schema_optional_fields():
    data = make_valid_request(
        total_rev_hi_lim=10000,
        tot_cur_bal=5000,
        loan_burden=0.2,
        revol_util=30.0,
        loan_amount=10000,
    )
    obj = LoanApplicationRequest(**data)
    assert obj.total_rev_hi_lim == 10000
    assert obj.revol_util == 30.0
