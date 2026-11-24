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
        "loan_amount": 15000,
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
        ("revol_util", 101),
        ("loan_amount", 0),
        ("tot_cur_bal", float("inf")),
        ("tot_cur_bal", float("nan")),
        ("annual_inc", True),
        ("annual_inc", "75000"),
        ("revol_util", "40"),
        ("purpose", "yacht"),
        ("verification_status", "verified"),
    ],
)
def test_schema_invalid_values(field, value):
    data = make_valid_request(**{field: value})
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**data)


def test_schema_optional_fields():
    data = make_valid_request(total_rev_hi_lim=10000, tot_cur_bal=5000, revol_util=30.0)
    obj = LoanApplicationRequest(**data)
    assert obj.total_rev_hi_lim == 10000
    assert obj.revol_util == 30.0


def test_schema_needs_exactly_one_of_loan_amount_and_loan_burden():
    data = make_valid_request()
    del data["loan_amount"]
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**data)
    assert LoanApplicationRequest(**data, loan_burden=0.2).loan_burden == 0.2
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**data, loan_burden=0.2, loan_amount=15000)


def test_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**make_valid_request(loan_amnt=5000))


def test_schema_accepts_every_trained_purpose():
    assert len(LoanPurpose) == 14
    for purpose in LoanPurpose:
        LoanApplicationRequest(**make_valid_request(purpose=purpose.value))
