"""Tests for the pipeline steps, including the fitting logic used in training."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.scoring.transformers import (
    ColumnDropper,
    FeatureEngineer,
    MissingValueImputer,
    WOETransformer,
)


@pytest.fixture()
def loans() -> pd.DataFrame:
    """Synthetic loans where a high interest rate and a risky purpose default more."""
    rng = np.random.default_rng(7)
    n = 4000
    int_rate = rng.uniform(5, 28, n)
    purpose = rng.choice(["credit_card", "small_business"], n)
    risk = 0.02 + 0.012 * (int_rate - 5) + 0.10 * (purpose == "small_business")
    return pd.DataFrame(
        {
            "int_rate": int_rate,
            "annual_inc": rng.uniform(20_000, 150_000, n),
            "purpose": purpose,
            "target": (rng.uniform(0, 1, n) < risk).astype(int),
        }
    )


def test_column_dropper_removes_only_listed_columns():
    frame = pd.DataFrame({"id": [1], "recoveries": [0.0], "int_rate": [9.5]})
    result = ColumnDropper().fit(frame).transform(frame)
    assert list(result.columns) == ["int_rate"]
    assert list(frame.columns) == ["id", "recoveries", "int_rate"]  # input untouched


def test_feature_engineer_derives_loan_burden():
    frame = pd.DataFrame(
        {
            "loan_amnt": [15000.0],
            "annual_inc": [74999.0],
            "dti": [12.0],
            "issue_d": ["Dec-2015"],
            "earliest_cr_line": ["Jan-2005"],
        }
    )
    result = FeatureEngineer().fit(frame).transform(frame)
    assert result["loan_burden"].iloc[0] == pytest.approx(0.2)
    assert set(result.columns) == {"annual_inc", "loan_burden"}


def test_feature_engineer_leaves_frames_without_loan_amount_alone():
    frame = pd.DataFrame({"annual_inc": [50000.0], "loan_burden": [0.3]})
    result = FeatureEngineer().transform(frame)
    assert result["loan_burden"].iloc[0] == 0.3


def test_imputer_learns_medians_and_modes_and_fills_gaps():
    train = pd.DataFrame(
        {
            "revol_util": [10.0, 30.0, 50.0, np.nan],
            "emp_length": ["2 years", "2 years", "10+ years", None],
            "tot_cur_bal": [np.nan, np.nan, np.nan, np.nan],
            "untouched": [1.0, np.nan, 3.0, 4.0],
        }
    )
    imputer = MissingValueImputer().fit(train)
    assert imputer.median_values == {"tot_cur_bal": 0, "revol_util": 30.0}
    assert imputer.mode_values == {"emp_length": "2 years"}

    new = pd.DataFrame(
        {
            "revol_util": [np.nan],
            "emp_length": [None],
            "tot_cur_bal": [np.nan],
            "untouched": [np.nan],
        }
    )
    filled = imputer.transform(new)
    assert filled["revol_util"].iloc[0] == 30.0
    assert filled["emp_length"].iloc[0] == "2 years"
    assert filled["tot_cur_bal"].iloc[0] == 0
    assert np.isnan(filled["untouched"].iloc[0])  # not in the imputation list


def test_imputer_falls_back_when_a_category_is_entirely_missing():
    imputer = MissingValueImputer().fit(pd.DataFrame({"emp_length": [None, None]}))
    assert imputer.mode_values == {"emp_length": "Unknown"}


def test_woe_fit_bins_numeric_features_and_orders_risk(loans):
    woe = WOETransformer(max_leaf_nodes=4).fit(loans)

    edges = woe.bins["int_rate"]
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert edges == sorted(edges)
    assert 3 <= len(edges) <= 5  # at most max_leaf_nodes bins

    # WOE = ln(share of good / share of bad): riskier bins score lower.
    by_bin = sorted(woe.woe_mappings["int_rate"].items(), key=lambda item: item[0].left)
    values = [value for _, value in by_bin]
    assert values == sorted(values, reverse=True)
    assert values[0] > 0 > values[-1]

    assert (
        woe.woe_mappings["purpose"]["credit_card"]
        > woe.woe_mappings["purpose"]["small_business"]
    )


def test_woe_matches_the_definition(loans):
    woe = WOETransformer()
    mapping, information_value = woe.calc_woe_iv(loans, "purpose")
    good = loans[loans.target == 0].purpose.value_counts(normalize=True)
    bad = loans[loans.target == 1].purpose.value_counts(normalize=True)
    for category in ("credit_card", "small_business"):
        assert mapping[category] == pytest.approx(
            np.log(good[category] / bad[category]), abs=1e-3
        )
    assert information_value > 0


def test_woe_transform_adds_a_woe_column_per_fitted_feature(loans):
    woe = WOETransformer(max_leaf_nodes=4).fit(loans)
    applicant = pd.DataFrame(
        {
            "int_rate": [6.0, 27.0],
            "annual_inc": [90_000.0, 30_000.0],
            "purpose": ["credit_card"] * 2,
        }
    )
    result = woe.transform(applicant)
    assert {"int_rate_woe", "annual_inc_woe", "purpose_woe"} <= set(result.columns)
    assert float(result["int_rate_woe"].iloc[0]) > float(result["int_rate_woe"].iloc[1])
    assert "int_rate_woe" not in applicant.columns  # input untouched


def test_woe_fit_without_a_target_learns_nothing(loans):
    woe = WOETransformer().fit(loans.drop(columns="target"))
    assert woe.bins == {} and woe.woe_mappings == {}
    # Transforming with an unfitted step adds no columns.
    result = woe.transform(loans.drop(columns="target"))
    assert not [column for column in result.columns if column.endswith("_woe")]


def test_woe_fit_skips_a_feature_it_cannot_bin(loans):
    broken = loans.assign(int_rate="not a number")
    woe = WOETransformer().fit(broken)
    assert "int_rate" not in woe.bins
    assert "annual_inc" in woe.bins  # other features are still fitted


def test_steps_compose_into_a_pipeline(loans):
    raw = loans.assign(loan_amnt=10_000.0, id=range(len(loans)))
    raw.loc[raw.index[:50], "annual_inc"] = np.nan
    pipeline = Pipeline(
        [
            ("column_dropper", ColumnDropper()),
            ("feature_engineer", FeatureEngineer()),
            ("imputer", MissingValueImputer()),
            ("woe_transformer", WOETransformer(max_leaf_nodes=4)),
        ]
    )
    result = pipeline.fit_transform(raw)
    assert "id" not in result.columns
    for column in ("int_rate_woe", "annual_inc_woe", "loan_burden_woe", "purpose_woe"):
        assert result[column].notna().all()
