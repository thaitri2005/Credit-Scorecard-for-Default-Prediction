import pandas as pd
import numpy as np
from app.models.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for the DataPreprocessor class"""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert isinstance(preprocessor.bin_thresholds, dict)
        assert isinstance(preprocessor.woe_mappings, dict)

    def test_loan_status_mapping(self):
        """Test loan status mapping"""
        preprocessor = DataPreprocessor()

        # Test bad statuses
        assert preprocessor.map_loan_status("Charged Off") == 1
        assert preprocessor.map_loan_status("Default") == 1
        assert preprocessor.map_loan_status("Late (31-120 days)") == 1

        # Test good statuses
        assert preprocessor.map_loan_status("Fully Paid") == 0
        assert preprocessor.map_loan_status("Current") == 0

        # Test ambiguous statuses
        assert preprocessor.map_loan_status("Issued") == -1
        assert preprocessor.map_loan_status("In Grace Period") == -1

    def test_quantile_binning(self):
        """Test quantile binning"""
        preprocessor = DataPreprocessor()

        # Create test data
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        bins = preprocessor.quantile_binning(data, n_bins=3)

        assert len(bins) == 4  # n_bins + 1
        assert bins[0] == -np.inf
        assert bins[-1] == np.inf
        assert bins[1] <= bins[2] <= bins[3]

    def test_decision_tree_binning(self):
        """Test decision tree binning"""
        preprocessor = DataPreprocessor()

        # Create test data with clear separation
        np.random.seed(42)
        X = pd.Series(np.random.normal(0, 1, 100))
        y = pd.Series((X > 0).astype(int))  # Simple binary split

        bins = preprocessor.decision_tree_binning(X, y, max_leaf_nodes=3)

        assert len(bins) >= 2
        assert bins[0] == -np.inf
        assert bins[-1] == np.inf

    def test_woe_iv_calculation(self):
        """Test WOE and IV calculation"""
        preprocessor = DataPreprocessor()

        # Create test data
        df = pd.DataFrame(
            {"feature": ["A", "A", "B", "B", "C", "C"], "target": [0, 1, 0, 1, 0, 1]}
        )

        woe_dict, iv_score = preprocessor.calculate_woe_iv(df, "feature", "target")

        assert isinstance(woe_dict, dict)
        assert isinstance(iv_score, float)
        assert len(woe_dict) == 3  # Three unique values
        assert "A" in woe_dict
        assert "B" in woe_dict
        assert "C" in woe_dict

    def test_clean_data_basic(self):
        """Test basic data cleaning"""
        preprocessor = DataPreprocessor()

        # Create test data
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "loan_status": ["Fully Paid", "Charged Off", "Current"],
                "annual_inc": [50000, 60000, 70000],
                "loan_amnt": [10000, 15000, 20000],
                "issue_d": ["Jan-2020", "Feb-2020", "Mar-2020"],
                "earliest_cr_line": ["Jan-2015", "Feb-2015", "Mar-2015"],
            }
        )

        cleaned_df = preprocessor.clean_data(df)

        assert "target" in cleaned_df.columns
        assert "id" not in cleaned_df.columns  # Should be dropped
        assert "loan_status" not in cleaned_df.columns  # Should be dropped
        assert "credit_history_length" in cleaned_df.columns
        assert "loan_burden" in cleaned_df.columns
        assert len(cleaned_df) == 3  # All rows should be kept

    def test_impute_missing_values(self):
        """Test missing value imputation"""
        preprocessor = DataPreprocessor()

        # Create test data with missing values
        df = pd.DataFrame(
            {
                "annual_inc": [50000, np.nan, 70000],
                "emp_length": ["5 years", np.nan, "10+ years"],
                "target": [0, 1, 0],
            }
        )

        preprocessor._impute_missing_values(df)

        # Check that missing values are filled
        assert not df["annual_inc"].isna().any()
        assert not df["emp_length"].isna().any()

    def test_feature_summary(self):
        """Test feature summary generation"""
        preprocessor = DataPreprocessor()

        # Create test data
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, np.nan],
                "feature2": ["A", "B", "A", "B"],
                "target": [0, 1, 0, 1],
            }
        )

        summary = preprocessor.get_feature_summary(df)

        assert "total_rows" in summary
        assert "total_columns" in summary
        assert "missing_values" in summary
        assert "data_types" in summary
        assert "target_distribution" in summary

        assert summary["total_rows"] == 4
        assert summary["total_columns"] == 3
        assert summary["missing_values"]["feature1"] == 1
        assert summary["target_distribution"][0] == 2
        assert summary["target_distribution"][1] == 2


class TestModelIntegration:
    """Integration tests for the complete model pipeline"""

    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline"""
        from app.models.model import CreditRiskModel

        model = CreditRiskModel()

        # Test with realistic data
        test_cases = [
            {
                "annual_inc": 75000,
                "int_rate": 12.5,
                "credit_history_length": 5.5,
                "purpose": "debt_consolidation",
                "verification_status": "verified",
            },
            {
                "annual_inc": 45000,
                "int_rate": 18.0,
                "credit_history_length": 2.0,
                "purpose": "car",
                "verification_status": "not_verified",
            },
            {
                "annual_inc": 120000,
                "int_rate": 8.5,
                "credit_history_length": 10.0,
                "purpose": "home_improvement",
                "verification_status": "verified",
            },
        ]

        for test_case in test_cases:
            result = model.predict_score(test_case)

            # Validate result structure
            assert "credit_score" in result
            assert "default_probability" in result
            assert "risk_level" in result

            # Validate score range (typical credit scores)
            assert 300 <= result["credit_score"] <= 850

            # Validate probability range
            assert 0 <= result["default_probability"] <= 1

            # Validate risk level
            assert result["risk_level"] in [
                "Low Risk",
                "Medium Risk",
                "High Risk",
                "Very High Risk",
            ]

    def test_model_consistency(self):
        """Test model consistency across multiple predictions"""
        from app.models.model import CreditRiskModel

        model = CreditRiskModel()

        test_data = {
            "annual_inc": 75000,
            "int_rate": 12.5,
            "credit_history_length": 5.5,
            "purpose": "debt_consolidation",
            "verification_status": "verified",
        }

        # Run prediction multiple times
        results = []
        for _ in range(5):
            result = model.predict_score(test_data)
            results.append(result)

        # All results should be identical (deterministic model)
        for i in range(1, len(results)):
            assert results[i]["credit_score"] == results[0]["credit_score"]
            assert (
                results[i]["default_probability"] == results[0]["default_probability"]
            )
            assert results[i]["risk_level"] == results[0]["risk_level"]
