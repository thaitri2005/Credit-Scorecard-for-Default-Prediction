import pickle
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Credit Risk Scorecard Model

    This class encapsulates the trained logistic regression model with WOE transformations
    and provides methods for credit score prediction.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.woe_mappings = {}
        self.bin_thresholds = {}
        self.scoring_params = {
            "PDO": 20,  # Points to Double the Odds
            "BaseScore": 600,  # Base score
            "BaseOdds": 50,  # Base odds ratio
        }

        # Final WOE features from the notebook
        self.final_woe_features = [
            "int_rate_woe",
            "total_rev_hi_lim_woe",
            "tot_cur_bal_woe",
            "annual_inc_woe",
            "purpose_woe",
            "loan_burden_woe",
            "credit_history_length_woe",
            "revol_util_woe",
            "verification_status_woe",
        ]

        # WOE mappings extracted from notebook (simplified for demo)
        self._initialize_woe_mappings()

    def _initialize_woe_mappings(self):
        """Initialize WOE mappings based on the notebook results"""
        # These are simplified mappings - in production, these would be loaded from training
        self.woe_mappings = {
            "int_rate": {
                "(-inf, 7.275]": 1.8652,
                "(7.275, 9.995]": 1.0530,
                "(9.995, 14.03]": 0.2890,
                "(14.03, 18.58]": -0.3812,
                "(18.58, inf]": -0.9700,
            },
            "annual_inc": {
                "(-inf, 43202.0]": -0.2922,
                "(43202.0, 66100.5]": -0.0839,
                "(66100.5, 80046.219]": 0.0909,
                "(80046.219, 100129.0]": 0.2157,
                "(100129.0, inf]": 0.3962,
            },
            "purpose": {
                "credit_card": 0.3129,
                "car": 0.1043,
                "debt_consolidation": -0.0432,
                "educational": -1.2466,
                "home_improvement": 0.0,  # Default value
                "other": 0.0,
            },
            "verification_status": {
                "Verified": 0.0,
                "Source Verified": 0.0,
                "Not Verified": 0.0,
            },
        }

        # Model coefficients from notebook
        self.model_coefficients = {
            "int_rate_woe": -0.9463,
            "total_rev_hi_lim_woe": -0.2874,
            "tot_cur_bal_woe": -0.7391,
            "annual_inc_woe": -0.3650,
            "purpose_woe": -0.3000,
            "loan_burden_woe": -0.2106,
            "credit_history_length_woe": -0.3059,
            "revol_util_woe": -0.2186,
            "verification_status_woe": -0.3554,
        }

        # Intercept from notebook
        self.intercept = 0.0  # This would be the actual intercept from training

    def load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                self.model = model_data["model"]
                self.feature_names = model_data["feature_names"]
                self.woe_mappings = model_data["woe_mappings"]
                self.bin_thresholds = model_data["bin_thresholds"]
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(
                f"Model file {model_path} not found. Using default mappings."
            )
            self._initialize_woe_mappings()

    def predict_score(self, input_data: Dict) -> Dict:
        """
        Predict credit score for given input data

        Args:
            input_data: Dictionary containing loan application data

        Returns:
            Dictionary with credit_score, default_probability, and risk_level
        """
        try:
            # Preprocess input data
            processed_data = self._preprocess_input(input_data)

            # Calculate log-odds manually (since we don't have the actual trained model)
            log_odds = self.intercept
            for feature, coef in self.model_coefficients.items():
                base_feature = feature.replace("_woe", "")
                woe_value = self._get_woe_value(
                    base_feature, processed_data.get(base_feature, 0)
                )
                log_odds += coef * woe_value

            # Convert log-odds to probability
            probability = 1 / (1 + np.exp(-log_odds))

            # Calculate score using standard scorecard formula
            factor = self.scoring_params["PDO"] / np.log(2)
            offset = self.scoring_params["BaseScore"] - factor * np.log(
                self.scoring_params["BaseOdds"]
            )
            score = offset - factor * log_odds

            return {
                "credit_score": round(score, 2),
                "default_probability": round(probability, 4),
                "risk_level": self._get_risk_level(score),
                "log_odds": round(log_odds, 4),
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def _preprocess_input(self, data: Dict) -> Dict:
        """Preprocess input data"""
        processed: Dict[str, float | str] = {}

        # Handle missing values and data type conversions
        for key, value in data.items():
            if value is None or value == "":
                # Use 0.0 for numeric fields, "" for string fields
                if key in ["purpose", "verification_status"]:
                    processed[key] = ""
                else:
                    processed[key] = 0.0
            elif isinstance(value, str):
                # Only lower-case for string fields that are not expected to be numeric
                if key in ["purpose", "verification_status"]:
                    processed[key] = value.strip().lower()
                else:
                    try:
                        processed[key] = float(value)
                    except ValueError:
                        processed[key] = 0.0
            elif isinstance(value, (int, float)):
                processed[key] = float(value)
            else:
                # For any other type, try to convert to float, else set to 0.0
                try:
                    processed[key] = float(value)
                except Exception:
                    processed[key] = 0.0

        # Calculate loan_burden if not provided
        if "loan_burden" not in processed and "annual_inc" in processed:
            # Estimate loan amount (this is a simplification)
            estimated_loan_amount = processed.get(
                "loan_amount", 15000.0
            )  # Default estimate
            try:
                annual_inc = float(processed["annual_inc"])
            except (ValueError, TypeError):
                annual_inc = 0.0
            try:
                processed["loan_burden"] = float(estimated_loan_amount) / (
                    annual_inc + 1.0
                )
            except Exception:
                processed["loan_burden"] = 0.0

        return processed

    def _calculate_woe_features(self, data: Dict) -> List[float]:
        """Calculate WOE features for prediction"""
        woe_features = []
        for feature in self.final_woe_features:
            base_feature = feature.replace("_woe", "")
            woe_value = self._get_woe_value(base_feature, data.get(base_feature, 0))
            woe_features.append(woe_value)
        return woe_features

    def _get_woe_value(self, feature: str, value: float) -> float:
        """Get WOE value for a feature"""
        if feature in self.woe_mappings:
            if feature in ["purpose", "verification_status"]:
                # Categorical features
                return self.woe_mappings[feature].get(str(value), 0.0)
            else:
                # Numerical features - find appropriate bin
                for bin_range, woe in self.woe_mappings[feature].items():
                    if self._value_in_bin(value, bin_range):
                        return woe
        return 0.0

    def _value_in_bin(self, value: float, bin_range: str) -> bool:
        """Check if value falls within bin range"""
        try:
            # Parse bin range like "(-inf, 43202.0]" or "(100129.0, inf]"
            bin_range = bin_range.strip("()[]")
            parts = bin_range.split(",")
            if len(parts) == 2:
                left, right = parts
                left_val = float(left) if left != "-inf" else float("-inf")
                right_val = float(right) if right != "inf" else float("inf")
                return left_val < value <= right_val
        except (ValueError, TypeError, AttributeError):
            pass
        return False

    def _get_risk_level(self, score: float) -> str:
        """Determine risk level based on score"""
        if score >= 700:
            return "Low Risk"
        elif score >= 600:
            return "Medium Risk"
        elif score >= 500:
            return "High Risk"
        else:
            return "Very High Risk"

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on coefficients"""
        return {feature: abs(coef) for feature, coef in self.model_coefficients.items()}

    def validate_input(self, input_data: Dict) -> Tuple[bool, List[str]]:
        """Validate input data"""
        errors = []
        required_fields = ["annual_inc", "int_rate", "purpose"]

        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate ranges
        if "annual_inc" in input_data and input_data["annual_inc"] <= 0:
            errors.append("Annual income must be positive")

        if "int_rate" in input_data and (
            input_data["int_rate"] < 0 or input_data["int_rate"] > 50
        ):
            errors.append("Interest rate must be between 0 and 50")

        return len(errors) == 0, errors
