# app/utils/helpers.py
import json
import logging
import joblib
import pandas as pd
import numpy as np
import platform
from pathlib import Path
from typing import Optional
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns that are non-predictive or contain data leakage"""

    def __init__(self):
        self.drop_cols = [
            "id",
            "member_id",
            "loan_status",
            "url",
            "desc",
            "title",
            "zip_code",
            "policy_code",
            "next_pymnt_d",
            "last_pymnt_d",
            "last_credit_pull_d",
            "annual_inc_joint",
            "dti_joint",
            "verification_status_joint",
            "emp_title",
            "out_prncp",
            "out_prncp_inv",
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "collection_recovery_fee",
            "mths_since_last_major_derog",
            # Additional columns with >95% missing values
            "il_util",
            "mths_since_rcnt_il",
            "total_bal_il",
            "open_il_24m",
            "open_il_12m",
            "open_acc_6m",
            "open_rv_12m",
            "open_rv_24m",
            "open_il_6m",
            "all_util",
            "inq_fi",
            "total_cu_tl",
            "inq_last_12m",
            "max_bal_bc",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Only drop columns that exist in the dataframe
        cols_to_drop = [col for col in self.drop_cols if col in X.columns]
        X = X.drop(columns=cols_to_drop)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers new features from existing ones (production-safe)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # ✅ Use provided credit_history_length directly
        # (no need to compute from issue_d / earliest_cr_line)
        X = X.drop(columns=["issue_d", "earliest_cr_line"], errors="ignore")

        # ✅ Create loan burden ratio if needed
        if "loan_amnt" in X.columns and "annual_inc" in X.columns:
            X["loan_burden"] = X["loan_amnt"] / (X["annual_inc"] + 1)

        # ✅ Drop redundant columns if they exist
        X = X.drop(columns=["loan_amnt", "dti"], errors="ignore")

        return X


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using median for numerical and mode for categorical"""

    def __init__(self):
        # Ensure all columns from file_context_0/impute_ok are included
        self.impute_columns = [
            "total_rev_hi_lim",
            "tot_coll_amt",
            "tot_cur_bal",
            "emp_length",
            "revol_util",
            "collections_12_mths_ex_med",
            "acc_now_delinq",
            "total_acc",
            "pub_rec",
            "open_acc",
            "inq_last_6mths",
            "delinq_2yrs",
            "credit_history_length",
            "annual_inc",
            "mths_since_last_record",
            "mths_since_last_delinq",
            "loan_amnt",
            "loan_burden",  # ensure loan_amnt is imputed!
        ]
        self.median_values = {}
        self.mode_values = {}

    def fit(self, X, y=None):
        X = X.copy()
        # Identify numerical and categorical columns
        numerical_cols = [
            col
            for col in self.impute_columns
            if col != "emp_length" and col in X.columns
        ]
        categorical_cols = [
            col
            for col in self.impute_columns
            if col == "emp_length" and col in X.columns
        ]

        # Fit numerical imputers
        for col in numerical_cols:
            if X[col].dropna().shape[0] > 0:
                self.median_values[col] = X[col].median()
            else:
                self.median_values[col] = 0  # fallback if all values are NaN

        # Fit categorical imputer
        for col in categorical_cols:
            if X[col].dropna().shape[0] > 0:
                self.mode_values[col] = X[col].mode(dropna=True)[0]
            else:
                self.mode_values[col] = "Unknown"  # fallback if all values are NaN

        return self

    def transform(self, X):
        X = X.copy()
        # Impute numerical columns
        for col, median_val in self.median_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)
        # Impute categorical columns
        for col, mode_val in self.mode_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(mode_val)
        return X


class WOETransformer(BaseEstimator, TransformerMixin):
    """Applies Weight of Evidence (WOE) transformation to features"""

    def __init__(self, max_leaf_nodes=5):
        self.max_leaf_nodes = max_leaf_nodes
        self.features_to_bin = [
            "annual_inc",
            "int_rate",
            "credit_history_length",
            "total_rev_hi_lim",
            "open_acc",
            "revol_util",
            "tot_cur_bal",
            "mths_since_last_record",
            "mths_since_last_delinq",
            "loan_burden",
        ]
        self.categorical_features = [
            "term",
            "home_ownership",
            "purpose",
            "emp_length",
            "verification_status",
        ]
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
        self.bins = {}
        self.woe_mappings = {}

    def decision_tree_binning(self, X, y):
        """Create bins using decision tree"""
        X_reshaped = X.values.reshape(-1, 1)
        tree = DecisionTreeClassifier(
            criterion="entropy",
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=0.05,
        )
        tree.fit(X_reshaped, y)

        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds != -2]  # -2 = leaf node

        bins = [-np.inf] + sorted(thresholds.tolist()) + [np.inf]
        return bins

    def calc_woe_iv(self, df, feature_bin_col, target_col="target"):
        """Calculate WOE and IV for a feature"""
        grouped = df.groupby(feature_bin_col)[target_col].agg(["count", "sum"])
        grouped.columns = ["total", "bad"]
        grouped["good"] = grouped["total"] - grouped["bad"]

        dist_good = grouped["good"] / grouped["good"].sum()
        dist_bad = grouped["bad"] / grouped["bad"].sum()

        grouped["woe"] = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))
        grouped["iv"] = (dist_good - dist_bad) * grouped["woe"]

        woe_dict = grouped["woe"].to_dict()
        iv_score = grouped["iv"].sum()

        return woe_dict, iv_score

    def fit(self, X, y=None):
        X = X.copy()

        # If target is in X, extract it for WOE calculation
        if "target" not in X.columns:
            print("Warning: Target column not found in X")
            return self

        target_col = X["target"]

        # Bin numerical features
        for feature in self.features_to_bin:
            if feature in X.columns:
                try:
                    bins = self.decision_tree_binning(X[feature], target_col)
                    self.bins[feature] = bins
                    X[feature + "_bin"] = pd.cut(X[feature], bins=bins)
                except Exception as e:
                    print(f"Could not bin {feature}: {e}")

        # Calculate WOE for binned numerical features
        for feature in self.features_to_bin:
            bin_col = feature + "_bin"
            if bin_col in X.columns:
                try:
                    woe_map, iv_score = self.calc_woe_iv(X, bin_col, "target")
                    self.woe_mappings[feature] = woe_map
                    print(f"✅ WOE calculated for {feature}, IV = {iv_score:.4f}")
                except Exception as e:
                    print(f"❌ Failed WOE calculation for {feature}: {e}")

        # Calculate WOE for categorical features
        for cat_feature in self.categorical_features:
            if cat_feature in X.columns:
                try:
                    woe_map, iv_score = self.calc_woe_iv(X, cat_feature, "target")
                    self.woe_mappings[cat_feature] = woe_map
                    print(f"✅ WOE calculated for {cat_feature}, IV = {iv_score:.4f}")
                except Exception as e:
                    print(f"❌ Failed WOE calculation for {cat_feature}: {e}")

        return self

    def transform(self, X):
        X = X.copy()

        # Apply binning and WOE to numerical features
        for feature in self.features_to_bin:
            if feature in X.columns and feature in self.bins:
                X[feature + "_bin"] = pd.cut(X[feature], bins=self.bins[feature])
                if feature in self.woe_mappings:
                    X[feature + "_woe"] = X[feature + "_bin"].map(
                        self.woe_mappings[feature]
                    )

        # Apply WOE to categorical features
        for cat_feature in self.categorical_features:
            if cat_feature in X.columns and cat_feature in self.woe_mappings:
                X[cat_feature + "_woe"] = X[cat_feature].map(
                    self.woe_mappings[cat_feature]
                )

        # Return all columns including target if it exists
        return X


# Custom unpickler for pipelines
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        custom_classes = {
            "ColumnDropper": ColumnDropper,
            "FeatureEngineer": FeatureEngineer,
            "MissingValueImputer": MissingValueImputer,
            "WOETransformer": WOETransformer,
        }
        if name in custom_classes:
            return custom_classes[name]
        return super().find_class(module, name)


class CreditRiskService:
    """Service to handle preprocessing, prediction, and scoring"""

    def __init__(self, model_dir: str = "models"):
        self.model_path = Path(model_dir) / "credit_risk_model.joblib"
        self.pipeline_path = Path(model_dir) / "preprocessing_pipeline.joblib"
        self.metadata_path = Path(model_dir) / "model_metadata.json"
        self.scorecard_path = Path(model_dir) / "scorecard.csv"

        # --- custom class resolver for pipeline unpickling ---
        # custom_classes = {
        #     "ColumnDropper": ColumnDropper,
        #     "FeatureEngineer": FeatureEngineer,
        #     "MissingValueImputer": MissingValueImputer,
        #     "WOETransformer": WOETransformer,
        # }

        # --- load model ---
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

        # --- load pipeline with custom unpickler ---
        try:
            with open(self.pipeline_path, "rb") as f:
                self.pipeline = CustomUnpickler(f).load()
            logger.info(f"✅ Loaded preprocessing pipeline from {self.pipeline_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline: {e}")
            raise

        # --- load metadata ---
        try:
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"✅ Loaded metadata from {self.metadata_path}")
        except Exception as e:
            logger.warning(f"⚠️ Metadata not found: {e}")
            self.metadata = {}

        # --- load scorecard (optional) ---
        try:
            self.scorecard = pd.read_csv(self.scorecard_path)
            logger.info(f"✅ Loaded scorecard from {self.scorecard_path}")
        except Exception:
            self.scorecard = None
            logger.warning("⚠️ Scorecard not found, continuing without it")

        # --- prediction ---

    def predict(self, input_dict: dict) -> dict:
        """Generate credit risk prediction for a single applicant"""
        X = pd.DataFrame([input_dict])

        logger.info(
            f"Raw verification_status received: {input_dict.get('verification_status')}"
        )
        X["verification_status"] = X["verification_status"].str.strip().str.title()

        # --- preprocessing ---
        try:
            X_proc = self.pipeline.transform(X)
        except Exception as e:
            logger.error(f"❌ Pipeline transform failed: {e}")
            raise

        # --- align with training features ---
        trained_features = self.metadata.get("features", [])
        missing = [f for f in trained_features if f not in X_proc.columns]
        if missing:
            logger.error(f"❌ Missing features in processed data: {missing}")
            raise ValueError(f"Missing features: {missing}")

        X_proc = X_proc[trained_features]

        # Debug step
        if X_proc.isnull().any().any():
            logger.error("NaNs detected in processed features!")
            logger.error(f"NaN columns: {list(X_proc.columns[X_proc.isnull().any()])}")
            logger.error(f"Row values: {X_proc.to_dict(orient='records')}")
            raise ValueError("Processed data contains NaNs")

        # --- prediction ---
        try:
            prob = self.model.predict_proba(X_proc)[:, 1][0]
        except Exception as e:
            logger.error(f"❌ Model prediction failed: {e}")
            raise

        # --- score scaling ---
        pdo = self.metadata.get("PDO", 20)
        base_score = self.metadata.get("BaseScore", 600)
        base_odds = self.metadata.get("BaseOdds", 50)

        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        log_odds = np.log(prob / (1 - prob + 1e-9))
        score = offset - factor * log_odds

        # --- risk level from credit score (standard credit rating scale) ---
        # Map credit score to standard S&P/Moody's/Fitch rating scale
        if score >= 750:
            risk_level = "AAA"
        elif score >= 700:
            risk_level = "AA"
        elif score >= 650:
            risk_level = "A"
        elif score >= 600:
            risk_level = "BBB"
        elif score >= 550:
            risk_level = "BB"
        elif score >= 500:
            risk_level = "B"
        elif score >= 450:
            risk_level = "CCC"
        elif score >= 400:
            risk_level = "CC"
        elif score >= 350:
            risk_level = "C"
        else:
            risk_level = "D"

        return {
            "credit_score": round(score, 2),
            "default_probability": round(prob, 4),
            "risk_level": risk_level,
            "log_odds": round(log_odds, 4),
        }


# Global instance
_model_instance: Optional[CreditRiskService] = None


def get_model_instance() -> CreditRiskService:
    global _model_instance
    if _model_instance is None:
        _model_instance = CreditRiskService()
    return _model_instance


def setup_logging():
    """Configure logging format and level"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("uvicorn.error").propagate = False
    logging.getLogger("uvicorn.access").propagate = False
    return logging.getLogger(__name__)


def validate_environment() -> bool:
    """Basic environment validation"""
    try:
        python_version = platform.python_version()
        logger = logging.getLogger(__name__)
        logger.info(f"✅ Running on Python {python_version}")
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Environment validation failed: {e}")
        return False


def get_api_metadata():
    """Return API metadata"""
    return {
        "service": "Credit Risk Scorecard API",
        "version": "1.0.0",
        "description": (
            "Predict loan default probability and credit score using "
            "logistic regression + WOE pipeline"
        ),
        "endpoints": [
            "/api/v1/health",
            "/api/v1/predict",
            "/api/v1/predict/batch",
            "/api/v1/model/info",
        ],
    }
