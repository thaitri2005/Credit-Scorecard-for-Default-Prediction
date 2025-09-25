import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing utilities for credit risk modeling
    """

    def __init__(self):
        self.bin_thresholds = {}
        self.woe_mappings = {}

    def decision_tree_binning(
        self, X: pd.Series, y: pd.Series, max_leaf_nodes: int = 5
    ) -> List[float]:
        """
        Create bins using decision tree binning

        Args:
            X: Feature values
            y: Target values
            max_leaf_nodes: Maximum number of leaf nodes

        Returns:
            List of bin thresholds
        """
        try:
            X_reshaped = X.values.reshape(-1, 1)
            tree = DecisionTreeClassifier(
                criterion="entropy",
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=0.05,
            )
            tree.fit(X_reshaped, y)

            thresholds = tree.tree_.threshold
            thresholds = thresholds[thresholds != -2]  # -2 = leaf node

            bins = [-np.inf] + sorted(thresholds.tolist()) + [np.inf]
            return bins
        except Exception as e:
            logger.error(f"Error in decision tree binning: {e}")
            # Fallback to quantile binning
            return self.quantile_binning(X, n_bins=max_leaf_nodes)

    def quantile_binning(self, X: pd.Series, n_bins: int = 5) -> List[float]:
        """
        Create bins using quantile binning

        Args:
            X: Feature values
            n_bins: Number of bins

        Returns:
            List of bin thresholds
        """
        try:
            quantiles = np.linspace(0, 1, n_bins + 1)
            thresholds = X.quantile(quantiles).tolist()
            thresholds[0] = -np.inf
            thresholds[-1] = np.inf
            return thresholds
        except Exception as e:
            logger.error(f"Error in quantile binning: {e}")
            return [-np.inf, np.inf]

    def calculate_woe_iv(
        self, df: pd.DataFrame, feature_col: str, target_col: str = "target"
    ) -> Tuple[Dict, float]:
        """
        Calculate Weight of Evidence (WOE) and Information Value (IV)

        Args:
            df: DataFrame with feature and target columns
            feature_col: Name of the feature column
            target_col: Name of the target column

        Returns:
            Tuple of (woe_mapping, iv_score)
        """
        try:
            grouped = df.groupby(feature_col)[target_col].agg(["count", "sum"])
            grouped.columns = ["total", "bad"]
            grouped["good"] = grouped["total"] - grouped["bad"]

            # Calculate distributions
            dist_good = grouped["good"] / grouped["good"].sum()
            dist_bad = grouped["bad"] / grouped["bad"].sum()

            # Calculate WOE
            grouped["woe"] = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))

            # Calculate IV
            grouped["iv"] = (dist_good - dist_bad) * grouped["woe"]

            woe_dict = grouped["woe"].to_dict()
            iv_score = grouped["iv"].sum()

            return woe_dict, iv_score
        except Exception as e:
            logger.error(f"Error calculating WOE/IV: {e}")
            return {}, 0.0

    def map_loan_status(self, status: str) -> int:
        """
        Map loan status to binary target

        Args:
            status: Loan status string

        Returns:
            0 for good, 1 for bad, -1 for ambiguous
        """
        bad_statuses = [
            "Charged Off",
            "Default",
            "Late (31-120 days)",
            "Late (16-30 days)",
            "Does not meet the credit policy. Status:Charged Off",
        ]

        good_statuses = [
            "Fully Paid",
            "Current",
            "Does not meet the credit policy. Status:Fully Paid",
        ]

        if status in bad_statuses:
            return 1  # BAD borrower
        elif status in good_statuses:
            return 0  # GOOD borrower
        else:
            return -1  # Exclude ambiguous cases by returning -1

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw loan data

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Map loan status to target
        df_clean["target"] = df_clean["loan_status"].apply(self.map_loan_status)

        # Remove rows without clear target
        df_clean = df_clean.dropna(subset=["target"])
        df_clean["target"] = df_clean["target"].astype(int)

        # Drop columns that are not useful for modeling
        drop_cols = [
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
        ]

        # Only drop columns that exist
        existing_drop_cols = [col for col in drop_cols if col in df_clean.columns]
        df_clean = df_clean.drop(columns=existing_drop_cols)

        # Drop heavily missing features
        drop_heavy_missing = [
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

        existing_heavy_missing = [
            col for col in drop_heavy_missing if col in df_clean.columns
        ]
        df_clean = df_clean.drop(columns=existing_heavy_missing)

        # Handle date columns
        if "issue_d" in df_clean.columns and "earliest_cr_line" in df_clean.columns:
            df_clean["issue_d"] = pd.to_datetime(
                df_clean["issue_d"], format="%b-%Y", errors="coerce"
            )
            df_clean["earliest_cr_line"] = pd.to_datetime(
                df_clean["earliest_cr_line"], format="%b-%Y", errors="coerce"
            )

            # Create credit history length
            df_clean["credit_history_length"] = (
                df_clean["issue_d"] - df_clean["earliest_cr_line"]
            ).dt.days / 365

            # Drop date columns
            df_clean = df_clean.drop(columns=["issue_d", "earliest_cr_line"])

        # Impute missing values
        self._impute_missing_values(df_clean)

        # Create loan burden feature
        if "loan_amnt" in df_clean.columns and "annual_inc" in df_clean.columns:
            df_clean["loan_burden"] = df_clean["loan_amnt"] / (
                df_clean["annual_inc"] + 1
            )

            # Drop original columns
            df_clean = df_clean.drop(columns=["loan_amnt", "dti"], errors="ignore")

        return df_clean

    def _impute_missing_values(self, df: pd.DataFrame):
        """Impute missing values in the dataset"""
        # Numerical columns to impute
        numerical_cols = [
            "total_rev_hi_lim",
            "tot_coll_amt",
            "tot_cur_bal",
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
        ]

        for col in numerical_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Categorical columns
        if "emp_length" in df.columns:
            mode_val = (
                df["emp_length"].mode()[0]
                if not df["emp_length"].mode().empty
                else "Unknown"
            )
            df["emp_length"] = df["emp_length"].fillna(mode_val)

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for features"""
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "target_distribution": df["target"].value_counts().to_dict()
            if "target" in df.columns
            else {},
        }
        return summary
