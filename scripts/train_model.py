#!/usr/bin/env python3
"""
Training script for Credit Risk Scorecard Model

This script trains the logistic regression model with WOE transformations
and saves the trained model for deployment.
"""

import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from app.models.preprocessing import DataPreprocessor
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent))


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load and return the loan data"""
    logger.info(f"Loading data from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    return df


def train_model(df: pd.DataFrame) -> dict:
    """Train the credit risk model"""
    logger.info("Starting model training")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Clean the data
    logger.info("Cleaning data...")
    df_clean = preprocessor.clean_data(df)
    logger.info(f"After cleaning: {len(df_clean)} rows")

    # Define features to bin
    features_to_bin = [
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

    # Create bins for numerical features
    logger.info("Creating bins for numerical features...")
    bin_thresholds = {}
    for feature in features_to_bin:
        if feature in df_clean.columns:
            try:
                bins = preprocessor.decision_tree_binning(
                    df_clean[feature], df_clean["target"], max_leaf_nodes=5
                )
                bin_thresholds[feature] = bins
                df_clean[feature + "_bin"] = pd.cut(df_clean[feature], bins=bins)
                logger.info(f"Binned {feature} into {len(bins) - 1} bins")
            except Exception as e:
                logger.warning(f"Could not bin {feature} : {e}")

    # Define categorical features
    categorical_features = [
        "term",
        "home_ownership",
        "purpose",
        "emp_length",
        "verification_status",
    ]

    # Calculate WOE and IV for all features
    logger.info("Calculating WOE and IV...")
    woe_mappings = {}
    iv_scores = {}

    # WOE for binned numerical features
    for feature in features_to_bin:
        bin_col = feature + "_bin"
        if bin_col in df_clean.columns:
            try:
                woe_map, iv_score = preprocessor.calculate_woe_iv(
                    df_clean, bin_col, "target"
                )
                woe_mappings[feature] = woe_map
                iv_scores[feature] = iv_score
                df_clean[feature + "_woe"] = df_clean[bin_col].map(woe_map)
                logger.info(f"WOE applied to {feature}, IV = {iv_score:.4f}")
            except Exception as e:
                logger.warning(f"Failed WOE on {feature}: {e}")

    # WOE for categorical features
    for cat_feature in categorical_features:
        if cat_feature in df_clean.columns:
            try:
                woe_map, iv_score = preprocessor.calculate_woe_iv(
                    df_clean, cat_feature, "target"
                )
                woe_mappings[cat_feature] = woe_map
                iv_scores[cat_feature] = iv_score
                df_clean[cat_feature + "_woe"] = df_clean[cat_feature].map(woe_map)
                logger.info(f"WOE applied to {cat_feature}, IV = {iv_score:.4f}")
            except Exception as e:
                logger.warning(f"Failed WOE on {cat_feature}: {e}")

    # Select final features based on IV
    final_features = []
    for feature, iv in iv_scores.items():
        if iv > 0.02:  # IV threshold
            final_features.append(feature + "_woe")
            logger.info(f"Selected {feature} (IV = {iv:.4f})")

    logger.info(f"Selected {len(final_features)} features for modeling")

    # Prepare training data
    X = df_clean[final_features].fillna(0)
    y = df_clean["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train logistic regression model
    logger.info("Training logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Model AUC: {auc_score:.4f}")

    # Print classification report
    y_pred = model.predict(X_test)
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Prepare model data for saving
    model_data = {
        "model": model,
        "feature_names": final_features,
        "woe_mappings": woe_mappings,
        "bin_thresholds": bin_thresholds,
        "iv_scores": iv_scores,
        "auc_score": auc_score,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return model_data


def save_model(model_data: dict, output_path: str):
    """Save the trained model"""
    logger.info(f"Saving model to {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save model
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info("Model saved successfully")


def main():
    """Main training function"""
    # Configuration
    data_path = "data/loan/loan.csv"
    model_path = "models/trained_model.pkl"

    try:
        # Load data
        df = load_data(data_path)

        # Train model
        model_data = train_model(df)

        # Save model
        save_model(model_data, model_path)

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Model AUC: {model_data['auc_score']:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
