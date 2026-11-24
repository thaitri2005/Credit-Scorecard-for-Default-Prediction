"""Loads the trained model and turns loan applications into credit scores."""
import hashlib
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from app.scoring.transformers import (
    ColumnDropper,
    FeatureEngineer,
    MissingValueImputer,
    WOETransformer,
)

logger = logging.getLogger(__name__)

# Resolved from this file so the service works from any working directory.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


class CustomUnpickler(pickle.Unpickler):
    """Resolves the pipeline's custom transformers regardless of the module
    they were pickled from (the notebook pickled them from ``__main__``)."""

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


# Scorecard scaling used when model_metadata.json does not override it.
DEFAULT_SCORING_PARAMS: Dict[str, float] = {"PDO": 20, "BaseScore": 600, "BaseOdds": 50}

# Internal risk grades, cut on the score range this model can actually produce
# (roughly 511-646). Used when model_metadata.json has no "rating_bands".
DEFAULT_RATING_BANDS: List[Dict[str, Any]] = [
    {"min_score": 620, "rating": "AAA", "description": "Minimal default risk"},
    {"min_score": 605, "rating": "AA", "description": "Very low default risk"},
    {"min_score": 590, "rating": "A", "description": "Low default risk"},
    {"min_score": 575, "rating": "BBB", "description": "Below-average default risk"},
    {"min_score": 560, "rating": "BB", "description": "Average default risk"},
    {"min_score": 545, "rating": "B", "description": "Above-average default risk"},
    {"min_score": 530, "rating": "CCC", "description": "High default risk"},
    {"min_score": 515, "rating": "CC", "description": "Very high default risk"},
]
# "D" is not used: it denotes an actual default, which a score cannot observe.
LOWEST_RATING: Dict[str, str] = {
    "rating": "C",
    "description": "Highest default risk this model assigns",
}

# Inputs that may be missing. A missing value is scored like the least
# favorable real bin of that feature, so leaving a field blank can never beat
# disclosing it.
OPTIONAL_NUMERIC_FEATURES = (
    "total_rev_hi_lim",
    "tot_cur_bal",
    "revol_util",
    "loan_burden",
)

# A bin narrower than this fraction of the imputation median that also contains
# the median is an artefact of "impute, then bin" at training time: it holds the
# rows whose value was missing, not borrowers who really had that value.
_IMPUTATION_BIN_REL_WIDTH = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CreditRiskService:
    """Service to handle preprocessing, prediction, and scoring"""

    def __init__(self, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR):
        self.model_path = Path(model_dir) / "credit_risk_model.joblib"
        self.pipeline_path = Path(model_dir) / "preprocessing_pipeline.joblib"
        self.metadata_path = Path(model_dir) / "model_metadata.json"

        try:
            with open(self.metadata_path, "r") as f:
                self.metadata: Dict[str, Any] = json.load(f)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Cannot read {self.metadata_path}: {e}") from e
        if not isinstance(self.metadata, dict):
            raise RuntimeError(f"{self.metadata_path} must contain a JSON object")

        # Pickles execute code on load, so only load the artifacts we shipped.
        self._verify_checksums()

        self.model = joblib.load(self.model_path)
        with open(self.pipeline_path, "rb") as f:
            self.pipeline = CustomUnpickler(f).load()
        logger.info(f"Loaded model and preprocessing pipeline from {model_dir}/")

        self.features: List[str] = self.metadata.get("features") or list(
            WOETransformer().final_woe_features
        )
        self.scoring_params, self.rating_bands, self.lowest_rating = self._scoring()
        self.factor = self.scoring_params["PDO"] / np.log(2)
        self.offset = self.scoring_params["BaseScore"] - self.factor * np.log(
            self.scoring_params["BaseOdds"]
        )
        self._imputation_bins = self._find_imputation_bins()
        self._missing_woe = self._find_missing_woe()

    def _verify_checksums(self) -> None:
        """Fail closed: every pickle must have a recorded, matching SHA-256."""
        expected = self.metadata.get("artifact_sha256")
        if not isinstance(expected, dict):
            raise RuntimeError("model_metadata.json has no artifact_sha256 section")
        for path in (self.model_path, self.pipeline_path):
            want = expected.get(path.name)
            if not isinstance(want, str):
                raise RuntimeError(f"No checksum recorded for {path.name}")
            if _sha256(path) != want.lower():
                raise RuntimeError(f"Checksum mismatch for model artifact {path.name}")

    def _scoring(self) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, str]]:
        """Read and validate scaling parameters and grade bands from metadata."""
        try:
            params = {
                key: float(self.metadata.get(key, default))
                for key, default in DEFAULT_SCORING_PARAMS.items()
            }
            bands: List[Dict[str, Any]] = [
                {
                    "min_score": float(band["min_score"]),
                    "rating": str(band["rating"]),
                    "description": str(band["description"]),
                }
                for band in self.metadata.get("rating_bands") or DEFAULT_RATING_BANDS
            ]
            lowest_raw = self.metadata.get("lowest_rating") or LOWEST_RATING
            lowest = {
                "rating": str(lowest_raw["rating"]),
                "description": str(lowest_raw["description"]),
            }
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Invalid scoring configuration in metadata: {e!r}"
            ) from e
        if params["PDO"] <= 0 or params["BaseOdds"] <= 0:
            raise RuntimeError("PDO and BaseOdds in metadata must be positive")
        bands.sort(key=lambda band: band["min_score"], reverse=True)
        return params, bands, lowest

    def _find_imputation_bins(self) -> Dict[str, Tuple[float, float, float]]:
        """Locate bins that only exist because missing values were median-imputed
        before binning. Returns {feature: (low, high, woe_of_bin_below)}."""
        steps = dict(self.pipeline.steps)
        medians = steps["imputer"].median_values
        woe = steps["woe_transformer"]
        found: Dict[str, Tuple[float, float, float]] = {}
        for feature in OPTIONAL_NUMERIC_FEATURES:
            edges = woe.bins.get(feature)
            median = medians.get(feature)
            if not edges or median is None:
                continue
            for i in range(1, len(edges) - 1):
                low, high = edges[i], edges[i + 1]
                is_sliver = (high - low) < _IMPUTATION_BIN_REL_WIDTH * abs(median)
                if low < median <= high and is_sliver:
                    # WOE map keys are pd.cut intervals with rounded edges.
                    below = [
                        w
                        for interval, w in woe.woe_mappings[feature].items()
                        if np.isclose(interval.right, low, rtol=1e-3)
                    ]
                    if below:
                        found[feature] = (low, high, float(below[0]))
        if found:
            logger.info(f"Imputation-artefact bins neutralised: {sorted(found)}")
        return found

    def _find_missing_woe(self) -> Dict[str, float]:
        """WOE used for a missing optional input: the least favorable bin that
        real borrowers fall into (the imputation-artefact bin is excluded)."""
        woe = dict(self.pipeline.steps)["woe_transformer"]
        result: Dict[str, float] = {}
        for feature in OPTIONAL_NUMERIC_FEATURES:
            artefact_low = self._imputation_bins.get(feature, (None,))[0]
            real = [
                float(w)
                for interval, w in woe.woe_mappings.get(feature, {}).items()
                if artefact_low is None
                or not np.isclose(interval.left, artefact_low, rtol=1e-3)
            ]
            result[feature] = min(real) if real else 0.0
        return result

    def rate(self, score: float) -> Dict[str, str]:
        """Map a credit score to its risk grade and description."""
        for band in self.rating_bands:
            if score >= band["min_score"]:
                return {"rating": band["rating"], "description": band["description"]}
        return dict(self.lowest_rating)

    def predict(self, input_dict: dict) -> dict:
        """Generate credit risk prediction for a single applicant"""
        return self.predict_many([input_dict])[0]

    def predict_many(self, inputs: List[dict]) -> List[dict]:
        """Score a list of applicants in one vectorised pass."""
        if not inputs:
            return []
        X = pd.DataFrame(inputs)

        for col in OPTIONAL_NUMERIC_FEATURES:
            if col not in X.columns:
                X[col] = np.nan
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        # The pipeline's FeatureEngineer expects the raw LendingClub column
        # "loan_amnt"; the API exposes "loan_amount". Derive the ratio here so
        # an explicitly supplied loan_burden always wins.
        if "loan_amount" in X.columns:
            amount = pd.to_numeric(X.pop("loan_amount"), errors="coerce").astype(float)
            X["loan_burden"] = X["loan_burden"].fillna(amount / (X["annual_inc"] + 1))

        missing = X[list(OPTIONAL_NUMERIC_FEATURES)].isna()
        raw = X[list(OPTIONAL_NUMERIC_FEATURES)].copy()

        X_proc = self.pipeline.transform(X)

        absent = [f for f in self.features if f not in X_proc.columns]
        if absent:
            raise ValueError(f"Missing features after preprocessing: {absent}")
        woe = X_proc[self.features].astype(float)

        for col in OPTIONAL_NUMERIC_FEATURES:
            woe_col = f"{col}_woe"
            if woe_col not in woe.columns:
                continue
            if col in self._imputation_bins:
                low, high, neighbour_woe = self._imputation_bins[col]
                in_sliver = (raw[col] > low) & (raw[col] <= high)
                woe.loc[in_sliver.to_numpy(), woe_col] = neighbour_woe
            woe.loc[missing[col].to_numpy(), woe_col] = self._missing_woe[col]

        if woe.isnull().any().any():
            bad_cols = list(woe.columns[woe.isnull().any()])
            raise ValueError(f"Unmapped values for features: {bad_cols}")

        probs = self.model.predict_proba(woe)[:, 1]
        # Clamp so the log-odds stay finite at the extremes.
        clamped = np.clip(probs, 1e-6, 1 - 1e-6)
        log_odds = np.log(clamped / (1 - clamped))
        scores = self.offset - self.factor * log_odds

        # Scorecard view of the same number: base points plus points per feature.
        base_points = self.offset - self.factor * float(self.model.intercept_[0])
        points = -self.factor * woe.to_numpy() * self.model.coef_[0]

        results = []
        for i in range(len(inputs)):
            score = round(float(scores[i]), 2)
            grade = self.rate(score)
            results.append(
                {
                    "credit_score": score,
                    "default_probability": round(float(probs[i]), 4),
                    "risk_level": grade["rating"],
                    "risk_description": grade["description"],
                    "log_odds": round(float(log_odds[i]), 4),
                    "base_points": round(base_points, 2),
                    "points": [
                        {
                            "feature": feature[: -len("_woe")],
                            "points": round(float(points[i, j]), 2),
                        }
                        for j, feature in enumerate(self.features)
                    ],
                    "missing_fields": [
                        col
                        for col in OPTIONAL_NUMERIC_FEATURES
                        if bool(missing[col].iloc[i])
                    ],
                }
            )
        return results


# Process-wide instance. The artifacts are baked into the image, so a failed
# load cannot fix itself: remember the failure instead of re-reading the files
# (and logging a traceback) on every request.
_model_instance: Optional[CreditRiskService] = None
_model_error: Optional[Exception] = None
_model_lock = threading.Lock()


def get_model_instance() -> CreditRiskService:
    global _model_instance, _model_error
    if _model_instance is None:
        with _model_lock:
            if _model_error is not None:
                raise _model_error
            if _model_instance is None:
                try:
                    _model_instance = CreditRiskService()
                except Exception as e:
                    _model_error = e
                    raise
    return _model_instance
