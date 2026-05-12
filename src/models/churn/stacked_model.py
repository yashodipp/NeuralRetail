"""Stacked XGBoost + LightGBM churn predictor with SHAP explainability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.common.logging import get_logger

logger = get_logger(__name__)

try:
    import shap
except ImportError:  # pragma: no cover - optional during static validation
    shap = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass(slots=True)
class ChurnPredictionBundle:
    """Prediction output with recommendations and explanations."""

    scores: pd.DataFrame
    auc: float


class ChurnStackedModel:
    """Stacked learner optimized for churn probability scoring."""

    def __init__(self) -> None:
        self.model: StackingClassifier | None = None
        self.feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame, label_col: str = "churn_label") -> float:
        """Train the stacked classifier and return validation AUC."""

        features = frame.drop(columns=[label_col, "customer_id"], errors="ignore")
        labels = frame[label_col].astype(int)
        self.feature_columns = list(features.columns)

        estimators = []
        if XGBClassifier:
            estimators.append(
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                    ),
                )
            )
        if LGBMClassifier:
            estimators.append(
                (
                    "lgbm",
                    LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                    ),
                )
            )
        if not estimators:
            estimators = [("gb", GradientBoostingClassifier(random_state=42))]

        x_train, x_valid, y_train, y_valid = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            passthrough=True,
        )
        self.model.fit(x_train, y_train)
        auc = float(roc_auc_score(y_valid, self.model.predict_proba(x_valid)[:, 1]))
        logger.info("Churn validation AUC=%.4f", auc)
        return auc

    def predict(self, frame: pd.DataFrame) -> ChurnPredictionBundle:
        """Predict churn probabilities with retention actions."""

        if not self.model:
            raise ValueError("Model must be fitted before prediction")
        features = frame[self.feature_columns]
        probabilities = self.model.predict_proba(features)[:, 1]
        scores = pd.DataFrame(
            {
                "customer_id": frame.get("customer_id", pd.Series(np.arange(len(frame)))),
                "churn_probability": probabilities,
                "retention_action": [self._recommend_action(prob) for prob in probabilities],
            }
        )
        auc = float(roc_auc_score(frame["churn_label"].astype(int), probabilities)) if "churn_label" in frame else float("nan")
        return ChurnPredictionBundle(scores=scores, auc=auc)

    def explain(self, frame: pd.DataFrame, sample_size: int = 50) -> dict[str, Any]:
        """Return SHAP explanations for a sample of customers."""

        if not self.model:
            raise ValueError("Model must be fitted before explanation")
        sample = frame[self.feature_columns].head(sample_size)
        if not shap:
            return {"available": False, "message": "shap is not installed"}
        try:
            explainer = shap.Explainer(self.model.predict_proba, sample)
            values = explainer(sample)
            return {
                "available": True,
                "base_value": np.asarray(values.base_values).tolist(),
                "feature_names": list(sample.columns),
                "mean_abs_shap": np.abs(values.values).mean(axis=0).tolist(),
            }
        except Exception as exc:  # pragma: no cover - explainer dependent
            logger.warning("SHAP explanation generation failed: %s", exc)
            return {"available": False, "message": str(exc)}

    @staticmethod
    def _recommend_action(probability: float) -> str:
        if probability >= 0.8:
            return "Offer high-touch retention call with premium discount"
        if probability >= 0.6:
            return "Trigger personalized incentive campaign"
        if probability >= 0.4:
            return "Send loyalty bundle recommendation"
        return "Maintain standard nurture sequence"
