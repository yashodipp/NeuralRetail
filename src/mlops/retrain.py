"""Auto-retrain policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from src.common.config import get_settings


@dataclass(slots=True)
class RetrainDecision:
    """Decision payload for automated retraining."""

    should_retrain: bool
    reasons: list[str]


def evaluate_retrain_policy(psi: float, mape: float) -> RetrainDecision:
    """Apply project thresholds for retraining."""

    settings = get_settings()
    reasons = []
    if psi > settings.retrain_psi_threshold:
        reasons.append(f"PSI exceeded threshold ({psi:.3f} > {settings.retrain_psi_threshold:.3f})")
    if mape > settings.retrain_mape_threshold:
        reasons.append(f"MAPE exceeded threshold ({mape:.2f} > {settings.retrain_mape_threshold:.2f})")
    return RetrainDecision(should_retrain=bool(reasons), reasons=reasons)
