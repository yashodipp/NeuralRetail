"""Drift detection and PSI utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.utils import ensure_directory

logger = get_logger(__name__)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:  # pragma: no cover - optional during static validation
    Report = None
    DataDriftPreset = None


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate population stability index for one feature."""

    reference = pd.Series(reference).astype(float)
    current = pd.Series(current).astype(float)
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(reference.quantile(quantiles).to_numpy())
    if len(breakpoints) <= 2:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=breakpoints)
    cur_hist, _ = np.histogram(current, bins=breakpoints)
    ref_pct = np.where(ref_hist == 0, 1e-6, ref_hist / ref_hist.sum())
    cur_pct = np.where(cur_hist == 0, 1e-6, cur_hist / cur_hist.sum())
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def generate_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, destination: str | Path) -> Path | None:
    """Write an Evidently HTML report if the dependency is available."""

    if not (Report and DataDriftPreset):
        logger.warning("evidently is not installed; skipping HTML drift report")
        return None
    destination = Path(destination)
    ensure_directory(destination.parent)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(destination))
    logger.info("Evidently report written to %s", destination)
    return destination
