"""Helpers to materialize engineered features into Feast-compatible files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.utils import ensure_directory


def export_for_feast(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Persist a feature table for Feast offline ingestion."""

    destination = Path(destination)
    ensure_directory(destination.parent)
    frame.to_parquet(destination, index=False)
    return destination
