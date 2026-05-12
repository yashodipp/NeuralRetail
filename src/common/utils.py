"""General-purpose utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def to_iso_date(series: pd.Series) -> pd.Series:
    """Normalize a pandas date series to ISO strings."""

    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def chunked(items: Iterable, size: int):
    """Yield successive chunks from any iterable."""

    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
