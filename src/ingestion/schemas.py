"""Schemas for multi-source ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class IngestionSource:
    """Represents a source feeding the data lake."""

    name: str
    source_type: Literal["csv", "parquet", "api", "kafka"]
    path: str | Path | None = None
    endpoint: str | None = None
    topic: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionJob:
    """Metadata for one logical ingestion run."""

    run_id: str
    bronze_table: str
    silver_table: str
    partition_by: list[str] = field(default_factory=lambda: ["event_date"])
