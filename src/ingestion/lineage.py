"""OpenLineage event helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LineageEvent:
    """Lightweight lineage event for observability."""

    job_name: str
    run_id: str
    inputs: list[str]
    outputs: list[str]
    event_time: str


class OpenLineagePublisher:
    """Publisher wrapper that can later be replaced with the native client."""

    def emit(self, job_name: str, run_id: str, inputs: list[str], outputs: list[str]) -> dict[str, Any]:
        """Create and log a lineage event."""

        event = LineageEvent(
            job_name=job_name,
            run_id=run_id,
            inputs=inputs,
            outputs=outputs,
            event_time=datetime.now(UTC).isoformat(),
        )
        payload = asdict(event)
        logger.info("OpenLineage event: %s", payload)
        return payload
