"""Retail ingestion orchestration pipeline."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger
from src.ingestion.lineage import OpenLineagePublisher
from src.ingestion.schemas import IngestionJob, IngestionSource
from src.ingestion.spark_ingestion import SparkIngestionService
from src.ingestion.validation import RetailDataValidator

logger = get_logger(__name__)


class IngestionPipeline:
    """Coordinate ingestion, validation, and lineage emission."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.validator = RetailDataValidator()
        self.lineage = OpenLineagePublisher()
        self.spark_service = SparkIngestionService()

    def ingest_local_transactions(self, source_path: str | Path | None = None) -> dict:
        """Ingest the bundled demo transactions CSV into Bronze and Silver Delta paths."""

        source_path = Path(source_path or self.settings.raw_data_dir / "transactions_sample.csv")
        job = IngestionJob(
            run_id=str(uuid4()),
            bronze_table=str(self.settings.processed_data_dir / "bronze" / "transactions"),
            silver_table=str(self.settings.processed_data_dir / "silver" / "transactions"),
            partition_by=["date"],
        )
        logger.info("Starting ingestion job %s", job.run_id)

        frame = pd.read_csv(source_path)
        validation = self.validator.validate(frame)
        if not validation["success"]:
            raise ValueError("Validation failed for transactions dataset")

        spark_frame = self.spark_service.read_csv(str(source_path))
        self.spark_service.write_delta(spark_frame, job.bronze_table, mode="overwrite", partition_by=job.partition_by)
        self.spark_service.write_delta(spark_frame, job.silver_table, mode="overwrite", partition_by=job.partition_by)

        lineage = self.lineage.emit(
            job_name="transactions_ingestion",
            run_id=job.run_id,
            inputs=[str(source_path)],
            outputs=[job.bronze_table, job.silver_table],
        )
        return {"run_id": job.run_id, "validation": validation, "lineage": lineage}

    def ingest_sources(self, sources: list[IngestionSource]) -> list[dict]:
        """Dispatch ingestion by source type for extensible pipelines."""

        results = []
        for source in sources:
            logger.info("Dispatching source %s (%s)", source.name, source.source_type)
            results.append({"source": source.name, "type": source.source_type, "status": "registered"})
        return results
