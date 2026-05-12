"""Spark + Delta Lake ingestion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

try:
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import DataFrame, SparkSession
except ImportError:  # pragma: no cover - optional for local static validation
    SparkSession = None  # type: ignore[assignment]
    DataFrame = Any  # type: ignore[assignment]
    configure_spark_with_delta_pip = None


def build_spark_session() -> SparkSession:
    """Create a Spark session configured for Delta Lake."""

    if SparkSession is None:
        raise RuntimeError("pyspark and delta-spark are required for Spark ingestion")
    settings = get_settings()
    builder = (
        SparkSession.builder.appName(settings.spark_app_name)
        .master(settings.spark_master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    if configure_spark_with_delta_pip:
        return configure_spark_with_delta_pip(builder).getOrCreate()
    return builder.getOrCreate()


class SparkIngestionService:
    """Service for ingesting multiple retail data sources."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        if spark is not None:
            self.spark = spark
        else:
            try:
                self.spark = build_spark_session()
            except Exception as exc:  # pragma: no cover - infra dependent
                logger.warning("Spark unavailable, falling back to pandas/parquet ingestion: %s", exc)
                self.spark = None

    def read_csv(self, path: str, **options: Any) -> DataFrame:
        if self.spark is None:
            return pd.read_csv(path, **options)
        return self.spark.read.options(header=True, inferSchema=True, **options).csv(path)

    def read_parquet(self, path: str, **options: Any) -> DataFrame:
        if self.spark is None:
            return pd.read_parquet(path, **options)
        return self.spark.read.options(**options).parquet(path)

    def read_api(self, endpoint: str, params: dict[str, Any] | None = None) -> DataFrame:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        payload = pd.DataFrame(response.json())
        if self.spark is None:
            return payload
        return self.spark.createDataFrame(payload)

    def read_kafka(self, topic: str, bootstrap_servers: str | None = None) -> DataFrame:
        if self.spark is None:
            raise RuntimeError("Kafka ingestion requires an active Spark session")
        settings = get_settings()
        return (
            self.spark.read.format("kafka")
            .option("subscribe", topic)
            .option("kafka.bootstrap.servers", bootstrap_servers or settings.kafka_bootstrap_servers)
            .load()
        )

    def write_delta(self, frame: DataFrame, destination: str, mode: str = "overwrite", partition_by: list[str] | None = None) -> None:
        if self.spark is None:
            destination_path = Path(destination)
            destination_path.mkdir(parents=True, exist_ok=True)
            output_file = destination_path / "part-0000.parquet"
            if isinstance(frame, pd.DataFrame):
                frame.to_parquet(output_file, index=False)
            else:
                pd.DataFrame(frame).to_parquet(output_file, index=False)
            logger.info("Parquet fallback write completed: %s", output_file)
            return
        writer = frame.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(destination)
        logger.info("Delta write completed: %s", destination)
