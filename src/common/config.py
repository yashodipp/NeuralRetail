"""Application settings and environment configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "NeuralRetail"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    debug: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501

    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    api_keys: list[str] = Field(default_factory=lambda: ["neuralretail-local-key"])
    service_username: str = "admin"
    service_password: str = "admin123"

    database_url: str = "postgresql+psycopg://postgres:postgres@postgres:5432/neuralretail"
    redis_url: str = "redis://redis:6379/0"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    feast_registry_path: str = str(ROOT_DIR / "data" / "feature_repo")
    s3_endpoint_url: str = "http://minio:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123"
    s3_bucket: str = "neuralretail"

    data_dir: Path = ROOT_DIR / "data"
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    artifacts_dir: Path = ROOT_DIR / "artifacts"

    prometheus_enabled: bool = True
    retrain_psi_threshold: float = 0.2
    retrain_mape_threshold: float = 15.0

    spark_app_name: str = "NeuralRetailIngestion"
    spark_master: str = "local[*]"
    kafka_bootstrap_servers: str = "localhost:9092"

    @property
    def access_token_ttl_seconds(self) -> int:
        return self.access_token_expire_minutes * 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""

    return Settings()
