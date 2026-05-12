"""Model persistence and MLflow integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.utils import ensure_directory

logger = get_logger(__name__)

try:
    import mlflow
except ImportError:  # pragma: no cover - optional for static validation
    mlflow = None


class ModelRegistry:
    """Persist local artifacts and register runs in MLflow."""

    def __init__(self, experiment_name: str = "neuralretail") -> None:
        self.settings = get_settings()
        self.experiment_name = experiment_name
        ensure_directory(self.settings.artifacts_dir)
        if mlflow:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name)

    def save(self, artifact_name: str, model: Any) -> Path:
        destination = self.settings.artifacts_dir / artifact_name
        ensure_directory(destination.parent)
        joblib.dump(model, destination)
        logger.info("Saved artifact to %s", destination)
        return destination

    def load(self, artifact_name: str) -> Any:
        destination = self.settings.artifacts_dir / artifact_name
        return joblib.load(destination)

    def log_run(self, run_name: str, params: dict[str, Any], metrics: dict[str, float], artifacts: dict[str, str] | None = None) -> None:
        if not mlflow:
            logger.warning("MLflow is not installed; skipping run logging for %s", run_name)
            return
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for artifact_path in (artifacts or {}).values():
                mlflow.log_artifact(artifact_path)
