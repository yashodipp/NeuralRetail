"""MLflow tracking utilities."""

from __future__ import annotations

from pathlib import Path

from src.models.registry import ModelRegistry


def log_training_run(run_name: str, params: dict, metrics: dict, artifacts: dict[str, str] | None = None) -> None:
    """Log a single training run to MLflow."""

    registry = ModelRegistry()
    registry.log_run(run_name=run_name, params=params, metrics=metrics, artifacts=artifacts)


def register_artifact(local_path: str | Path, artifact_name: str) -> Path:
    """Persist an artifact in the local registry directory."""

    registry = ModelRegistry()
    return registry.save(artifact_name, str(local_path))
