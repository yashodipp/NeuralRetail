"""End-to-end training pipeline for local runs and Airflow tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import get_settings
from src.features.feature_engineering import FeatureEngineeringService
from src.features.feast_export import export_for_feast
from src.ingestion.pipeline import IngestionPipeline
from src.mlops.tracking import log_training_run
from src.models.churn.stacked_model import ChurnStackedModel
from src.models.inventory.optimizer import InventoryOptimizer
from src.models.pricing.optimizer import PriceOptimizer
from src.models.registry import ModelRegistry
from src.models.segmentation.segmenter import CustomerSegmentationService


def run_training_pipeline() -> dict:
    """Train all core models from the bundled sample data."""

    settings = get_settings()
    registry = ModelRegistry()
    ingestion_result = IngestionPipeline().ingest_local_transactions()

    transactions = pd.read_csv(settings.raw_data_dir / "transactions_sample.csv")
    features = FeatureEngineeringService(country_code="IN").build(transactions)

    time_series_path = export_for_feast(
        features.time_series_features.assign(event_timestamp=pd.Timestamp.utcnow()),
        settings.processed_data_dir / "time_series_features.parquet",
    )
    customer_path = export_for_feast(
        features.customer_features.assign(event_timestamp=pd.Timestamp.utcnow()),
        settings.processed_data_dir / "customer_features.parquet",
    )

    churn_model = ChurnStackedModel()
    churn_auc = churn_model.fit(features.customer_features)
    registry.save("churn_model.joblib", churn_model)

    segmentation = CustomerSegmentationService().fit_predict(features.customer_features)
    registry.save("segmentation_profile.joblib", segmentation.profile)

    elasticity = PriceOptimizer().fit(features.time_series_features)
    inventory_summary = InventoryOptimizer().recommend(transactions)
    inventory_path = settings.artifacts_dir / "inventory_recommendations.parquet"
    inventory_summary.to_parquet(inventory_path, index=False)

    metrics = {
        "churn_auc": float(churn_auc),
        "segmentation_silhouette": float(segmentation.score),
        "price_elasticity": float(elasticity),
    }
    params = {
        "transactions_rows": len(transactions),
        "time_series_path": str(time_series_path),
        "customer_path": str(customer_path),
    }
    log_training_run(
        run_name="neuralretail_training",
        params=params,
        metrics=metrics,
        artifacts={"inventory": str(inventory_path)},
    )
    return {
        "ingestion": ingestion_result,
        "metrics": metrics,
        "outputs": {
            "time_series_features": str(time_series_path),
            "customer_features": str(customer_path),
            "inventory_recommendations": str(inventory_path),
        },
    }
