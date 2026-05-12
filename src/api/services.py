"""Bootstrapped inference services for API and dashboard consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger
from src.features.feature_engineering import FeatureEngineeringService
from src.models.churn.stacked_model import ChurnStackedModel
from src.models.demand.hybrid_forecaster import DemandHybridForecaster
from src.models.inventory.optimizer import InventoryOptimizer
from src.models.pricing.optimizer import PriceOptimizer
from src.models.segmentation.segmenter import CustomerSegmentationService

logger = get_logger(__name__)


@dataclass
class PlatformState:
    """In-memory platform state for demo and local usage."""

    transactions: pd.DataFrame
    time_series_features: pd.DataFrame
    customer_features: pd.DataFrame
    churn_model: ChurnStackedModel
    churn_auc: float
    segmentation_service: CustomerSegmentationService
    inventory_optimizer: InventoryOptimizer
    price_optimizer: PriceOptimizer
    demand_models: dict[str, DemandHybridForecaster] = field(default_factory=dict)


class PlatformService:
    """Service facade exposing platform operations."""

    def __init__(self) -> None:
        settings = get_settings()
        transactions = pd.read_csv(settings.raw_data_dir / "transactions_sample.csv")
        feature_service = FeatureEngineeringService(country_code="IN")
        artifacts = feature_service.build(transactions)

        churn_model = ChurnStackedModel()
        churn_auc = churn_model.fit(artifacts.customer_features)

        price_optimizer = PriceOptimizer()
        price_optimizer.fit(artifacts.time_series_features)

        self.state = PlatformState(
            transactions=transactions,
            time_series_features=artifacts.time_series_features,
            customer_features=artifacts.customer_features,
            churn_model=churn_model,
            churn_auc=churn_auc,
            segmentation_service=CustomerSegmentationService(),
            inventory_optimizer=InventoryOptimizer(),
            price_optimizer=price_optimizer,
        )
        logger.info("Platform bootstrapped with %s transactions", len(transactions))

    @property
    def models_ready(self) -> bool:
        """Return whether core models are available."""

        return self.state.churn_model is not None and self.state.price_optimizer is not None

    def forecast_demand(self, sku: str, region: str, horizon_days: int) -> dict:
        """Forecast demand for one SKU-region pair."""

        key = f"{sku}:{region}"
        frame = self.state.time_series_features.query("sku == @sku and region == @region").copy()
        if frame.empty:
            raise ValueError(f"No historical records found for sku={sku} region={region}")
        model = self.state.demand_models.get(key)
        if model is None:
            model = DemandHybridForecaster()
            model.fit(frame)
            self.state.demand_models[key] = model
        result = model.predict(horizon_days=horizon_days)
        return {
            "sku": sku,
            "region": region,
            "metrics": result.metrics,
            "forecast": result.forecast.assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        }

    def predict_churn(self, customer_id: str | None = None, features: dict | None = None) -> dict:
        """Predict churn for an existing or inline customer record."""

        if customer_id:
            frame = self.state.customer_features.query("customer_id == @customer_id").copy()
            if frame.empty:
                raise ValueError(f"Unknown customer_id={customer_id}")
        elif features:
            frame = pd.DataFrame([features])
            frame.insert(0, "customer_id", "ad_hoc_customer")
        else:
            raise ValueError("Either customer_id or features must be provided")

        bundle = self.state.churn_model.predict(frame)
        shap_summary = self.state.churn_model.explain(frame)
        record = bundle.scores.iloc[0].to_dict()
        record["model_auc"] = self.state.churn_auc
        record["shap_summary"] = shap_summary
        return record

    def segment_customers(self, customer_ids: list[str] | None = None) -> dict:
        """Create customer segments and return assignments plus profile."""

        frame = self.state.customer_features.copy()
        if customer_ids:
            frame = frame[frame["customer_id"].isin(customer_ids)].copy()
        result = self.state.segmentation_service.fit_predict(frame)
        return {
            "algorithm": result.algorithm,
            "silhouette_score": result.score,
            "segment_profile": result.profile.to_dict(orient="records"),
            "assignments": result.segments[["customer_id", "segment_id"]].to_dict(orient="records"),
        }

    def inventory_reorder(self, sku: str | None, lead_time_days: int, ordering_cost: float, holding_cost: float) -> dict:
        """Generate reorder recommendations."""

        recommendations = self.state.inventory_optimizer.recommend(
            self.state.transactions,
            lead_time_days=lead_time_days,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
        )
        if sku:
            recommendations = recommendations.query("sku == @sku").copy()
        return {"recommendations": recommendations.to_dict(orient="records")}

    def price_what_if(self, sku: str, candidate_price: float) -> dict:
        """Simulate pricing outcomes for a SKU."""

        sku_frame = self.state.time_series_features.query("sku == @sku").copy()
        if sku_frame.empty:
            raise ValueError(f"Unknown sku={sku}")
        base_price = float(sku_frame["avg_price"].tail(7).mean())
        baseline_demand = float(sku_frame["quantity"].tail(7).mean())
        scenario = self.state.price_optimizer.simulate(
            sku=sku,
            base_price=base_price,
            candidate_price=candidate_price,
            baseline_demand=baseline_demand,
        )
        return asdict(scenario)


@lru_cache(maxsize=1)
def get_platform_service_singleton() -> PlatformService:
    """Return a cached platform service instance."""

    return PlatformService()
