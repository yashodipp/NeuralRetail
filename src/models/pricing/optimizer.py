"""Price elasticity estimation and what-if simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.common.logging import get_logger

logger = get_logger(__name__)

try:
    from dowhy import CausalModel
except ImportError:  # pragma: no cover
    CausalModel = None

try:
    from econml.dml import LinearDML
except ImportError:  # pragma: no cover
    LinearDML = None


@dataclass(slots=True)
class PriceScenario:
    """Result of a what-if price simulation."""

    sku: str
    base_price: float
    candidate_price: float
    expected_demand: float
    expected_revenue: float
    price_elasticity: float


class PriceOptimizer:
    """Estimate elasticity and produce price recommendations."""

    def __init__(self) -> None:
        self.elasticity_: float = -1.0
        self.model = None
        self.price_column_: str | None = None

    def fit(self, frame: pd.DataFrame) -> float:
        """Fit a price response model."""

        training = frame.copy()
        self.price_column_ = self._resolve_price_column(training)
        training["log_quantity"] = np.log1p(training["quantity"])
        training["log_price"] = np.log(training[self.price_column_].clip(lower=0.01))

        feature_columns = [col for col in ["promotion_flag", "weather_index", "is_holiday"] if col in training.columns]
        try:
            if LinearDML and feature_columns:
                self.model = LinearDML(random_state=42)
                y = training["quantity"].to_numpy()
                t = training[self.price_column_].to_numpy()
                x = training[feature_columns].to_numpy()
                self.model.fit(y, t, X=x)
                self.elasticity_ = float(np.mean(self.model.effect(x)))
            elif CausalModel and feature_columns:
                causal = CausalModel(
                    data=training[["quantity", self.price_column_, *feature_columns]],
                    treatment=self.price_column_,
                    outcome="quantity",
                    common_causes=feature_columns,
                )
                estimand = causal.identify_effect()
                estimate = causal.estimate_effect(estimand, method_name="backdoor.linear_regression")
                self.elasticity_ = float(estimate.value)
            else:
                raise RuntimeError("Causal dependencies unavailable")
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("Falling back to linear elasticity model: %s", exc)
            regression = LinearRegression()
            regression.fit(training[["log_price"]], training["log_quantity"])
            self.model = regression
            self.elasticity_ = float(regression.coef_[0])
        logger.info("Estimated price elasticity=%.4f", self.elasticity_)
        return self.elasticity_

    @staticmethod
    def _resolve_price_column(frame: pd.DataFrame) -> str:
        """Return the available price column from either raw or engineered data."""

        for column in ("unit_price", "avg_price"):
            if column in frame.columns:
                return column
        raise KeyError("Expected one of ['unit_price', 'avg_price'] in pricing features")

    def simulate(self, sku: str, base_price: float, candidate_price: float, baseline_demand: float) -> PriceScenario:
        """Run a what-if simulation for a proposed price change."""

        price_change_ratio = (candidate_price - base_price) / max(base_price, 0.01)
        expected_demand = baseline_demand * (1 + self.elasticity_ * price_change_ratio)
        expected_demand = max(expected_demand, 0)
        expected_revenue = expected_demand * candidate_price
        return PriceScenario(
            sku=sku,
            base_price=base_price,
            candidate_price=candidate_price,
            expected_demand=float(expected_demand),
            expected_revenue=float(expected_revenue),
            price_elasticity=float(self.elasticity_),
        )

    def recommend_price(self, sku: str, base_price: float, baseline_demand: float, search_grid: list[float] | None = None) -> PriceScenario:
        """Choose the best revenue-maximizing price from a candidate grid."""

        if search_grid is None:
            search_grid = [base_price * factor for factor in np.linspace(0.85, 1.20, 8)]
        scenarios = [self.simulate(sku, base_price, candidate, baseline_demand) for candidate in search_grid]
        return max(scenarios, key=lambda item: item.expected_revenue)
