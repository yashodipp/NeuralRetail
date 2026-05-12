"""Feature engineering tests."""

from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import FeatureEngineeringService


def test_feature_builder_creates_rfm_and_lag_columns():
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "customer_id": "C1",
                "sku": "SKU-1",
                "region": "North",
                "channel": "Web",
                "quantity": 5,
                "unit_price": 10.0,
                "discount_pct": 0.0,
                "promotion_flag": 0,
                "inventory_on_hand": 50,
                "weather_index": 0.8,
                "revenue": 50.0,
                "churned": 0,
            },
            {
                "date": "2025-01-02",
                "customer_id": "C1",
                "sku": "SKU-1",
                "region": "North",
                "channel": "Web",
                "quantity": 6,
                "unit_price": 10.0,
                "discount_pct": 0.0,
                "promotion_flag": 1,
                "inventory_on_hand": 49,
                "weather_index": 0.7,
                "revenue": 60.0,
                "churned": 0,
            },
        ]
    )
    artifacts = FeatureEngineeringService(country_code="IN").build(frame)

    assert "lag_qty_1" in artifacts.time_series_features.columns
    assert "rolling_qty_7" in artifacts.time_series_features.columns
    assert "rfm_score" in artifacts.customer_features.columns
    assert artifacts.customer_features["rfm_score"].ge(1).all()
