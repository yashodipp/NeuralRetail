"""Feast feature view definitions."""

from datetime import timedelta

from feast import FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

from data.feature_repo.entities import customer, sku


customer_source = FileSource(
    path="data/processed/customer_features.parquet",
    timestamp_field="event_timestamp",
)

time_series_source = FileSource(
    path="data/processed/time_series_features.parquet",
    timestamp_field="event_timestamp",
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="recency", dtype=Int64),
        Field(name="frequency", dtype=Int64),
        Field(name="monetary", dtype=Float32),
        Field(name="rfm_score", dtype=Float32),
        Field(name="purchase_velocity", dtype=Float32),
        Field(name="avg_order_value", dtype=Float32),
    ],
    source=customer_source,
)

time_series_features = FeatureView(
    name="time_series_features",
    entities=[sku],
    ttl=timedelta(days=7),
    schema=[
        Field(name="region", dtype=String),
        Field(name="rolling_qty_7", dtype=Float32),
        Field(name="rolling_qty_14", dtype=Float32),
        Field(name="rolling_qty_30", dtype=Float32),
        Field(name="lag_qty_1", dtype=Float32),
        Field(name="lag_qty_7", dtype=Float32),
        Field(name="lag_qty_14", dtype=Float32),
        Field(name="avg_price", dtype=Float32),
        Field(name="promotion_flag", dtype=Int64),
    ],
    source=time_series_source,
)

retail_online_features = FeatureService(
    name="retail_online_features",
    features=[customer_features, time_series_features],
)
