"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    """Service credentials used to obtain a JWT."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT auth response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class DemandRequest(BaseModel):
    """Demand forecasting request."""

    sku: str = Field(..., examples=["SKU-1001"])
    region: str = Field(..., examples=["North"])
    horizon_days: int = Field(default=30, ge=1, le=90)


class DemandResponse(BaseModel):
    """Demand forecasting response payload."""

    sku: str
    region: str
    metrics: dict[str, float]
    forecast: list[dict[str, Any]]


class ChurnFeatures(BaseModel):
    """Inline features for churn scoring."""

    recency: float
    frequency: float
    monetary: float
    avg_order_value: float
    avg_discount: float
    churn_label: int = 0
    rfm_score: float
    tenure_days: float
    purchase_velocity: float
    monetary_per_visit: float
    region_North: int = 0
    region_South: int = 0
    region_East: int = 0
    region_West: int = 0
    preferred_channel_Store: int = 0
    preferred_channel_Web: int = 0
    preferred_channel_App: int = 0


class ChurnRequest(BaseModel):
    """Churn prediction request."""

    customer_id: str | None = None
    features: ChurnFeatures | None = None


class ChurnResponse(BaseModel):
    """Churn scoring response."""

    customer_id: str
    churn_probability: float
    retention_action: str
    model_auc: float | None = None
    shap_summary: dict[str, Any] | None = None


class SegmentRequest(BaseModel):
    """Customer segmentation request."""

    customer_ids: list[str] | None = None


class SegmentResponse(BaseModel):
    """Customer segmentation response."""

    algorithm: str
    silhouette_score: float
    segment_profile: list[dict[str, Any]]
    assignments: list[dict[str, Any]]


class InventoryRequest(BaseModel):
    """Inventory optimization request."""

    sku: str | None = None
    lead_time_days: int = Field(default=7, ge=1, le=90)
    ordering_cost: float = Field(default=150.0, gt=0)
    holding_cost: float = Field(default=8.0, gt=0)


class InventoryResponse(BaseModel):
    """Inventory reorder response."""

    generated_at: datetime
    recommendations: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    app_name: str
    timestamp: datetime
    models_ready: bool
    cache_ready: bool
