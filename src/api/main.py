"""FastAPI entrypoint for NeuralRetail prediction services."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, status

from src.api.dependencies import get_cache, get_platform_service, require_auth
from src.api.schemas import (
    ChurnRequest,
    ChurnResponse,
    DemandRequest,
    DemandResponse,
    HealthResponse,
    InventoryRequest,
    InventoryResponse,
    SegmentRequest,
    SegmentResponse,
    TokenRequest,
    TokenResponse,
)
from src.api.services import get_platform_service_singleton
from src.common.cache import AsyncCache
from src.common.config import get_settings
from src.common.security import create_access_token, verify_service_credentials

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:  # pragma: no cover - optional during lightweight local runs
    Instrumentator = None

settings = get_settings()
app = FastAPI(title="NeuralRetail API", version="1.0.0")

if settings.prometheus_enabled and Instrumentator:
    Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def startup_event() -> None:
    """Warm up shared services."""

    app.state.platform_service = get_platform_service_singleton()
    app.state.cache = AsyncCache(settings.redis_url)


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def issue_token(payload: TokenRequest) -> TokenResponse:
    """Issue a JWT for service access."""

    if not verify_service_credentials(payload.username, payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=payload.username, extra_claims={"scope": "inference"})
    return TokenResponse(access_token=token, expires_in=settings.access_token_ttl_seconds)


@app.post("/predict/demand", response_model=DemandResponse, tags=["predictions"])
async def predict_demand(
    payload: DemandRequest,
    _: dict = Depends(require_auth),
    service=Depends(get_platform_service),
    cache: AsyncCache = Depends(get_cache),
) -> DemandResponse:
    """Return multi-horizon demand forecasts."""

    cache_key = f"demand:{payload.sku}:{payload.region}:{payload.horizon_days}"
    cached = await cache.get(cache_key)
    if cached:
        return DemandResponse(**cached)
    try:
        result = service.forecast_demand(payload.sku, payload.region, payload.horizon_days)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    await cache.set(cache_key, result, ttl=600)
    return DemandResponse(**result)


@app.post("/predict/churn", response_model=ChurnResponse, tags=["predictions"])
async def predict_churn(
    payload: ChurnRequest,
    _: dict = Depends(require_auth),
    service=Depends(get_platform_service),
    cache: AsyncCache = Depends(get_cache),
) -> ChurnResponse:
    """Return churn probability and retention recommendation."""

    cache_key = f"churn:{payload.customer_id or hash(str(payload.features))}"
    cached = await cache.get(cache_key)
    if cached:
        return ChurnResponse(**cached)
    try:
        result = service.predict_churn(
            customer_id=payload.customer_id,
            features=payload.features.model_dump() if payload.features else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    await cache.set(cache_key, result, ttl=600)
    return ChurnResponse(**result)


@app.post("/segment/score", response_model=SegmentResponse, tags=["predictions"])
async def segment_customers(
    payload: SegmentRequest,
    _: dict = Depends(require_auth),
    service=Depends(get_platform_service),
    cache: AsyncCache = Depends(get_cache),
) -> SegmentResponse:
    """Return customer segment assignments and cluster profile."""

    cache_key = f"segment:{','.join(payload.customer_ids or []) or 'all'}"
    cached = await cache.get(cache_key)
    if cached:
        return SegmentResponse(**cached)
    result = service.segment_customers(payload.customer_ids)
    await cache.set(cache_key, result, ttl=300)
    return SegmentResponse(**result)


@app.post("/inventory/reorder", response_model=InventoryResponse, tags=["operations"])
async def inventory_reorder(
    payload: InventoryRequest,
    _: dict = Depends(require_auth),
    service=Depends(get_platform_service),
    cache: AsyncCache = Depends(get_cache),
) -> InventoryResponse:
    """Return EOQ and reorder recommendations."""

    cache_key = f"inventory:{payload.sku}:{payload.lead_time_days}:{payload.ordering_cost}:{payload.holding_cost}"
    cached = await cache.get(cache_key)
    if cached:
        return InventoryResponse(**cached)
    result = service.inventory_reorder(
        sku=payload.sku,
        lead_time_days=payload.lead_time_days,
        ordering_cost=payload.ordering_cost,
        holding_cost=payload.holding_cost,
    )
    response = {"generated_at": datetime.now(UTC), **result}
    await cache.set(cache_key, response, ttl=300)
    return InventoryResponse(**response)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(
    service=Depends(get_platform_service),
    cache: AsyncCache = Depends(get_cache),
) -> HealthResponse:
    """Basic readiness probe."""

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        timestamp=datetime.now(UTC),
        models_ready=service.models_ready,
        cache_ready=cache is not None,
    )
