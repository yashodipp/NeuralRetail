"""Caching helpers backed by Redis with an in-memory fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.common.logging import get_logger

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency at import time
    redis = None


logger = get_logger(__name__)


@dataclass
class MemoryCache:
    """Simple fallback cache when Redis is unavailable."""

    store: dict[str, str] = field(default_factory=dict)

    async def get(self, key: str) -> Any | None:
        raw = self.store.get(key)
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:  # noqa: ARG002
        self.store[key] = json.dumps(value, default=str)


class AsyncCache:
    """Redis-first cache adapter."""

    def __init__(self, redis_url: str) -> None:
        self._memory = MemoryCache()
        self._client = redis.from_url(redis_url, decode_responses=True) if redis else None

    async def get(self, key: str) -> Any | None:
        if self._client:
            try:
                raw = await self._client.get(key)
                return json.loads(raw) if raw else None
            except Exception as exc:  # pragma: no cover - infrastructure dependent
                logger.warning("Redis get failed, falling back to memory cache: %s", exc)
        return await self._memory.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        payload = json.dumps(value, default=str)
        if self._client:
            try:
                await self._client.set(key, payload, ex=ttl)
                return
            except Exception as exc:  # pragma: no cover - infrastructure dependent
                logger.warning("Redis set failed, falling back to memory cache: %s", exc)
        await self._memory.set(key, value, ttl)
