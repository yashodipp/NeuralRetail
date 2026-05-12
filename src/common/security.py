"""Security helpers for hashing, API keys, and JWT handling."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt

from src.common.config import get_settings


def hash_pii(value: str) -> str:
    """Hash sensitive values with SHA-256."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_access_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    """Create a signed JWT for API consumers."""

    settings = get_settings()
    payload = {
        "sub": subject,
        "exp": datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT."""

    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc


def verify_api_key(candidate: str | None) -> bool:
    """Validate a caller API key."""

    settings = get_settings()
    return bool(candidate and candidate in settings.api_keys)


def verify_service_credentials(username: str, password: str) -> bool:
    """Validate the dashboard or service account login."""

    settings = get_settings()
    return username == settings.service_username and password == settings.service_password
