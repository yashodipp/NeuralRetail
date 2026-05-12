"""FastAPI dependencies for auth and shared services."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.common.cache import AsyncCache
from src.common.security import decode_access_token, verify_api_key

security_scheme = HTTPBearer(auto_error=False)


def get_platform_service(request: Request):
    """Return the bootstrapped platform service from app state."""

    return request.app.state.platform_service


def get_cache(request: Request) -> AsyncCache:
    """Return the shared cache adapter."""

    return request.app.state.cache


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    api_key: str | None = Header(default=None, alias="x-api-key"),
) -> dict:
    """Require both a bearer token and an API key."""

    if not verify_api_key(api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    try:
        return decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
