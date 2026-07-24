"""API-key auth for the gateway.

If GATEWAY_API_KEYS is empty the gateway is open (localhost dev). Set it in the
cloud; your platform then sends one of those keys as its OpenAI api key. This is
the single credential you rotate — no code change on the platform side.
"""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException, status

from .config import settings


def _extract_key(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return authorization.strip()
    if x_api_key:
        return x_api_key.strip()
    return None


async def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> None:
    """FastAPI dependency. No-op when auth is disabled."""
    if not settings.auth_required:
        return
    key = _extract_key(authorization, x_api_key)
    # constant-time comparison to avoid a timing side-channel on the gateway key
    if key is None or not any(hmac.compare_digest(key, k) for k in settings.api_key_list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid API key.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
