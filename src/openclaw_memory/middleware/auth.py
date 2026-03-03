"""
API key authentication middleware.

When ``OPENCLAW_API_KEY_REQUIRED=true``, every request (except /health)
must include a valid ``X-API-Key`` header. The key is SHA-256 hashed
and looked up in the ``api_keys`` table.

When disabled (default), all requests pass through without auth.
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import Depends, HTTPException, Request
import psycopg

from ..db import queries
from ..dependencies import get_db


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def verify_api_key(
    request: Request,
    conn: psycopg.Connection[Any] = Depends(get_db),
) -> str | None:
    """
    FastAPI dependency: verify the X-API-Key header.

    Returns the ``user_id`` associated with the key, or ``None`` if auth
    is disabled via settings.

    Raises HTTPException 401 if auth is required but key is missing/invalid.
    """
    from ..config import get_settings

    settings = get_settings()
    if not settings.api_key_required:
        return None

    # Skip auth for health endpoint
    if request.url.path == "/health":
        return None

    raw_key = request.headers.get("x-api-key", "")
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    key_hash = hash_api_key(raw_key)
    user_id = queries.verify_api_key(conn, key_hash)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    return user_id
