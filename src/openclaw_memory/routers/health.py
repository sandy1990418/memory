"""Health check endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
import psycopg

from ..dependencies import get_db

router = APIRouter()


@router.get("/health")
def health_check(conn: psycopg.Connection[Any] = Depends(get_db)) -> dict[str, Any]:
    """Health check — verifies DB connectivity."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
    }
