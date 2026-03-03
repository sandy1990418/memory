"""
PostgreSQL connection pool management.

Provides a singleton connection pool for the application lifecycle,
managed via FastAPI lifespan events.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import psycopg
from psycopg_pool import ConnectionPool

_pool: ConnectionPool | None = None


def init_pool(dsn: str, *, min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    """Initialize the global connection pool. Called once at app startup."""
    global _pool
    if _pool is not None:
        return _pool
    _pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)
    return _pool


def close_pool() -> None:
    """Close the global connection pool. Called at app shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_pool() -> ConnectionPool:
    """Get the active connection pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError(
            "Connection pool not initialized. Call init_pool() first "
            "(typically via FastAPI lifespan)."
        )
    return _pool


@contextmanager
def get_connection() -> Generator[psycopg.Connection[Any], None, None]:
    """Get a connection from the pool as a context manager."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def get_sync_connection(dsn: str) -> psycopg.Connection[Any]:
    """
    Open a standalone synchronous connection (for migrations, CLI, etc.).

    Prefer get_connection() for request-scoped work.
    """
    return psycopg.connect(dsn)
