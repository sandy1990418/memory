"""
FastAPI dependency injection providers.

Provides database connections, embedding providers, and the MemoryService
as injectable dependencies for route handlers.

Threading model: All routers use sync ``def`` endpoints (not ``async def``).
FastAPI automatically runs sync endpoints in a thread pool, so each request
gets its own thread with its own psycopg connection from the pool. This
avoids blocking the async event loop while keeping the DB layer simple.
No ``run_in_executor`` wrapping is needed.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import Any

import psycopg

from .config import AppSettings, get_settings
from .core.embeddings import EmbeddingProvider, create_embedding_provider
from .core.service import MemoryService
from .db.connection import get_pool

# ---------------------------------------------------------------------------
# Singleton caches (initialized at startup via lifespan)
# ---------------------------------------------------------------------------

_embedding_provider: EmbeddingProvider | None = None
_memory_service: MemoryService | None = None
_llm_fn = None


def init_services(
    settings: AppSettings,
    llm_fn=None,
    *,
    llm_fns: dict | None = None,
) -> None:
    """Initialize singleton services. Called from app lifespan.

    Parameters
    ----------
    llm_fn : callable, optional
        Default LLM function (fallback for any operation without a specific LLM).
    llm_fns : dict, optional
        Per-operation LLM callables. Keys are operation names:
        extraction, conflict, rerank, answer, consolidation, promotion.
    """
    global _embedding_provider, _memory_service, _llm_fn
    _llm_fn = llm_fn
    _embedding_provider = create_embedding_provider(
        provider=settings.embedding_provider,
        model=settings.embedding_model or None,
    )
    _memory_service = MemoryService(
        embedding_provider=_embedding_provider,
        settings=settings,
        llm_fn=llm_fn,
        llm_fns=llm_fns,
    )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_db() -> Generator[psycopg.Connection[Any], None, None]:
    """Yield a sync connection from the pool."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def get_embedding_provider() -> EmbeddingProvider:
    """Return the singleton embedding provider."""
    if _embedding_provider is None:
        raise RuntimeError("Embedding provider not initialized. Call init_services() first.")
    return _embedding_provider


def get_memory_service() -> MemoryService:
    """Return the singleton MemoryService."""
    if _memory_service is None:
        raise RuntimeError("MemoryService not initialized. Call init_services() first.")
    return _memory_service


def get_settings_dep() -> AppSettings:
    """Return application settings."""
    return get_settings()
