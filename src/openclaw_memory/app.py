"""
FastAPI application factory.

Creates the app with lifespan management for connection pool and services.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .core.llm_client import create_llm_client
from .db.connection import close_pool, init_pool
from .db.schema import ensure_schema
from .dependencies import init_services

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: init DB pool + services on startup."""
    settings = get_settings()

    # Initialize connection pool
    init_pool(settings.pg_dsn, min_size=settings.pg_pool_min, max_size=settings.pg_pool_max)

    # Ensure database schema is up to date
    from .db.connection import get_pool
    with get_pool().connection() as conn:
        ensure_schema(conn)

    # Create LLM client(s)
    llm_fn = create_llm_client(provider=settings.llm_provider, model=settings.llm_model)

    extraction_llm_fn = None
    answer_llm_fn = None
    if settings.extraction_llm_model:
        extraction_llm_fn = create_llm_client(
            provider=settings.llm_provider, model=settings.extraction_llm_model
        )
    if settings.answer_llm_model:
        answer_llm_fn = create_llm_client(
            provider=settings.llm_provider, model=settings.answer_llm_model
        )

    # Initialize services with LLM
    init_services(
        settings,
        llm_fn=llm_fn,
        extraction_llm_fn=extraction_llm_fn,
        answer_llm_fn=answer_llm_fn,
    )

    yield

    # Cleanup
    close_pool()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="OpenClaw Memory",
        description="Three-layer memory system for chatbot applications",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Rate limiting middleware
    from .middleware.rate_limit import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_rpm,
        enabled=settings.rate_limit_enabled,
    )

    # Register routers
    from .routers import admin, health, memory, search, session

    app.include_router(health.router, tags=["health"])
    app.include_router(session.router, prefix="/session", tags=["session"])
    app.include_router(memory.router, prefix="/memory", tags=["memory"])
    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])

    return app
