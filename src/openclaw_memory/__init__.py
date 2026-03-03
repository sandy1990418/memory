"""
openclaw-memory: Three-layer memory system for chatbot applications.

Public API:
    create_app       -- FastAPI application factory
    MemoryService    -- thin orchestrator for memory operations
    AppSettings      -- Pydantic-based configuration

Core types:
    MemorySearchResult, MemoryIndex, MemoryContext
    ExtractedMemory, ConsolidationReport

Heavy dependencies (fastapi, psycopg_pool) are imported lazily so that
lightweight submodules (utils, pipeline.retrieval, etc.) can be imported
without pulling in the full stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import create_app as create_app
    from .config import AppSettings as AppSettings
    from .config import get_settings as get_settings
    from .core.service import MemoryService as MemoryService
    from .core.types import ConsolidationReport as ConsolidationReport
    from .core.types import ExtractedMemory as ExtractedMemory
    from .core.types import MemoryContext as MemoryContext
    from .core.types import MemoryIndex as MemoryIndex
    from .core.types import MemorySearchResult as MemorySearchResult

__all__ = [
    "create_app",
    "AppSettings",
    "get_settings",
    "MemoryService",
    "MemorySearchResult",
    "MemoryIndex",
    "MemoryContext",
    "ExtractedMemory",
    "ConsolidationReport",
]

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "create_app": (".app", "create_app"),
    "AppSettings": (".config", "AppSettings"),
    "get_settings": (".config", "get_settings"),
    "MemoryService": (".core.service", "MemoryService"),
    "MemorySearchResult": (".core.types", "MemorySearchResult"),
    "MemoryIndex": (".core.types", "MemoryIndex"),
    "MemoryContext": (".core.types", "MemoryContext"),
    "ExtractedMemory": (".core.types", "ExtractedMemory"),
    "ConsolidationReport": (".core.types", "ConsolidationReport"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
