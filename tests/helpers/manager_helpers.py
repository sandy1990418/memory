"""
Helpers for creating a MemoryManager with the mock embedding provider.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tests.helpers.embeddings_mock import MockEmbeddingProvider


def make_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace layout inside tmp_path."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("# Memory\n\nTest workspace memory.\n", encoding="utf-8")
    return tmp_path


def make_manager(
    tmp_path: Path,
    *,
    extra_paths: list[str] | None = None,
    provider: MockEmbeddingProvider | None = None,
    **manager_kwargs: Any,
) -> Any:
    """
    Create a MemoryManager (or compatible object) wired to a MockEmbeddingProvider
    inside a temporary workspace.

    Returns the manager instance.  Tests import this lazily so that the module
    can be written before manager.py exists — the ImportError will surface only
    when the test actually runs.
    """
    from openclaw_memory.manager import MemoryManager  # type: ignore[import]

    workspace = make_workspace(tmp_path)
    mock_provider = provider or MockEmbeddingProvider()

    db_path = str(tmp_path / "memory.db")
    mgr = MemoryManager(
        workspace_dir=str(workspace),
        db_path=db_path,
        embedding_provider=mock_provider,
        extra_paths=extra_paths or [],
        **manager_kwargs,
    )
    return mgr
