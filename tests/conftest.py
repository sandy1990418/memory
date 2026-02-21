"""
Shared pytest fixtures for openclaw-memory tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers.embeddings_mock import MockEmbeddingProvider


# ---------------------------------------------------------------------------
# Workspace fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    """
    A temporary workspace directory containing:
      - MEMORY.md
      - memory/            (subdirectory)
      - memory/2026-01-12.md
    """
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)

    (tmp_path / "MEMORY.md").write_text(
        "# Memory\n\nRoot memory file.\n\n## Notes\n\n- Test note.\n",
        encoding="utf-8",
    )
    (memory_dir / "2026-01-12.md").write_text(
        "# Session 2026-01-12\n\nCompleted chunking algorithm.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def tmp_index_path(tmp_path: Path) -> str:
    """Return a path to a (not-yet-existing) SQLite DB file inside tmp_path."""
    return str(tmp_path / "memory_index.db")


# ---------------------------------------------------------------------------
# Mock embedding provider
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Fresh MockEmbeddingProvider with empty call tracking."""
    return MockEmbeddingProvider()
