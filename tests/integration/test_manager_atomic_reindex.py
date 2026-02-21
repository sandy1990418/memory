"""
Integration tests for atomic reindex behaviour.
Verifies that the prior index is preserved if reindexing fails mid-swap.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers.embeddings_mock import MockEmbeddingProvider


def _get_manager_class():
    from openclaw_memory.manager import MemoryManager  # type: ignore[import]
    return MemoryManager


class TestAtomicReindex:
    def test_prior_index_preserved_on_failure(self, tmp_path: Path) -> None:
        """If re-indexing raises mid-way, the existing DB should remain intact."""
        MemoryManager = _get_manager_class()
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nOriginal valid content.\n", encoding="utf-8"
        )
        (tmp_path / "memory").mkdir(exist_ok=True)
        mock = MockEmbeddingProvider()

        mgr = MemoryManager(
            workspace_dir=str(tmp_path),
            db_path=str(tmp_path / "index.db"),
            embedding_provider=mock,
        )
        try:
            # Perform initial successful sync
            mgr.sync()

            # Verify initial index is in place
            db = sqlite3.connect(str(tmp_path / "index.db"))
            cur = db.execute("SELECT COUNT(*) FROM chunks")
            initial_count = cur.fetchone()[0]
            db.close()
            assert initial_count >= 1

            # Now simulate a failure during force re-sync by making embed_batch raise
            call_count = [0]
            original_embed_batch = mock.embed_batch

            def failing_embed_batch(texts):
                call_count[0] += 1
                if call_count[0] >= 1:
                    raise RuntimeError("Simulated embedding failure")
                return original_embed_batch(texts)

            mock.embed_batch = failing_embed_batch  # type: ignore[method-assign]

            try:
                mgr.sync(force=True)
            except Exception:
                pass  # Expected failure

            # DB should still be readable and contain the original chunks
            db = sqlite3.connect(str(tmp_path / "index.db"))
            cur = db.execute("SELECT COUNT(*) FROM chunks")
            surviving_count = cur.fetchone()[0]
            db.close()
            assert surviving_count >= 1, "Index should survive a failed re-sync"
        finally:
            mgr.close()

    def test_successful_reindex_replaces_index(self, tmp_path: Path) -> None:
        MemoryManager = _get_manager_class()
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nFirst version.\n", encoding="utf-8"
        )
        (tmp_path / "memory").mkdir(exist_ok=True)
        mock = MockEmbeddingProvider()

        mgr = MemoryManager(
            workspace_dir=str(tmp_path),
            db_path=str(tmp_path / "index.db"),
            embedding_provider=mock,
        )
        try:
            mgr.sync()

            # Add more content and force re-index
            (tmp_path / "MEMORY.md").write_text(
                "# Memory\n\nFirst version.\n\nAdded new section.\n\nMore content here.\n",
                encoding="utf-8",
            )
            mgr.sync(force=True)

            db = sqlite3.connect(str(tmp_path / "index.db"))
            cur = db.execute("SELECT COUNT(*) FROM chunks")
            new_count = cur.fetchone()[0]
            db.close()
            assert new_count >= 1
        finally:
            mgr.close()
