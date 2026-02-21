"""
Integration tests for MemoryManager indexing and search.

These tests use real SQLite + MockEmbeddingProvider in tmp dirs.
They import manager lazily so that missing module errors surface at test time.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.helpers.embeddings_mock import MockEmbeddingProvider


def _get_manager_class():
    from openclaw_memory.manager import MemoryManager  # type: ignore[import]
    return MemoryManager


def _make_manager(tmp_path: Path, provider=None):
    MemoryManager = _get_manager_class()
    (tmp_path / "MEMORY.md").write_text("# Memory\n\nRoot memory.\n", encoding="utf-8")
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(exist_ok=True)
    mock = provider or MockEmbeddingProvider()
    return MemoryManager(
        workspace_dir=str(tmp_path),
        db_path=str(tmp_path / "index.db"),
        embedding_provider=mock,
    ), mock


class TestManagerIndex:
    def test_index_creates_db_file(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            assert (tmp_path / "index.db").exists()
        finally:
            mgr.close()

    def test_index_stores_chunks_in_db(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            db = sqlite3.connect(str(tmp_path / "index.db"))
            cur = db.execute("SELECT COUNT(*) FROM chunks")
            count = cur.fetchone()[0]
            db.close()
            assert count >= 1
        finally:
            mgr.close()

    def test_index_stores_file_record(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            db = sqlite3.connect(str(tmp_path / "index.db"))
            cur = db.execute("SELECT path FROM files")
            paths = [row[0] for row in cur.fetchall()]
            db.close()
            assert any("MEMORY.md" in p for p in paths)
        finally:
            mgr.close()

    def test_search_returns_results(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nPrefer Python over JavaScript.\n",
            encoding="utf-8",
        )
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            results = mgr.search("Python programming", max_results=5)
            assert isinstance(results, list)
            # Should return at least one result
            assert len(results) >= 1
        finally:
            mgr.close()

    def test_search_result_structure(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            results = mgr.search("memory", max_results=5)
            for r in results:
                assert hasattr(r, "path")
                assert hasattr(r, "score")
                assert hasattr(r, "snippet")
                assert hasattr(r, "start_line")
                assert hasattr(r, "end_line")
                assert r.score >= 0.0
        finally:
            mgr.close()

    def test_fts_only_mode_no_provider(self, tmp_path: Path) -> None:
        MemoryManager = _get_manager_class()
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nFTS only mode test.\n", encoding="utf-8"
        )
        mgr = MemoryManager(
            workspace_dir=str(tmp_path),
            db_path=str(tmp_path / "index.db"),
            embedding_provider=None,
        )
        try:
            mgr.sync()
            results = mgr.search("FTS only", max_results=5)
            assert isinstance(results, list)
        finally:
            mgr.close()

    def test_embed_batch_called_during_sync(self, tmp_path: Path) -> None:
        mock = MockEmbeddingProvider()
        mgr, _ = _make_manager(tmp_path, provider=mock)
        try:
            mgr.sync()
            assert len(mock.embed_batch_calls) >= 1
        finally:
            mgr.close()

    def test_multiple_files_indexed(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("# Root\n\nRoot memory.\n", encoding="utf-8")
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir(exist_ok=True)
        (mem_dir / "notes.md").write_text("# Notes\n\nSome notes.\n", encoding="utf-8")
        mock = MockEmbeddingProvider()
        MemoryManager = _get_manager_class()
        mgr = MemoryManager(
            workspace_dir=str(tmp_path),
            db_path=str(tmp_path / "index.db"),
            embedding_provider=mock,
        )
        try:
            mgr.sync()
            db = sqlite3.connect(str(tmp_path / "index.db"))
            cur = db.execute("SELECT COUNT(*) FROM files")
            count = cur.fetchone()[0]
            db.close()
            assert count >= 2
        finally:
            mgr.close()
