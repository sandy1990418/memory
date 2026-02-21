"""
Integration tests for MemoryManager sync/dirty/caching behaviour.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from tests.helpers.embeddings_mock import MockEmbeddingProvider


def _get_manager_class():
    from openclaw_memory.manager import MemoryManager  # type: ignore[import]
    return MemoryManager


def _make_manager(tmp_path: Path, provider=None):
    MemoryManager = _get_manager_class()
    (tmp_path / "MEMORY.md").write_text(
        "# Memory\n\nOriginal content.\n", encoding="utf-8"
    )
    (tmp_path / "memory").mkdir(exist_ok=True)
    mock = provider or MockEmbeddingProvider()
    mgr = MemoryManager(
        workspace_dir=str(tmp_path),
        db_path=str(tmp_path / "index.db"),
        embedding_provider=mock,
    )
    return mgr, mock


class TestManagerSync:
    def test_dirty_flag_after_file_change(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            # Modify file
            (tmp_path / "MEMORY.md").write_text(
                "# Memory\n\nChanged content!\n", encoding="utf-8"
            )
            status = mgr.status()
            assert status.dirty is True
        finally:
            mgr.close()

    def test_resync_picks_up_changes(self, tmp_path: Path) -> None:
        mgr, mock = _make_manager(tmp_path)
        try:
            mgr.sync()
            initial_calls = len(mock.embed_batch_calls)

            # Modify file content
            (tmp_path / "MEMORY.md").write_text(
                "# Memory\n\nCompletely new content for resync.\n", encoding="utf-8"
            )
            mgr.sync()

            # After resync, more embed_batch calls should have been made
            assert len(mock.embed_batch_calls) > initial_calls
        finally:
            mgr.close()

    def test_unchanged_file_does_not_reembed(self, tmp_path: Path) -> None:
        mock = MockEmbeddingProvider()
        mgr, _ = _make_manager(tmp_path, provider=mock)
        try:
            mgr.sync()
            calls_after_first_sync = len(mock.embed_batch_calls)

            # Sync again without any file changes
            mgr.sync()
            calls_after_second_sync = len(mock.embed_batch_calls)

            # No new embed_batch calls should occur
            assert calls_after_second_sync == calls_after_first_sync
        finally:
            mgr.close()

    def test_status_reports_file_count(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            status = mgr.status()
            assert status.files >= 1
        finally:
            mgr.close()

    def test_status_reports_chunk_count(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            status = mgr.status()
            assert status.chunks >= 1
        finally:
            mgr.close()

    def test_status_before_sync_is_dirty(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            status = mgr.status()
            assert status.dirty is True
        finally:
            mgr.close()

    def test_status_not_dirty_after_sync(self, tmp_path: Path) -> None:
        mgr, _ = _make_manager(tmp_path)
        try:
            mgr.sync()
            status = mgr.status()
            assert status.dirty is False
        finally:
            mgr.close()

    def test_force_sync_reindexes_all(self, tmp_path: Path) -> None:
        mock = MockEmbeddingProvider()
        mgr, _ = _make_manager(tmp_path, provider=mock)
        try:
            mgr.sync()
            calls_first = len(mock.embed_batch_calls)

            # Force sync should reindex even unchanged files
            mgr.sync(force=True)
            calls_second = len(mock.embed_batch_calls)

            assert calls_second > calls_first
        finally:
            mgr.close()
