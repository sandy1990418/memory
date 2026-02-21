"""Integration tests for openclaw_memory.schema."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from openclaw_memory.schema import ensure_memory_index_schema


class TestSchemaCreation:
    def test_creates_tables_on_empty_db(self, tmp_path: Path) -> None:
        db = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            result = ensure_memory_index_schema(db)
            cur = db.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
            assert "meta" in tables
            assert "files" in tables
            assert "chunks" in tables
            assert "embedding_cache" in tables
        finally:
            db.close()

    def test_idempotent_recreation(self, tmp_path: Path) -> None:
        db = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            # Should not raise on second call
            ensure_memory_index_schema(db)
            ensure_memory_index_schema(db)
        finally:
            db.close()

    def test_fts_available_when_supported(self, tmp_path: Path) -> None:
        db = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            result = ensure_memory_index_schema(db, fts_enabled=True)
            # FTS5 is available in most Python SQLite builds
            assert isinstance(result["fts_available"], bool)
        finally:
            db.close()

    def test_fts_disabled(self, tmp_path: Path) -> None:
        db = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            result = ensure_memory_index_schema(db, fts_enabled=False)
            assert result["fts_available"] is False
        finally:
            db.close()

    def test_custom_table_names(self, tmp_path: Path) -> None:
        db = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            result = ensure_memory_index_schema(
                db,
                embedding_cache_table="my_cache",
                fts_table="my_fts",
                fts_enabled=True,
            )
            cur = db.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
            assert "my_cache" in tables
        finally:
            db.close()

    def test_ensure_column_migration(self, tmp_path: Path) -> None:
        db = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            # First create schema without source column on files
            db.execute(
                "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, hash TEXT, mtime INTEGER, size INTEGER)"
            )
            db.commit()
            # Running schema creation should add missing 'source' column
            ensure_memory_index_schema(db)
            cur = db.cursor()
            cur.execute("PRAGMA table_info(files)")
            cols = {row[1] for row in cur.fetchall()}
            assert "source" in cols
        finally:
            db.close()

    def test_indexes_created(self, tmp_path: Path) -> None:
        db = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            ensure_memory_index_schema(db)
            cur = db.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = {row[0] for row in cur.fetchall()}
            assert "idx_chunks_path" in indexes
            assert "idx_chunks_source" in indexes
            assert "idx_embedding_cache_updated_at" in indexes
        finally:
            db.close()
