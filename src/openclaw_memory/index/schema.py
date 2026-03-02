"""
SQLite schema creation for the memory index.
Mirrors: src/memory/memory-schema.ts
"""

from __future__ import annotations

import re
import sqlite3

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> None:
    """Reject SQL identifiers that aren't simple alphanumeric names."""
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")


def ensure_memory_index_schema(
    db: sqlite3.Connection,
    *,
    embedding_cache_table: str = "embedding_cache",
    fts_table: str = "chunks_fts",
    fts_enabled: bool = True,
) -> dict[str, object]:
    """
    Create all required tables and indexes.
    Returns {"fts_available": bool, "fts_error": str | None}.
    """
    _validate_identifier(embedding_cache_table)
    _validate_identifier(fts_table)

    cur = db.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            path   TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'memory',
            hash   TEXT NOT NULL,
            mtime  INTEGER NOT NULL,
            size   INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id         TEXT PRIMARY KEY,
            path       TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT 'memory',
            start_line INTEGER NOT NULL,
            end_line   INTEGER NOT NULL,
            hash       TEXT NOT NULL,
            model      TEXT NOT NULL,
            text       TEXT NOT NULL,
            embedding  TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {embedding_cache_table} (
            provider     TEXT NOT NULL,
            model        TEXT NOT NULL,
            provider_key TEXT NOT NULL,
            hash         TEXT NOT NULL,
            embedding    TEXT NOT NULL,
            dims         INTEGER,
            updated_at   INTEGER NOT NULL,
            PRIMARY KEY (provider, model, provider_key, hash)
        );
    """)
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at "
        f"ON {embedding_cache_table}(updated_at);"
    )

    fts_available = False
    fts_error: str | None = None

    if fts_enabled:
        try:
            cur.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5("
                f"  text,"
                f"  id UNINDEXED,"
                f"  path UNINDEXED,"
                f"  source UNINDEXED,"
                f"  model UNINDEXED,"
                f"  start_line UNINDEXED,"
                f"  end_line UNINDEXED"
                f");"
            )
            fts_available = True
        except sqlite3.OperationalError as exc:
            fts_error = str(exc)

    # Ensure columns exist (migration-safe)
    _ensure_column(cur, "files", "source", "TEXT NOT NULL DEFAULT 'memory'")
    _ensure_column(cur, "chunks", "source", "TEXT NOT NULL DEFAULT 'memory'")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);")

    db.commit()
    return {"fts_available": fts_available, "fts_error": fts_error}


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, definition: str) -> None:
    _validate_identifier(table)
    _validate_identifier(column)
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    existing = {row[1] for row in rows}
    if column not in existing:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
