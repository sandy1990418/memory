"""
PostgreSQL + pgvector schema for the multi-tenant memory system.

Provides:
  - PostgresConfig   — connection / pool configuration dataclass
  - get_pg_connection(dsn) — open a psycopg3 connection
  - ensure_pg_schema(conn) — idempotently apply the full schema
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only imported for type-checking; psycopg is lazy-loaded at runtime.
    import psycopg  # type: ignore[import]

# Path to the bundled SQL migration file.
_MIGRATION_SQL = Path(__file__).parent.parent.parent / "migrations" / "001_initial_schema.sql"

# Fallback inline SQL in case the file is not found (e.g. installed as a wheel
# without the migrations/ directory).  Kept in sync with the .sql file.
_INLINE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodic_memories (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT        NOT NULL,
    thread_id   TEXT,
    content     TEXT        NOT NULL,
    embedding   vector(1536),
    memory_type TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB       NOT NULL DEFAULT '{}'
);

ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_episodic_user_id
    ON episodic_memories (user_id);

CREATE INDEX IF NOT EXISTS idx_episodic_user_created
    ON episodic_memories (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_episodic_embedding_hnsw
    ON episodic_memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_episodic_tsv
    ON episodic_memories USING gin (tsv);

CREATE TABLE IF NOT EXISTS semantic_memories (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    embedding   vector(1536),
    memory_type TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB       NOT NULL DEFAULT '{}'
);

ALTER TABLE semantic_memories
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_semantic_user_id
    ON semantic_memories (user_id);

CREATE INDEX IF NOT EXISTS idx_semantic_user_created
    ON semantic_memories (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_semantic_embedding_hnsw
    ON semantic_memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_semantic_tsv
    ON semantic_memories USING gin (tsv);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id    TEXT        PRIMARY KEY,
    profile    JSONB       NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id
    ON user_profiles (user_id);

CREATE TABLE IF NOT EXISTS working_messages (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    TEXT        NOT NULL,
    role       TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_working_messages_user_created
    ON working_messages (user_id, created_at DESC);
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PostgresConfig:
    """Connection and pool settings for the PostgreSQL backend."""

    dsn: str = field(default_factory=lambda: os.environ.get("OPENCLAW_PG_DSN", ""))
    pool_min: int = 2
    pool_max: int = 10


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def get_pg_connection(dsn: str) -> psycopg.Connection[Any]:
    """
    Open and return a psycopg3 connection.

    The import is deferred so that psycopg is not required for projects
    that only use the SQLite backend.

    Raises:
        ImportError: if psycopg is not installed.
        psycopg.OperationalError: if the connection cannot be established.
    """
    try:
        import psycopg  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "psycopg (psycopg3) is required for PostgreSQL support. "
            "Install it with: pip install psycopg[binary]"
        ) from exc

    return psycopg.connect(dsn)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def _load_migration_sql() -> str:
    """Return the migration SQL, preferring the .sql file on disk."""
    if _MIGRATION_SQL.exists():
        return _MIGRATION_SQL.read_text(encoding="utf-8")
    return _INLINE_SQL


def ensure_pg_schema(conn: psycopg.Connection[Any]) -> None:
    """
    Idempotently create all tables, indexes and the pgvector extension.

    This executes ``migrations/001_initial_schema.sql`` (or the embedded
    inline copy) inside a single transaction.  All DDL statements use
    ``IF NOT EXISTS`` / ``IF NOT EXISTS`` guards so the function is safe
    to call on every application startup.

    Args:
        conn: An open psycopg3 connection.  Auto-commit must be enabled
              for ``CREATE EXTENSION`` to work, or the caller must handle
              that themselves.  This function temporarily enables
              autocommit, applies the schema, then restores the prior
              state.
    """
    sql = _load_migration_sql()

    # psycopg3: autocommit must be True to run CREATE EXTENSION / CREATE INDEX
    # CONCURRENTLY outside a transaction block.  We toggle it for the duration.
    prior_autocommit: bool = conn.autocommit  # type: ignore[attr-defined]
    try:
        conn.autocommit = True  # type: ignore[attr-defined]
        with conn.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(sql)  # type: ignore[arg-type]
    finally:
        conn.autocommit = prior_autocommit  # type: ignore[attr-defined]
