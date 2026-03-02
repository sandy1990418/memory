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
    import psycopg

# Path to the bundled SQL migration files.
_MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"
_MIGRATION_SQL = _MIGRATIONS_DIR / "001_initial_schema.sql"
_MIGRATION_002_SQL = _MIGRATIONS_DIR / "002_consolidation.sql"

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

CREATE TABLE IF NOT EXISTS canonical_memories (
    id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               TEXT        NOT NULL,
    memory_key            TEXT        NOT NULL,
    value                 TEXT        NOT NULL,
    memory_type           TEXT        NOT NULL,
    confidence            FLOAT       NOT NULL DEFAULT 0.8,
    event_time            TIMESTAMPTZ,
    status                TEXT        NOT NULL DEFAULT 'active',
    supersedes_memory_id  UUID        REFERENCES canonical_memories(id),
    embedding             vector(1536),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata              JSONB       NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_active_key
    ON canonical_memories (user_id, memory_key) WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_canonical_user_id
    ON canonical_memories (user_id);

CREATE INDEX IF NOT EXISTS idx_canonical_embedding_hnsw
    ON canonical_memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS memory_update_queue (
    id            BIGSERIAL   PRIMARY KEY,
    user_id       TEXT        NOT NULL,
    payload       JSONB       NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'pending',
    attempts      INTEGER     NOT NULL DEFAULT 0,
    last_error    TEXT,
    available_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_update_queue_status_time
    ON memory_update_queue (status, available_at);

CREATE INDEX IF NOT EXISTS idx_memory_update_queue_user_status
    ON memory_update_queue (user_id, status);
"""

_INLINE_SQL_002 = """
CREATE TABLE IF NOT EXISTS short_term_buffer (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT        NOT NULL,
    thread_id   TEXT,
    messages    JSONB       NOT NULL DEFAULT '[]',
    topic_summary TEXT,
    token_count INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_stb_user_id
    ON short_term_buffer (user_id);

CREATE INDEX IF NOT EXISTS idx_stb_user_updated
    ON short_term_buffer (user_id, updated_at DESC);

ALTER TABLE canonical_memories
    ADD COLUMN IF NOT EXISTS consolidated_from UUID[] DEFAULT '{}';

ALTER TABLE canonical_memories
    ADD COLUMN IF NOT EXISTS consolidation_round INTEGER NOT NULL DEFAULT 0;

ALTER TABLE canonical_memories
    ADD COLUMN IF NOT EXISTS last_consolidated_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS consolidation_log (
    id                  BIGSERIAL   PRIMARY KEY,
    user_id             TEXT        NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    memories_scanned    INTEGER     NOT NULL DEFAULT 0,
    memories_merged     INTEGER     NOT NULL DEFAULT 0,
    memories_deleted    INTEGER     NOT NULL DEFAULT 0,
    memories_abstracted INTEGER     NOT NULL DEFAULT 0,
    status              TEXT        NOT NULL DEFAULT 'running',
    error               TEXT,
    metadata            JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_consolidation_log_user_started
    ON consolidation_log (user_id, started_at DESC);
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
        import psycopg  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "psycopg (psycopg3) is required for PostgreSQL support. "
            "Install it with: pip install psycopg[binary]"
        ) from exc

    return psycopg.connect(dsn)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def _load_migration_sql(migration_path: Path, inline_fallback: str) -> str:
    """Return the migration SQL, preferring the .sql file on disk."""
    if migration_path.exists():
        return migration_path.read_text(encoding="utf-8")
    return inline_fallback


def ensure_pg_schema(conn: psycopg.Connection[Any]) -> None:
    """
    Idempotently apply all schema migrations in order.

    Executes ``migrations/001_initial_schema.sql`` and
    ``migrations/002_consolidation.sql`` (or their embedded inline copies).
    All DDL statements use ``IF NOT EXISTS`` guards so the function is safe
    to call on every application startup.

    Args:
        conn: An open psycopg3 connection.  Auto-commit must be enabled
              for ``CREATE EXTENSION`` to work, or the caller must handle
              that themselves.  This function temporarily enables
              autocommit, applies the schema, then restores the prior
              state.
    """
    migrations = [
        (_MIGRATION_SQL, _INLINE_SQL),
        (_MIGRATION_002_SQL, _INLINE_SQL_002),
    ]

    # psycopg3: autocommit must be True to run CREATE EXTENSION / CREATE INDEX
    # CONCURRENTLY outside a transaction block.  We toggle it for the duration.
    prior_autocommit: bool = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for path, fallback in migrations:
                sql = _load_migration_sql(path, fallback)
                cur.execute(sql)
    finally:
        conn.autocommit = prior_autocommit
