"""
PostgreSQL schema management and migrations.

All DDL uses IF NOT EXISTS guards so ensure_schema() is safe to call
on every application startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

# Path to bundled SQL migration files
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "migrations"
_MIGRATION_001 = _MIGRATIONS_DIR / "001_initial_schema.sql"
_MIGRATION_002 = _MIGRATIONS_DIR / "002_consolidation.sql"

# Inline fallback SQL (used when installed as a wheel without migrations/)
_INLINE_001 = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodic_memories (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT        NOT NULL,
    session_id  TEXT,
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

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id    TEXT        PRIMARY KEY,
    profile    JSONB       NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS working_messages (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    TEXT        NOT NULL,
    session_id TEXT,
    role       TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_working_messages_user_session
    ON working_messages (user_id, session_id, created_at DESC);

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
"""

_INLINE_002 = """
CREATE TABLE IF NOT EXISTS short_term_buffer (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT        NOT NULL,
    session_id  TEXT,
    messages    JSONB       NOT NULL DEFAULT '[]',
    topic_summary TEXT,
    token_count INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_stb_user_id
    ON short_term_buffer (user_id);

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

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS token_budget_daily INTEGER NOT NULL DEFAULT 100000;

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS tokens_used_today INTEGER NOT NULL DEFAULT 0;

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS budget_reset_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash    TEXT        PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    label       TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id
    ON api_keys (user_id);
"""


def _load_migration(path: Path, fallback: str) -> str:
    """Load migration SQL from file, falling back to inline."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def ensure_schema(conn: psycopg.Connection[Any]) -> None:
    """
    Idempotently apply all schema migrations.

    Safe to call on every application startup. Temporarily enables
    autocommit for CREATE EXTENSION support.
    """
    migrations = [
        (_MIGRATION_001, _INLINE_001),
        (_MIGRATION_002, _INLINE_002),
    ]

    prior_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for path, fallback in migrations:
                sql = _load_migration(path, fallback)
                cur.execute(sql)
    finally:
        conn.autocommit = prior_autocommit
