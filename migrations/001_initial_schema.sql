-- 001_initial_schema.sql
-- PostgreSQL + pgvector schema for multi-tenant memory system.
-- Requires: PostgreSQL 15+, pgvector extension.

-- Enable pgvector extension (requires superuser or pg_extension_owner)
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- episodic_memories
-- ---------------------------------------------------------------------------

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

-- Backward compatibility: older DBs used thread_id instead of session_id.
ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS session_id TEXT;

ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- B-tree index for per-user queries (most common filter)
CREATE INDEX IF NOT EXISTS idx_episodic_user_id
    ON episodic_memories (user_id);

-- Composite index for temporal queries scoped to a user
CREATE INDEX IF NOT EXISTS idx_episodic_user_created
    ON episodic_memories (user_id, created_at DESC);

-- HNSW index for fast approximate nearest-neighbour on embedding
CREATE INDEX IF NOT EXISTS idx_episodic_embedding_hnsw
    ON episodic_memories USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_episodic_tsv
    ON episodic_memories USING gin (tsv);

-- ---------------------------------------------------------------------------
-- semantic_memories
-- ---------------------------------------------------------------------------

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

-- B-tree index for per-user queries
CREATE INDEX IF NOT EXISTS idx_semantic_user_id
    ON semantic_memories (user_id);

-- Composite index for temporal queries scoped to a user
CREATE INDEX IF NOT EXISTS idx_semantic_user_created
    ON semantic_memories (user_id, created_at DESC);

-- HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_semantic_embedding_hnsw
    ON semantic_memories USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_semantic_tsv
    ON semantic_memories USING gin (tsv);

-- ---------------------------------------------------------------------------
-- user_profiles
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id    TEXT        PRIMARY KEY,
    profile    JSONB       NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- B-tree index on user_id (already the PK, but explicit for consistency)
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id
    ON user_profiles (user_id);

-- ---------------------------------------------------------------------------
-- working_messages
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS working_messages (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT        NOT NULL,
    session_id  TEXT,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE working_messages
    ADD COLUMN IF NOT EXISTS session_id TEXT;

CREATE INDEX IF NOT EXISTS idx_working_messages_user_created
    ON working_messages (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_working_messages_user_session
    ON working_messages (user_id, session_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- canonical_memories
-- ---------------------------------------------------------------------------

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

-- Unique active key per user
CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_active_key
    ON canonical_memories (user_id, memory_key) WHERE status = 'active';

-- User lookup
CREATE INDEX IF NOT EXISTS idx_canonical_user_id
    ON canonical_memories (user_id);

-- HNSW for similarity fallback
CREATE INDEX IF NOT EXISTS idx_canonical_embedding_hnsw
    ON canonical_memories USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- memory_update_queue (sleep-time / offline canonical updates)
-- ---------------------------------------------------------------------------

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
