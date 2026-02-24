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
