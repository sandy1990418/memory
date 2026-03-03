-- 002_consolidation.sql
-- Adds short-term buffer, consolidation tracking, and compact search support.
-- Requires: 001_initial_schema.sql applied first.

-- ---------------------------------------------------------------------------
-- short_term_buffer – topic-aware message accumulation before extraction
-- ---------------------------------------------------------------------------

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

-- ---------------------------------------------------------------------------
-- canonical_memories – consolidation tracking columns
-- ---------------------------------------------------------------------------

ALTER TABLE canonical_memories
    ADD COLUMN IF NOT EXISTS consolidated_from UUID[] DEFAULT '{}';

ALTER TABLE canonical_memories
    ADD COLUMN IF NOT EXISTS consolidation_round INTEGER NOT NULL DEFAULT 0;

ALTER TABLE canonical_memories
    ADD COLUMN IF NOT EXISTS last_consolidated_at TIMESTAMPTZ;

-- ---------------------------------------------------------------------------
-- consolidation_log – audit trail for sleep-time consolidation runs
-- ---------------------------------------------------------------------------

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
