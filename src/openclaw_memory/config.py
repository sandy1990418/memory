"""
Application configuration via Pydantic Settings.

All settings can be overridden via environment variables with the
OPENCLAW_ prefix (e.g. OPENCLAW_PG_DSN, OPENCLAW_EMBEDDING_PROVIDER).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Root configuration — loaded from environment variables."""

    model_config = {"env_prefix": "OPENCLAW_", "case_sensitive": False}

    # -- Database --
    pg_dsn: str = Field(
        default="",
        description="PostgreSQL connection DSN",
    )
    pg_pool_min: int = Field(default=2, ge=1)
    pg_pool_max: int = Field(default=10, ge=1)

    # -- Embedding --
    embedding_provider: str = Field(
        default="auto",
        description="Embedding provider: auto / openai / gemini / voyage",
    )
    embedding_model: str = Field(
        default="",
        description="Embedding model name (auto-resolved if empty)",
    )
    embedding_dimensions: int = Field(
        default=1536, ge=64, le=4096,
        description="Embedding vector dimensions. Must match your model output "
        "(e.g. 1536 for text-embedding-3-small, 768 for nomic-embed). "
        "Changing this requires a schema migration.",
    )

    # -- LLM --
    # Each operation can use a different LLM model. If a per-operation model
    # is empty, it falls back to llm_model. Set via env vars like
    # OPENCLAW_EXTRACTION_LLM_MODEL, OPENCLAW_ANSWER_LLM_MODEL, etc.
    llm_provider: str = Field(
        default="auto",
        description="LLM provider: auto / openai / anthropic / gemini",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Default LLM model (fallback for all operations)",
    )
    extraction_llm_model: str = Field(
        default="",
        description="LLM model for fact extraction",
    )
    conflict_llm_model: str = Field(
        default="",
        description="LLM model for conflict resolution",
    )
    rerank_llm_model: str = Field(
        default="",
        description="LLM model for search re-ranking",
    )
    answer_llm_model: str = Field(
        default="",
        description="LLM model for answer generation",
    )
    consolidation_llm_model: str = Field(
        default="",
        description="LLM model for memory consolidation",
    )
    promotion_llm_model: str = Field(
        default="",
        description="LLM model for episodic-to-semantic promotion",
    )

    # -- Sensory pipeline (LightMem Stage 1) --
    sensory_pre_compress: bool = Field(
        default=True,
        description="Enable TF-IDF compression of verbose messages",
    )
    sensory_topic_segment: bool = Field(
        default=True,
        description="Enable topic-aware segmentation",
    )
    sensory_max_input_tokens: int = Field(default=4096, ge=64)
    sensory_per_message_char_limit: int = Field(default=320, ge=40)
    sensory_topic_token_threshold: int = Field(default=800, ge=64)

    # -- Extraction --
    extraction_min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence to store an extracted memory",
    )

    # -- Search --
    search_max_results: int = Field(default=10, ge=1)
    search_vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    search_text_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    search_temporal_decay_enabled: bool = Field(default=True)
    search_temporal_decay_half_life_days: float = Field(default=30.0, gt=0)
    search_mmr_enabled: bool = Field(default=True)
    search_mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)

    # -- Consolidation --
    consolidation_similarity_threshold: float = Field(
        default=0.90, ge=0.5, le=1.0,
        description="Cosine similarity threshold for clustering in consolidation",
    )
    consolidation_max_cluster_size: int = Field(default=10, ge=2)

    # -- Working memory --
    working_memory_max_messages: int = Field(default=20, ge=1)

    # -- Memory lifecycle --
    buffer_drain_every: int = Field(
        default=8, ge=2,
        description="Trigger mid-session extraction every N messages",
    )
    consolidation_trigger_threshold: int = Field(
        default=20, ge=5,
        description="Trigger consolidation after N new canonical memories since last run",
    )
    orphan_session_timeout_hours: float = Field(
        default=2.0, gt=0,
        description="Working messages older than N hours are considered orphaned",
    )
    superseded_cleanup_days: int = Field(
        default=30, ge=7,
        description="Physically delete superseded/deleted canonical memories after N days",
    )

    # -- Auth --
    api_key_required: bool = Field(
        default=False,
        description="Require X-API-Key header on all requests (except /health)",
    )

    # -- Rate limiting --
    rate_limit_enabled: bool = Field(default=False)
    rate_limit_rpm: int = Field(
        default=60, ge=1,
        description="Max requests per minute per client",
    )

    # -- Server --
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)


def get_settings() -> AppSettings:
    """Load settings from environment. Cached at module level."""
    return AppSettings()
