"""Config resolution for memory search (mirrors agents/memory-search.ts)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class RemoteBatchConfig:
    enabled: bool = False
    wait: bool = True
    concurrency: int = 2
    poll_interval_ms: int = 5_000
    timeout_minutes: int = 30


@dataclass
class RemoteConfig:
    base_url: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    batch: RemoteBatchConfig = field(default_factory=RemoteBatchConfig)


@dataclass
class StoreVectorConfig:
    enabled: bool = True
    extension_path: str | None = None


@dataclass
class StoreConfig:
    driver: str = "sqlite"
    path: str = ""  # resolved at runtime
    vector: StoreVectorConfig = field(default_factory=StoreVectorConfig)


@dataclass
class ChunkingConfig:
    tokens: int = 400
    overlap: int = 80


@dataclass
class SessionSyncConfig:
    delta_bytes: int = 100_000
    delta_messages: int = 50


@dataclass
class SyncConfig:
    on_session_start: bool = True
    on_search: bool = True
    watch: bool = True
    watch_debounce_ms: int = 1_500
    interval_minutes: float = 0.0
    sessions: SessionSyncConfig = field(default_factory=SessionSyncConfig)


@dataclass
class MMRConfig:
    enabled: bool = False
    lambda_: float = 0.7


@dataclass
class TemporalDecayConfig:
    enabled: bool = False
    half_life_days: float = 30.0


@dataclass
class HybridConfig:
    enabled: bool = True
    vector_weight: float = 0.7
    text_weight: float = 0.3
    candidate_multiplier: float = 4.0
    mmr: MMRConfig = field(default_factory=MMRConfig)
    temporal_decay: TemporalDecayConfig = field(default_factory=TemporalDecayConfig)


@dataclass
class QueryConfig:
    max_results: int = 6
    min_score: float = 0.35
    hybrid: HybridConfig = field(default_factory=HybridConfig)


@dataclass
class CacheConfig:
    enabled: bool = True
    max_entries: int | None = None


@dataclass
class LocalConfig:
    model_path: str | None = None
    model_cache_dir: str | None = None


@dataclass
class ExperimentalConfig:
    session_memory: bool = False


@dataclass
class PostgresConfig:
    """Connection and pool settings for the PostgreSQL backend."""

    dsn: str = ""
    pool_min: int = 2
    pool_max: int = 10


# ---------------------------------------------------------------------------
# Top-level resolved config
# ---------------------------------------------------------------------------


@dataclass
class ResolvedMemorySearchConfig:
    enabled: bool
    sources: list[str]  # ["memory"] or ["memory", "sessions"]
    extra_paths: list[str]
    provider: str  # "openai" | "local" | "gemini" | "voyage" | "auto"
    fallback: str  # "openai" | "gemini" | "local" | "voyage" | "none"
    model: str
    remote: RemoteConfig
    local: LocalConfig
    store: StoreConfig
    chunking: ChunkingConfig
    sync: SyncConfig
    query: QueryConfig
    cache: CacheConfig
    experimental: ExperimentalConfig


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
_DEFAULT_VOYAGE_MODEL = "voyage-4-large"

_STATE_DIR_DEFAULT = os.path.join(os.path.expanduser("~"), ".openclaw")


def _resolve_state_dir() -> str:
    return os.environ.get("OPENCLAW_STATE_DIR", _STATE_DIR_DEFAULT)


def _resolve_db_path(agent_id: str) -> str:
    """Default DB path: ~/.openclaw/memory/<agentId>.sqlite"""
    env = os.environ.get("OPENCLAW_DB_PATH")
    if env:
        return env.replace("{agentId}", agent_id)
    return os.path.join(_resolve_state_dir(), "memory", f"{agent_id}.sqlite")


def _resolve_provider() -> str:
    env = os.environ.get("OPENCLAW_MEMORY_PROVIDER", "").strip().lower()
    if env in ("openai", "gemini", "voyage", "local", "auto"):
        return env
    return "auto"


def _resolve_model(provider: str) -> str:
    env = os.environ.get("OPENCLAW_MEMORY_MODEL", "").strip()
    if env:
        return env
    if provider == "gemini":
        return _DEFAULT_GEMINI_MODEL
    if provider == "voyage":
        return _DEFAULT_VOYAGE_MODEL
    return _DEFAULT_OPENAI_MODEL


def _resolve_fallback() -> str:
    env = os.environ.get("OPENCLAW_MEMORY_FALLBACK", "").strip().lower()
    if env in ("openai", "gemini", "local", "voyage", "none"):
        return env
    return "none"


def _resolve_remote() -> RemoteConfig:
    return RemoteConfig(
        base_url=os.environ.get("OPENCLAW_MEMORY_REMOTE_BASE_URL") or None,
        api_key=os.environ.get("OPENCLAW_MEMORY_REMOTE_API_KEY") or None,
    )


def resolve_memory_search_config(
    agent_id: str = "main",
    workspace_dir: str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> ResolvedMemorySearchConfig:
    """
    Resolve memory search config from environment variables and optional overrides dict.

    Environment variables:
        OPENCLAW_STATE_DIR       — base state directory (default: ~/.openclaw)
        OPENCLAW_DB_PATH         — SQLite DB path (supports {agentId} token)
        OPENCLAW_WORKSPACE       — workspace directory
        OPENCLAW_MEMORY_PROVIDER — embedding provider (auto/openai/gemini/voyage/local)
        OPENCLAW_MEMORY_MODEL    — embedding model name
        OPENCLAW_MEMORY_FALLBACK — fallback provider
        OPENCLAW_MEMORY_REMOTE_BASE_URL — remote API base URL
        OPENCLAW_MEMORY_REMOTE_API_KEY  — remote API key override
        OPENAI_API_KEY           — used when provider=openai
        GEMINI_API_KEY           — used when provider=gemini
        VOYAGE_API_KEY           — used when provider=voyage
    """
    overrides = overrides or {}

    provider = overrides.get("provider") or _resolve_provider()
    model = overrides.get("model") or _resolve_model(provider)
    fallback = overrides.get("fallback") or _resolve_fallback()

    db_path = overrides.get("db_path") or _resolve_db_path(agent_id)

    return ResolvedMemorySearchConfig(
        enabled=True,
        sources=overrides.get("sources", ["memory"]),
        extra_paths=overrides.get("extra_paths", []),
        provider=provider,
        fallback=fallback,
        model=model,
        remote=overrides.get("remote") or _resolve_remote(),
        local=LocalConfig(
            model_path=overrides.get("local_model_path"),
            model_cache_dir=overrides.get("local_model_cache_dir"),
        ),
        store=StoreConfig(
            path=db_path,
            vector=StoreVectorConfig(
                enabled=overrides.get("vector_enabled", True),
                extension_path=overrides.get("vector_extension_path"),
            ),
        ),
        chunking=ChunkingConfig(
            tokens=overrides.get("chunk_tokens", 400),
            overlap=overrides.get("chunk_overlap", 80),
        ),
        sync=SyncConfig(),
        query=QueryConfig(
            max_results=overrides.get("max_results", 6),
            min_score=overrides.get("min_score", 0.35),
            hybrid=HybridConfig(
                enabled=overrides.get("hybrid_enabled", True),
                vector_weight=overrides.get("vector_weight", 0.7),
                text_weight=overrides.get("text_weight", 0.3),
                candidate_multiplier=overrides.get("candidate_multiplier", 4.0),
                mmr=MMRConfig(
                    enabled=overrides.get("mmr_enabled", False),
                    lambda_=overrides.get("mmr_lambda", 0.7),
                ),
                temporal_decay=TemporalDecayConfig(
                    enabled=overrides.get("temporal_decay_enabled", False),
                    half_life_days=overrides.get("temporal_decay_half_life_days", 30.0),
                ),
            ),
        ),
        cache=CacheConfig(
            enabled=overrides.get("cache_enabled", True),
            max_entries=overrides.get("cache_max_entries"),
        ),
        experimental=ExperimentalConfig(
            session_memory=overrides.get("session_memory", False),
        ),
    )
