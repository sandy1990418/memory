"""
MemoryService — unified orchestrator for the multi-layer memory system.

Layers:
  L1  Working memory  (Redis, per-session, TTL-based)
  L2  Episodic memory (PostgreSQL episodic_memories, time-decayed)
  L3  Semantic memory (PostgreSQL semantic_memories, evergreen)

Search pipeline (read path):
    query_expansion -> pg_search_vector + pg_search_keyword -> merge_hybrid_results
    -> apply_temporal_decay_by_created_at -> apply_mmr_to_hybrid_results
    -> optional llm_rerank

Write pipeline (write path):
    LightMem-style pre-processing (optional) -> extract_memories -> store

All operations are scoped by user_id — no cross-user data is ever returned.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .answer_contract import AnswerPayload, generate_answer
from .embeddings import EmbeddingProvider
from .extraction import ExtractedMemory, extract_memories
from .hybrid import merge_hybrid_results
from .lightmem import (
    DistillPrepConfig,
    build_compact_session_text,
    normalize_messages_use,
    prepare_messages_for_distill,
)
from .llm_rerank import llm_rerank
from .mmr import MMRConfig, apply_mmr_to_hybrid_results
from .pg_schema import get_pg_connection
from .pg_search import pg_search_keyword, pg_search_vector
from .query_expansion import extract_keywords
from .temporal_decay import TemporalDecayConfig, apply_temporal_decay_by_created_at
from .types import MemorySearchResult
from .working_memory import DBWorkingMemory, WorkingMemory

if TYPE_CHECKING:
    import psycopg

_PGVECTOR_DIMS = 1536

_VALID_RESOLVER_UPDATE_MODES = frozenset({"sync", "offline"})
_VALID_SAVE_SESSION_MODES = frozenset({"raw", "summary"})

_DEFAULT_MAX_DISTILL_TOKENS = 2200
_DEFAULT_TOPIC_TOKEN_THRESHOLD = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_dict_to_search_result(d: dict[str, Any]) -> MemorySearchResult:
    return MemorySearchResult(
        path=d.get("path", ""),
        start_line=d.get("start_line", 0),
        end_line=d.get("end_line", 0),
        score=d.get("score", 0.0),
        snippet=d.get("snippet", ""),
        source=d.get("source", "memory"),
    )


def _working_memory_to_results(
    messages: list[dict[str, Any]], user_id: str, thread_id: str | None = None
) -> list[dict[str, Any]]:
    """Convert raw working-memory messages to result dicts compatible with merge pipeline."""
    results: list[dict[str, Any]] = []
    scope = thread_id if thread_id else "db"
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        results.append(
            {
                "id": f"wm:{user_id}:{scope}:{i}",
                "path": f"working/{user_id}/{scope}/{i}",
                "start_line": i,
                "end_line": i,
                "source": "working_memory",
                "snippet": content,
                "vector_score": 1.0,
                "text_score": 1.0,
                "score": 1.0,
                "created_at": None,
            }
        )
    return results


def _coerce_pgvector_dims(
    embedding: list[float], expected_dims: int = _PGVECTOR_DIMS
) -> list[float]:
    if len(embedding) == expected_dims:
        return embedding
    if len(embedding) > expected_dims:
        return embedding[:expected_dims]
    return embedding + ([0.0] * (expected_dims - len(embedding)))


def _safe_int(value: Any, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(minimum, default)
    return max(minimum, parsed)


def _safe_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


# ---------------------------------------------------------------------------
# MemoryService
# ---------------------------------------------------------------------------


class MemoryService:
    """
    Unified orchestrator for L1/L2/L3 memory operations.

    Args:
        pg_dsn:                   PostgreSQL connection DSN.
        embedding_provider:       Embedding provider implementing EmbeddingProvider protocol.
        llm_fn:                   Optional LLM callable for extraction and re-ranking.
        redis_url:                Optional Redis URL for L1 working memory.
        working_memory_provider:  Optional DBWorkingMemory instance for DB-backed L1.
                                  When provided, it takes precedence over Redis for
                                  record_message() and search() DB working memory.
                                  If None and pg_dsn is set, a DBWorkingMemory is
                                  auto-created using pg_dsn.
    """

    def __init__(
        self,
        pg_dsn: str,
        embedding_provider: EmbeddingProvider,
        llm_fn: Callable[[str], str] | None = None,
        redis_url: str | None = None,
        working_memory_provider: DBWorkingMemory | None = None,
        *,
        enable_structured_distill: bool = False,
        enable_conflict_resolver: bool = False,
        enable_answer_contract: bool = False,
        enable_lightmem: bool = False,
        pre_compress: bool | None = None,
        messages_use: str | None = None,
        topic_segment: bool | None = None,
        max_distill_tokens: int = _DEFAULT_MAX_DISTILL_TOKENS,
        topic_token_threshold: int = _DEFAULT_TOPIC_TOKEN_THRESHOLD,
        distill_min_confidence: float = 0.0,
        resolver_update_mode: str = "sync",
        save_session_mode: str = "raw",
        session_summary_chars: int = 1800,
    ) -> None:
        self._pg_dsn = pg_dsn
        self._embedding_provider = embedding_provider
        self._llm_fn = llm_fn
        self._working_memory: WorkingMemory | None = (
            WorkingMemory(redis_url) if redis_url else None
        )

        # Feature flags
        self._enable_structured_distill = enable_structured_distill
        self._enable_conflict_resolver = enable_conflict_resolver
        self._enable_answer_contract = enable_answer_contract

        # LightMem-inspired toggles
        self._enable_lightmem = enable_lightmem
        self._lightmem_pre_compress = pre_compress if pre_compress is not None else enable_lightmem
        default_use = "user_only" if enable_lightmem else "all"
        self._lightmem_messages_use = normalize_messages_use(messages_use or default_use)
        self._lightmem_topic_segment = (
            topic_segment if topic_segment is not None else enable_lightmem
        )
        self._max_distill_tokens = _safe_int(
            max_distill_tokens,
            _DEFAULT_MAX_DISTILL_TOKENS,
            minimum=64,
        )
        self._topic_token_threshold = _safe_int(
            topic_token_threshold,
            _DEFAULT_TOPIC_TOKEN_THRESHOLD,
            minimum=64,
        )
        default_conf = 0.5 if enable_lightmem else 0.0
        self._distill_min_confidence = _safe_float(
            distill_min_confidence,
            default=default_conf,
            minimum=0.0,
            maximum=1.0,
        )

        update_mode = (resolver_update_mode or "sync").strip().lower()
        if update_mode not in _VALID_RESOLVER_UPDATE_MODES:
            update_mode = "sync"
        if enable_lightmem and update_mode == "sync":
            update_mode = "offline"
        self._resolver_update_mode = update_mode

        session_mode = (save_session_mode or "raw").strip().lower()
        if session_mode not in _VALID_SAVE_SESSION_MODES:
            session_mode = "raw"
        if enable_lightmem and session_mode == "raw":
            session_mode = "summary"
        self._save_session_mode = session_mode
        self._session_summary_chars = _safe_int(session_summary_chars, 1800, minimum=200)

        # DB-backed working memory: use explicit provider, or auto-create from DSN
        if working_memory_provider is not None:
            self._db_working_memory: DBWorkingMemory | None = working_memory_provider
        elif pg_dsn:
            self._db_working_memory = DBWorkingMemory(
                conn_factory=lambda: get_pg_connection(pg_dsn)
            )
        else:
            self._db_working_memory = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> psycopg.Connection[Any]:
        return get_pg_connection(self._pg_dsn)

    def _is_structured_distill_enabled(self) -> bool:
        return bool(getattr(self, "_enable_structured_distill", False))

    def _is_conflict_resolver_enabled(self) -> bool:
        return bool(getattr(self, "_enable_conflict_resolver", False))

    def _is_answer_contract_enabled(self) -> bool:
        return bool(getattr(self, "_enable_answer_contract", False))

    def _is_offline_resolver_enabled(self) -> bool:
        mode = str(getattr(self, "_resolver_update_mode", "sync")).strip().lower()
        return self._is_conflict_resolver_enabled() and mode == "offline"

    def _distill_prep_config(self) -> DistillPrepConfig:
        return DistillPrepConfig(
            pre_compress=bool(getattr(self, "_lightmem_pre_compress", False)),
            messages_use=normalize_messages_use(
                str(getattr(self, "_lightmem_messages_use", "all"))
            ),
            topic_segment=bool(getattr(self, "_lightmem_topic_segment", False)),
            max_input_tokens=_safe_int(
                getattr(self, "_max_distill_tokens", _DEFAULT_MAX_DISTILL_TOKENS),
                _DEFAULT_MAX_DISTILL_TOKENS,
                minimum=64,
            ),
            topic_token_threshold=_safe_int(
                getattr(self, "_topic_token_threshold", _DEFAULT_TOPIC_TOKEN_THRESHOLD),
                _DEFAULT_TOPIC_TOKEN_THRESHOLD,
                minimum=64,
            ),
        )

    def _prepare_conversation_for_extract(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        cfg = self._distill_prep_config()
        return prepare_messages_for_distill(conversation, config=cfg)

    def _table_for_memory(self, memory: ExtractedMemory) -> str:
        return "episodic_memories" if memory.memory_type == "event" else "semantic_memories"

    def _normalize_memories_for_basic_mode(self, memories: list[ExtractedMemory]) -> None:
        for mem in memories:
            mem.memory_key = ""
            mem.value = mem.content
            mem.event_time = None
            mem.source_refs = []

    def _store_memory(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        memory: ExtractedMemory,
        table: str = "episodic_memories",
    ) -> None:
        """Insert a single memory directly into the given table."""
        embedding = _coerce_pgvector_dims(self._embedding_provider.embed_query(memory.content))
        sql = f"""
            INSERT INTO {table} (user_id, content, embedding, memory_type, metadata)
            VALUES (%s, %s, %s::vector, %s, %s)
        """
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    user_id,
                    memory.content,
                    embedding_str,
                    memory.memory_type,
                    json.dumps(memory.metadata),
                ),
            )

    def _store_memory_with_resolver(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        memory: ExtractedMemory,
    ) -> None:
        """
        Store a memory using the conflict resolver pipeline (canonical_memories table).

        Falls back to direct storage into episodic/semantic tables if resolver
        cannot be used.
        """
        try:
            from .conflict_resolver import apply_resolution, resolve_conflict  # noqa: PLC0415

            resolution = resolve_conflict(
                conn, user_id, memory, self._embedding_provider
            )
            value_for_embedding = memory.value or memory.content
            embedding = _coerce_pgvector_dims(
                self._embedding_provider.embed_query(value_for_embedding)
            )
            apply_resolution(conn, user_id, resolution, embedding=embedding)
        except Exception:
            # Resolver unavailable or schema not ready; keep write path functional.
            table = self._table_for_memory(memory)
            self._store_memory(conn, user_id, memory, table=table)

    def _memory_to_queue_payload(self, memory: ExtractedMemory) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": memory.content,
            "memory_type": memory.memory_type,
            "confidence": memory.confidence,
            "metadata": memory.metadata,
            "memory_key": memory.memory_key,
            "value": memory.value,
            "event_time": memory.event_time,
            "source_refs": memory.source_refs,
        }
        return payload

    def _enqueue_memory_update(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        memory: ExtractedMemory,
    ) -> None:
        payload = self._memory_to_queue_payload(memory)
        sql = """
            INSERT INTO memory_update_queue (user_id, payload, status, attempts, updated_at)
            VALUES (%s, %s::jsonb, 'pending', 0, now())
        """
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, json.dumps(payload)))

    def _store_raw_episode(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        thread_id: str,
        content: str,
    ) -> None:
        """Store a raw conversation string as a single episodic memory row."""
        embedding = _coerce_pgvector_dims(self._embedding_provider.embed_query(content[:2000]))
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        sql = """
            INSERT INTO episodic_memories (user_id, thread_id, content, embedding, memory_type)
            VALUES (%s, %s, %s, %s::vector, 'session')
        """
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, thread_id, content, embedding_str))

    def _session_text_raw(self, conversation: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _session_text_for_storage(self, conversation: list[dict[str, Any]]) -> str:
        mode = str(getattr(self, "_save_session_mode", "raw")).strip().lower()
        if mode == "summary":
            compact = build_compact_session_text(
                conversation,
                config=self._distill_prep_config(),
                max_chars=_safe_int(getattr(self, "_session_summary_chars", 1800), 1800, 200),
            )
            if compact:
                return compact
        return self._session_text_raw(conversation)

    def _claim_pending_updates(
        self,
        conn: psycopg.Connection[Any],
        *,
        limit: int,
        user_id: str | None,
    ) -> list[tuple[int, str, Any, int]]:
        if user_id is None:
            sql = """
                WITH picked AS (
                    SELECT id
                    FROM memory_update_queue
                    WHERE status = 'pending'
                      AND available_at <= now()
                    ORDER BY id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE memory_update_queue q
                SET status = 'processing',
                    attempts = q.attempts + 1,
                    updated_at = now()
                FROM picked
                WHERE q.id = picked.id
                RETURNING q.id, q.user_id, q.payload, q.attempts
            """
            params: tuple[Any, ...] = (limit,)
        else:
            sql = """
                WITH picked AS (
                    SELECT id
                    FROM memory_update_queue
                    WHERE status = 'pending'
                      AND available_at <= now()
                      AND user_id = %s
                    ORDER BY id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE memory_update_queue q
                SET status = 'processing',
                    attempts = q.attempts + 1,
                    updated_at = now()
                FROM picked
                WHERE q.id = picked.id
                RETURNING q.id, q.user_id, q.payload, q.attempts
            """
            params = (user_id, limit)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            (int(row[0]), str(row[1]), row[2], int(row[3]))
            for row in rows
        ]

    def _payload_to_memory(self, payload: Any) -> ExtractedMemory:
        data: dict[str, Any]
        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str):
            parsed = json.loads(payload)
            data = parsed if isinstance(parsed, dict) else {}
        else:
            data = {}

        content = str(data.get("content", "")).strip()
        memory_type = str(data.get("memory_type", "fact")).strip() or "fact"
        confidence = _safe_float(data.get("confidence", 0.8), 0.8, 0.0, 1.0)

        metadata_raw = data.get("metadata", {})
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

        mem = ExtractedMemory(
            content=content,
            memory_type=memory_type,
            confidence=confidence,
            metadata=metadata,
        )
        mem.memory_key = str(data.get("memory_key", "")).strip()
        value = str(data.get("value", "")).strip()
        mem.value = value if value else content

        event_time_raw = data.get("event_time")
        mem.event_time = str(event_time_raw).strip() if event_time_raw else None

        refs_raw = data.get("source_refs", [])
        mem.source_refs = [str(r) for r in refs_raw] if isinstance(refs_raw, list) else []
        return mem

    def _mark_queue_done(self, conn: psycopg.Connection[Any], queue_id: int) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_update_queue
                SET status = 'done',
                    last_error = NULL,
                    updated_at = now()
                WHERE id = %s
                """,
                (queue_id,),
            )

    def _mark_queue_error(
        self,
        conn: psycopg.Connection[Any],
        queue_id: int,
        *,
        err_text: str,
        retry: bool,
        backoff_minutes: int,
    ) -> None:
        clipped = err_text[:500]
        with conn.cursor() as cur:
            if retry:
                cur.execute(
                    """
                    UPDATE memory_update_queue
                    SET status = 'pending',
                        last_error = %s,
                        available_at = now() + (%s || ' minutes')::interval,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (clipped, backoff_minutes, queue_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE memory_update_queue
                    SET status = 'failed',
                        last_error = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (clipped, queue_id),
                )

    # ------------------------------------------------------------------
    # Working memory write path
    # ------------------------------------------------------------------

    def record_message(self, user_id: str, role: str, content: str) -> None:
        """
        Append a single message to DB-backed working memory for *user_id*.

        Requires DB working memory to be configured (auto-created from pg_dsn
        or injected via working_memory_provider).  No-ops silently if DB
        working memory is not available.

        Args:
            user_id: Tenant identifier.
            role:    Message role (e.g. ``"user"``, ``"assistant"``).
            content: Message text content.
        """
        if self._db_working_memory is None:
            return
        self._db_working_memory.append(user_id, role, content)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search(
        self,
        user_id: str,
        query: str,
        *,
        max_results: int = 6,
        include_working_memory: bool = True,
        enable_llm_rerank: bool = False,
        thread_id: str | None = None,
        temporal_decay_config: TemporalDecayConfig | None = None,
        mmr_config: MMRConfig | None = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        working_memory_limit: int = 20,
    ) -> list[MemorySearchResult]:
        """
        Unified search across all memory layers.

        Pipeline:
          1. L1 working memory (Redis or DB, optional)
          2. L2 episodic + L3 semantic (PostgreSQL)
          3. merge_hybrid_results
          4. temporal decay (L2 only)
          5. MMR re-ranking
          6. optional LLM re-ranking

        Args:
            user_id:               Tenant scoping — mandatory.
            query:                 Natural language search query.
            max_results:           Maximum number of results to return.
            include_working_memory: Whether to prepend L1 (Redis or DB) results.
            enable_llm_rerank:     Whether to apply LLM re-ranking step.
            thread_id:             Optional thread for Redis working-memory lookup.
            temporal_decay_config: Override temporal decay settings.
            mmr_config:            Override MMR settings.
            vector_weight:         Weight for vector scores in hybrid merge.
            text_weight:           Weight for keyword scores in hybrid merge.
            working_memory_limit:  Max messages to retrieve from DB working memory
                                   (default 20).
        """
        # --- Query expansion ---
        keywords = extract_keywords(query)
        keyword_query = " ".join(keywords) if keywords else query

        # --- Embedding ---
        query_vec = self._embedding_provider.embed_query(query)

        # --- PostgreSQL search (L2 + L3) ---
        canonical_vec: list[dict[str, Any]] = []
        canonical_kw: list[dict[str, Any]] = []

        conn = self._get_conn()
        try:
            episodic_vec = pg_search_vector(
                conn, user_id, query_vec, limit=20, table="episodic_memories"
            )
            episodic_kw = pg_search_keyword(
                conn, user_id, keyword_query, limit=20, table="episodic_memories"
            )
            semantic_vec = pg_search_vector(
                conn, user_id, query_vec, limit=20, table="semantic_memories"
            )
            semantic_kw = pg_search_keyword(
                conn, user_id, keyword_query, limit=20, table="semantic_memories"
            )
            if self._is_conflict_resolver_enabled() and not self._is_offline_resolver_enabled():
                canonical_vec = pg_search_vector(
                    conn, user_id, query_vec, limit=20, table="canonical_memories"
                )
                canonical_kw = pg_search_keyword(
                    conn, user_id, keyword_query, limit=20, table="canonical_memories"
                )
        finally:
            conn.close()

        # Combine all vector/keyword results
        all_vector = episodic_vec + semantic_vec + canonical_vec
        all_keyword = episodic_kw + semantic_kw + canonical_kw

        # --- Hybrid merge ---
        merged = merge_hybrid_results(
            vector=all_vector,
            keyword=all_keyword,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )

        # --- Temporal decay on episodic results ---
        # Re-attach created_at for decay: build lookup from original search results
        created_at_map: dict[str, Any] = {}
        for r in episodic_vec + episodic_kw:
            if "created_at" in r:
                created_at_map[r["path"]] = r["created_at"]

        merged_with_ts = []
        for r in merged:
            ca = created_at_map.get(r["path"])
            merged_with_ts.append({**r, "created_at": ca})

        decay_cfg = temporal_decay_config or TemporalDecayConfig(
            enabled=False, half_life_days=30.0
        )
        decayed = apply_temporal_decay_by_created_at(merged_with_ts, config=decay_cfg)

        # --- MMR ---
        mmr_cfg = mmr_config or MMRConfig(enabled=True, lambda_=0.7)
        reranked = apply_mmr_to_hybrid_results(decayed, mmr_cfg)

        # --- Optional LLM re-ranking ---
        if enable_llm_rerank and self._llm_fn is not None:
            reranked = llm_rerank(query, reranked, self._llm_fn, top_k=max_results)

        # --- Prepend L1 working memory ---
        # Priority: Redis (thread-scoped) > DB (user-scoped)
        wm_results: list[dict[str, Any]] = []
        if include_working_memory:
            if self._working_memory is not None and thread_id:
                # Redis-backed working memory (thread-scoped)
                messages = self._working_memory.get(user_id, thread_id)
                if messages:
                    wm_results = _working_memory_to_results(messages, user_id, thread_id)
            elif self._db_working_memory is not None:
                # DB-backed working memory (user-scoped, no thread_id required)
                messages = self._db_working_memory.get_recent(
                    user_id, limit=working_memory_limit
                )
                if messages:
                    wm_results = _working_memory_to_results(messages, user_id)

        # Working memory results appear first (highest priority = most recent context)
        combined = wm_results + reranked

        # Slice to max_results
        final = combined[:max_results]

        return [_result_dict_to_search_result(r) for r in final]

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def ingest_conversation(
        self,
        user_id: str,
        thread_id: str,
        conversation: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Full write pipeline: extract -> classify -> store.

        Returns a summary dict with counts of stored memories.
        """
        if not conversation:
            return {"inserted": 0, "skipped": 0}

        if self._llm_fn is None:
            return {"inserted": 0, "skipped": 0, "error": "llm_fn not configured"}

        prepared_conversation = self._prepare_conversation_for_extract(conversation)
        if not prepared_conversation:
            return {"inserted": 0, "skipped": len(conversation)}

        memories = extract_memories(prepared_conversation, self._llm_fn)
        if not self._is_structured_distill_enabled():
            self._normalize_memories_for_basic_mode(memories)

        inserted = 0
        skipped = 0
        min_conf = _safe_float(getattr(self, "_distill_min_confidence", 0.0), 0.0, 0.0, 1.0)
        use_resolver = (
            self._is_structured_distill_enabled()
            and self._is_conflict_resolver_enabled()
        )
        offline_resolver = use_resolver and self._is_offline_resolver_enabled()

        conn = self._get_conn()
        try:
            conn.autocommit = False
            for mem in memories:
                if mem.confidence < min_conf:
                    skipped += 1
                    continue

                if use_resolver and offline_resolver:
                    table = self._table_for_memory(mem)
                    self._store_memory(conn, user_id, mem, table=table)
                    self._enqueue_memory_update(conn, user_id, mem)
                elif use_resolver:
                    self._store_memory_with_resolver(conn, user_id, mem)
                else:
                    table = self._table_for_memory(mem)
                    self._store_memory(conn, user_id, mem, table=table)
                inserted += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        # Update working memory
        if self._working_memory is not None:
            self._working_memory.set(user_id, thread_id, conversation)

        return {"inserted": inserted, "skipped": skipped}

    def save_session(
        self,
        user_id: str,
        thread_id: str,
        conversation: list[dict[str, Any]],
    ) -> None:
        """
        Called on session end.

        Depending on ``save_session_mode`` this stores either the raw conversation
        or a compact summary in episodic memory, then clears working memory.
        """
        if not conversation:
            return

        content = self._session_text_for_storage(conversation)

        conn = self._get_conn()
        try:
            conn.autocommit = False
            self._store_raw_episode(conn, user_id, thread_id, content)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        # Clear working memory after session is saved
        if self._working_memory is not None:
            self._working_memory.delete(user_id, thread_id)

    def memory_flush(
        self,
        user_id: str,
        conversation: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Called when context is near-full — extract durable memories and store.

        Similar to ingest_conversation but without a thread_id (no working-memory update).
        """
        if not conversation:
            return {"inserted": 0, "skipped": 0}

        if self._llm_fn is None:
            return {"inserted": 0, "skipped": 0, "error": "llm_fn not configured"}

        prepared_conversation = self._prepare_conversation_for_extract(conversation)
        if not prepared_conversation:
            return {"inserted": 0, "skipped": len(conversation)}

        memories = extract_memories(prepared_conversation, self._llm_fn)
        if not self._is_structured_distill_enabled():
            self._normalize_memories_for_basic_mode(memories)

        inserted = 0
        skipped = 0
        min_conf = _safe_float(getattr(self, "_distill_min_confidence", 0.0), 0.0, 0.0, 1.0)
        use_resolver = (
            self._is_structured_distill_enabled()
            and self._is_conflict_resolver_enabled()
        )
        offline_resolver = use_resolver and self._is_offline_resolver_enabled()

        conn = self._get_conn()
        try:
            conn.autocommit = False
            for mem in memories:
                if mem.confidence < min_conf:
                    skipped += 1
                    continue

                if use_resolver and offline_resolver:
                    table = self._table_for_memory(mem)
                    self._store_memory(conn, user_id, mem, table=table)
                    self._enqueue_memory_update(conn, user_id, mem)
                elif use_resolver:
                    self._store_memory_with_resolver(conn, user_id, mem)
                else:
                    table = self._table_for_memory(mem)
                    self._store_memory(conn, user_id, mem, table=table)
                inserted += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return {"inserted": inserted, "skipped": skipped}

    def drain_update_queue(
        self,
        *,
        limit: int = 100,
        user_id: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, int]:
        """
        Process queued canonical-memory updates in a background/sleep-time cycle.

        This method is intentionally decoupled from user request paths.
        """
        if not self._is_conflict_resolver_enabled():
            return {"claimed": 0, "processed": 0, "retried": 0, "failed": 0}

        claim_limit = _safe_int(limit, 100, minimum=1)
        max_attempts_safe = _safe_int(max_attempts, 3, minimum=1)

        claim_conn = self._get_conn()
        try:
            claim_conn.autocommit = False
            claimed = self._claim_pending_updates(
                claim_conn,
                limit=claim_limit,
                user_id=user_id,
            )
            claim_conn.commit()
        except Exception:
            claim_conn.rollback()
            raise
        finally:
            claim_conn.close()

        processed = 0
        retried = 0
        failed = 0

        from .conflict_resolver import apply_resolution, resolve_conflict  # noqa: PLC0415

        for queue_id, queue_user_id, payload, attempts in claimed:
            item_conn = self._get_conn()
            try:
                item_conn.autocommit = False
                memory = self._payload_to_memory(payload)
                if not memory.content:
                    raise ValueError("empty queued memory payload")
                resolution = resolve_conflict(
                    item_conn,
                    queue_user_id,
                    memory,
                    self._embedding_provider,
                )
                value = memory.value or memory.content
                embedding = _coerce_pgvector_dims(self._embedding_provider.embed_query(value))
                apply_resolution(item_conn, queue_user_id, resolution, embedding=embedding)
                self._mark_queue_done(item_conn, queue_id)
                item_conn.commit()
                processed += 1
            except Exception as exc:
                item_conn.rollback()
                retry = attempts < max_attempts_safe
                backoff = min(60, 2 ** max(1, attempts))
                try:
                    item_conn.autocommit = False
                    self._mark_queue_error(
                        item_conn,
                        queue_id,
                        err_text=str(exc),
                        retry=retry,
                        backoff_minutes=backoff,
                    )
                    item_conn.commit()
                except Exception:
                    item_conn.rollback()
                if retry:
                    retried += 1
                else:
                    failed += 1
            finally:
                item_conn.close()

        return {
            "claimed": len(claimed),
            "processed": processed,
            "retried": retried,
            "failed": failed,
        }

    # ------------------------------------------------------------------
    # Answer (schema-constrained LLM response)
    # ------------------------------------------------------------------

    def answer(
        self,
        user_id: str,
        query: str,
        **search_kwargs: Any,
    ) -> AnswerPayload:
        """
        Search -> evidence package -> LLM answer (schema constrained) -> validation.

        Args:
            user_id:        Tenant identifier.
            query:          The user's natural-language question.
            **search_kwargs: Forwarded to self.search() (e.g. max_results, thread_id).

        Returns:
            AnswerPayload — abstains if no LLM is configured or search returns nothing.
        """
        results = self.search(user_id, query, **search_kwargs)

        # Legacy answer mode: return top snippet without schema-constrained generation.
        if not self._is_answer_contract_enabled():
            if not results:
                return AnswerPayload(
                    answer="",
                    evidence=[],
                    confidence=0.0,
                    abstain=True,
                    abstain_reason="No relevant memory found",
                )
            top = results[0]
            return AnswerPayload(
                answer=top.snippet,
                evidence=[],
                confidence=max(0.0, min(1.0, top.score)),
                abstain=False,
                abstain_reason="",
            )

        if self._llm_fn is None:
            return AnswerPayload(
                answer="",
                evidence=[],
                confidence=0.0,
                abstain=True,
                abstain_reason="LLM not configured",
            )

        # Convert MemorySearchResult objects to evidence chunk dicts
        evidence_chunks = [
            {
                "path": r.path,
                "id": r.path,
                "snippet": r.snippet,
                "score": r.score,
            }
            for r in results
        ]

        return generate_answer(query, evidence_chunks, self._llm_fn)

    # ------------------------------------------------------------------
    # Profile CRUD
    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """
        Retrieve user profile from user_profiles table.

        Returns an empty dict if no profile exists yet.
        """
        sql = "SELECT profile FROM user_profiles WHERE user_id = %s"
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return {}
        profile = row[0]
        if isinstance(profile, str):
            try:
                profile = json.loads(profile)
            except (json.JSONDecodeError, TypeError):
                profile = {}
        return profile if isinstance(profile, dict) else {}

    def update_user_profile(self, user_id: str, updates: dict[str, Any]) -> None:
        """
        Upsert user profile fields.

        Uses PostgreSQL jsonb merge (||) to merge updates into existing profile.
        """
        sql = """
            INSERT INTO user_profiles (user_id, profile, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (user_id)
            DO UPDATE SET
                profile    = user_profiles.profile || EXCLUDED.profile,
                updated_at = now()
        """
        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, json.dumps(updates)))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
