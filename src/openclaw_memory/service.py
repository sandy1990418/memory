"""
MemoryService — unified orchestrator for the multi-layer memory system.

Layers:
  L1  Working memory  (Redis, per-session, TTL-based)
  L2  Episodic memory (PostgreSQL episodic_memories, time-decayed)
  L3  Semantic memory (PostgreSQL semantic_memories, evergreen)

Search pipeline (read path):
    query_expansion → pg_search_vector + pg_search_keyword → merge_hybrid_results
    → apply_temporal_decay_by_created_at → apply_mmr_to_hybrid_results
    → optional llm_rerank

Write pipeline (write path):
    extract_memories → store_with_dedup (per memory)

All operations are scoped by user_id — no cross-user data is ever returned.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .embeddings import EmbeddingProvider
from .extraction import ExtractedMemory, extract_memories
from .hybrid import merge_hybrid_results
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
    ) -> None:
        self._pg_dsn = pg_dsn
        self._embedding_provider = embedding_provider
        self._llm_fn = llm_fn
        self._working_memory: WorkingMemory | None = (
            WorkingMemory(redis_url) if redis_url else None
        )
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

    def _store_memory(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        memory: ExtractedMemory,
        table: str = "episodic_memories",
    ) -> None:
        """Insert a single memory directly into the given table."""
        embedding = self._embedding_provider.embed_query(memory.content)
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

    def _store_raw_episode(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        thread_id: str,
        content: str,
    ) -> None:
        """Store a raw conversation string as a single episodic memory row."""
        embedding = self._embedding_provider.embed_query(content[:2000])
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        sql = """
            INSERT INTO episodic_memories (user_id, thread_id, content, embedding, memory_type)
            VALUES (%s, %s, %s, %s::vector, 'session')
        """
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, thread_id, content, embedding_str))

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
        finally:
            conn.close()

        # Combine all vector/keyword results
        all_vector = episodic_vec + semantic_vec
        all_keyword = episodic_kw + semantic_kw

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
        Full write pipeline: extract → classify → store.

        Returns a summary dict with counts of stored memories.
        """
        if not conversation:
            return {"inserted": 0, "skipped": 0}

        if self._llm_fn is None:
            return {"inserted": 0, "skipped": 0, "error": "llm_fn not configured"}

        memories = extract_memories(conversation, self._llm_fn)

        inserted = 0
        conn = self._get_conn()
        try:
            conn.autocommit = False
            for mem in memories:
                # Classify: "fact"/"preference"/"decision" → semantic; "event" → episodic
                table = (
                    "episodic_memories"
                    if mem.memory_type == "event"
                    else "semantic_memories"
                )
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

        return {"inserted": inserted, "skipped": 0}

    def save_session(
        self,
        user_id: str,
        thread_id: str,
        conversation: list[dict[str, Any]],
    ) -> None:
        """
        Called on session end — saves raw conversation as an episodic memory.
        Also clears working memory for the thread.
        """
        if not conversation:
            return

        # Format conversation as a single text blob
        parts: list[str] = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        content = "\n".join(parts)

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
            return {"inserted": 0}

        if self._llm_fn is None:
            return {"inserted": 0, "error": "llm_fn not configured"}

        memories = extract_memories(conversation, self._llm_fn)

        inserted = 0
        conn = self._get_conn()
        try:
            conn.autocommit = False
            for mem in memories:
                table = (
                    "episodic_memories"
                    if mem.memory_type == "event"
                    else "semantic_memories"
                )
                self._store_memory(conn, user_id, mem, table=table)
                inserted += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return {"inserted": inserted}

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
