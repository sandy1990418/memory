"""
MemoryService — thin orchestrator for the three-layer memory system.

Coordinates the ingest and retrieval pipelines across:
  L1  Working memory  (session-scoped, PG-backed)
  L2  Episodic memory (time-decayed)
  L3  Semantic memory (consolidated, evergreen)

All operations require (user_id), session operations also require (session_id).
Connection management is external — callers provide a psycopg connection.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import psycopg

from .embeddings import EmbeddingProvider, coerce_pgvector_dims, embedding_to_pg_literal
from .types import ExtractedMemory, MemoryContext, MemoryIndex, MemorySearchResult
from ..config import AppSettings
from ..db import queries
from ..utils.llm import TokenTracker
from ..memory.episodic import store_episodic, store_session_episode
from ..memory.semantic import store_semantic, table_for_memory
from ..memory.working import WorkingMemory
from ..pipeline.ingest.conflict import apply_resolution, resolve_conflict
from ..pipeline.ingest.extraction import extract_memories
from ..pipeline.ingest.sensory import SensoryConfig, build_session_summary, prepare_for_extraction
from ..consolidation.consolidator import MemoryConsolidator
from ..consolidation.promotion import promote_events_to_semantic
from ..pipeline.retrieval.answer import AnswerPayload, generate_answer
from ..pipeline.retrieval.ranking import MMRConfig, TemporalDecayConfig
from ..pipeline.retrieval.search import (
    search as pipeline_search,
    search_compact as pipeline_search_compact,
    search_detail as pipeline_search_detail,
    search_timeline as pipeline_search_timeline,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Thin orchestrator for the multi-layer memory system.

    Does not manage connections — callers provide a psycopg connection.
    Does not toggle feature flags — sensory filtering and conflict resolution
    are always active.
    """

    # Operations that can each use a different LLM.
    LLM_OPERATIONS = (
        "extraction", "conflict", "rerank", "answer", "consolidation", "promotion",
    )

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        settings: AppSettings,
        llm_fn: Callable[[str], str] | None = None,
        *,
        llm_fns: dict[str, Callable[[str], str]] | None = None,
    ) -> None:
        self._emb = embedding_provider
        self._settings = settings
        # Per-operation LLM callables. Each key is an operation name
        # (extraction, conflict, rerank, answer, consolidation, promotion).
        # Falls back to the default llm_fn if a specific one is not provided.
        self._default_llm_fn = llm_fn
        self._llm_fns: dict[str, Callable[[str], str]] = dict(llm_fns or {})
        self._working = WorkingMemory(max_messages=settings.working_memory_max_messages)
        # Track how many messages have been drained per session (in-memory).
        # Key: (user_id, session_id) -> number of messages already extracted.
        # Lost on process restart — conflict resolution NOOP handles re-extraction.
        self._drain_offsets: dict[tuple[str, str], int] = {}

    def _get_llm(self, operation: str) -> Callable[[str], str] | None:
        """Get the LLM callable for a specific operation, falling back to default."""
        return self._llm_fns.get(operation) or self._default_llm_fn

    @property
    def sensory_config(self) -> SensoryConfig:
        s = self._settings
        return SensoryConfig(
            pre_compress=s.sensory_pre_compress,
            topic_segment=s.sensory_topic_segment,
            max_input_tokens=s.sensory_max_input_tokens,
            per_message_char_limit=s.sensory_per_message_char_limit,
            topic_token_threshold=s.sensory_topic_token_threshold,
        )

    @property
    def temporal_decay_config(self) -> TemporalDecayConfig:
        return TemporalDecayConfig(
            enabled=self._settings.search_temporal_decay_enabled,
            half_life_days=self._settings.search_temporal_decay_half_life_days,
        )

    @property
    def mmr_config(self) -> MMRConfig:
        return MMRConfig(
            enabled=self._settings.search_mmr_enabled,
            lambda_=self._settings.search_mmr_lambda,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
    ) -> None:
        """Initialize a session (clear stale working memory if any)."""
        self._working.clear_session(conn, user_id, session_id)
        # Piggyback: clean up orphaned sessions for this user
        timeout_hours = self._settings.orphan_session_timeout_hours
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM working_messages "
                "WHERE user_id = %s AND session_id != %s "
                "AND created_at < NOW() - make_interval(hours => %s)",
                (user_id, session_id, timeout_hours),
            )

    def record_message(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> dict[str, Any] | None:
        """
        Append a message to L1 working memory.

        Every ``buffer_drain_every`` messages, automatically triggers
        mid-session extraction so memories are persisted incrementally
        instead of only at session end.
        """
        self._working.append(conn, user_id, session_id, role, content)

        drain_every = self._settings.buffer_drain_every
        if drain_every <= 0:
            return None

        msg_count = self._working.count(conn, user_id, session_id)
        if msg_count > 0 and msg_count % drain_every == 0:
            return self._drain_buffer(conn, user_id, session_id)
        return None

    def end_session(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        End a session: extract remaining memories from working memory,
        save a session episode, clear working memory, and trigger
        background consolidation if needed.

        Only extracts messages that haven't been drained mid-session.
        """
        messages = self._working.get_recent(conn, user_id, session_id)
        key = (user_id, session_id)

        if not messages:
            self._working.clear_session(conn, user_id, session_id)
            self._drain_offsets.pop(key, None)
            return {"extracted": 0, "stored": 0}

        # Only extract messages not yet drained
        last_offset = self._drain_offsets.get(key, 0)
        remaining = messages[last_offset:]

        result: dict[str, Any] = {"extracted": 0, "stored": 0}
        if remaining:
            conversation = [{"role": m["role"], "content": m["content"]} for m in remaining]
            result = self._extract_and_store(conn, user_id, conversation, session_id=session_id)

        # Save full session summary as episodic memory
        all_conversation = [{"role": m["role"], "content": m["content"]} for m in messages]
        summary = build_session_summary(all_conversation, self.sensory_config)
        if summary:
            store_session_episode(conn, user_id, session_id, summary, self._emb)

        # Clear working memory and drain offset
        self._working.clear_session(conn, user_id, session_id)
        self._drain_offsets.pop(key, None)

        # Piggyback: consolidation check + lazy cleanup
        self._maybe_consolidate(conn, user_id)
        self._lazy_cleanup(conn, user_id)

        return result

    # ------------------------------------------------------------------
    # Write path (ingest pipeline)
    # ------------------------------------------------------------------

    def ingest_conversation(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        conversation: list[dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Full ingest pipeline: sensory filter -> LLM extract -> conflict resolve -> store.

        Returns {"extracted": N, "stored": N, "skipped": N}.
        """
        return self._extract_and_store(conn, user_id, conversation, session_id=session_id)

    def add_memory(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        content: str,
        *,
        memory_type: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Directly add a single memory without LLM extraction."""
        memory = ExtractedMemory(
            content=content,
            memory_type=memory_type,
            confidence=1.0,
            metadata=metadata or {},
        )
        return self._store_single(conn, user_id, memory, session_id=None)

    # ------------------------------------------------------------------
    # Read path (search pipeline)
    # ------------------------------------------------------------------

    def search(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        query: str,
        *,
        top_k: int | None = None,
        include_working: bool = False,
        session_id: str | None = None,
        tables: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """
        Quota-based layered search across L2 + L3 with hybrid merge and ranking.

        Optionally includes L1 working memory, merged by score.
        """
        max_results = top_k or self._settings.search_max_results

        results = pipeline_search(
            conn, user_id, query, self._emb,
            vector_weight=self._settings.search_vector_weight,
            text_weight=self._settings.search_text_weight,
            temporal_decay=self.temporal_decay_config,
            mmr=self.mmr_config,
            llm_fn=self._get_llm("rerank"),
            top_k=max_results,
            tables=tables,
            canonical_ratio=self._settings.search_canonical_ratio,
        )

        if include_working and session_id:
            wm_messages = self._working.get_recent(conn, user_id, session_id)
            wm_results = self._working.to_search_results(wm_messages, user_id, session_id)
            wm_typed = [
                MemorySearchResult(
                    id=r["id"],
                    content=r["snippet"],
                    memory_type="working",
                    score=r["score"],
                    created_at=str(r.get("created_at", "")),
                    source="working_memory",
                )
                for r in wm_results
            ]
            results = sorted(results + wm_typed, key=lambda r: r.score, reverse=True)

        return results[:max_results]

    def search_compact(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[MemoryIndex]:
        """Tier 1: Lightweight index entries (~50 tokens each)."""
        return pipeline_search_compact(conn, user_id, query, self._emb, limit=limit)

    def search_timeline(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        memory_id: str,
        *,
        depth_before: int = 3,
        depth_after: int = 3,
    ) -> MemoryContext | None:
        """Tier 2: Temporal context around a memory."""
        return pipeline_search_timeline(
            conn, user_id, memory_id,
            depth_before=depth_before, depth_after=depth_after,
        )

    def search_detail(
        self,
        conn: psycopg.Connection[Any],
        memory_ids: list[str],
    ) -> list[MemorySearchResult]:
        """Tier 3: Full content for selected IDs."""
        return pipeline_search_detail(conn, memory_ids)

    # ------------------------------------------------------------------
    # Answer (RAG pipeline)
    # ------------------------------------------------------------------

    def answer(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        query: str,
        *,
        top_k: int = 6,
        session_id: str | None = None,
        tables: list[str] | None = None,
    ) -> AnswerPayload:
        """Search + LLM answer generation with structured evidence."""
        if self._get_llm("answer") is None:
            return AnswerPayload(
                answer="", evidence=[], confidence=0.0,
                abstain=True, abstain_reason="LLM not configured",
            )

        # Budget check
        if not self._check_budget(conn, user_id):
            return AnswerPayload(
                answer="", evidence=[], confidence=0.0,
                abstain=True, abstain_reason="Daily token budget exceeded",
            )

        results = self.search(
            conn, user_id, query,
            top_k=top_k, include_working=bool(session_id), session_id=session_id,
            tables=tables,
        )
        if not results:
            return AnswerPayload(
                answer="", evidence=[], confidence=0.0,
                abstain=True, abstain_reason="No relevant memory found",
            )

        evidence_chunks = [
            {"id": r.id, "snippet": r.content, "score": r.score}
            for r in results
        ]
        tracker = self._tracked_llm("answer")
        result = generate_answer(query, evidence_chunks, tracker)
        self._record_usage(conn, user_id, tracker)
        return result

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def get_user_memories(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all memories for a user."""
        return queries.get_user_memories(conn, user_id, limit=limit, offset=offset)

    def delete_memory(
        self,
        conn: psycopg.Connection[Any],
        memory_id: str,
        user_id: str,
    ) -> bool:
        """Delete a memory by ID."""
        return queries.delete_memory(conn, memory_id, user_id)

    # ------------------------------------------------------------------
    # Token budget helpers
    # ------------------------------------------------------------------

    def _check_budget(self, conn: psycopg.Connection[Any], user_id: str) -> bool:
        """Return True if user has remaining token budget."""
        remaining, _ = queries.check_token_budget(conn, user_id)
        return remaining > 0

    def _record_usage(
        self, conn: psycopg.Connection[Any], user_id: str, tracker: TokenTracker,
    ) -> None:
        """Record token usage from a tracker to the user's profile."""
        if tracker.total_tokens > 0:
            queries.record_token_usage(conn, user_id, tracker.total_tokens)

    def _tracked_llm(self, operation: str) -> TokenTracker | None:
        """Return a TokenTracker wrapping the LLM for the given operation."""
        llm_fn = self._get_llm(operation)
        if llm_fn is None:
            return None
        # Resolve per-operation model name from settings
        model_attr = f"{operation}_llm_model"
        model = getattr(self._settings, model_attr, "") or self._settings.llm_model
        return TokenTracker(llm_fn, model=model, operation=operation)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_store(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        conversation: list[dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Core ingest pipeline: sensory -> extract -> conflict resolve -> store."""
        if not conversation or self._get_llm("extraction") is None:
            return {"extracted": 0, "stored": 0, "skipped": 0}

        # Budget check — skip LLM steps if over budget
        if not self._check_budget(conn, user_id):
            logger.info("Token budget exceeded for user %s, skipping LLM extraction", user_id)
            return {"extracted": 0, "stored": 0, "skipped": len(conversation), "budget_exceeded": True}

        # Stage 1: Sensory filtering
        prepared = prepare_for_extraction(conversation, self.sensory_config)
        if not prepared:
            return {"extracted": 0, "stored": 0, "skipped": len(conversation)}

        # Stage 2: LLM extraction (tracked)
        tracker = self._tracked_llm("extraction")
        memories = extract_memories(prepared, tracker)
        if not memories:
            self._record_usage(conn, user_id, tracker)
            return {"extracted": 0, "stored": 0, "skipped": 0}

        # Stage 3: Conflict resolution + storage (tracked)
        conflict_tracker = self._tracked_llm("conflict") or tracker.with_operation("conflict")
        min_conf = self._settings.extraction_min_confidence
        stored = 0
        skipped = 0

        for mem in memories:
            if mem.confidence < min_conf:
                skipped += 1
                continue
            self._store_single(conn, user_id, mem, session_id=session_id, llm_fn=conflict_tracker)
            stored += 1

        # Record total token usage
        self._record_usage(conn, user_id, tracker)

        return {"extracted": len(memories), "stored": stored, "skipped": skipped}

    def _drain_buffer(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """
        Mid-session extraction: extract memories from messages added since
        the last drain.  Does NOT clear L1 (session is still active).
        """
        key = (user_id, session_id)
        last_offset = self._drain_offsets.get(key, 0)

        all_messages = self._working.get_recent(conn, user_id, session_id)
        new_messages = all_messages[last_offset:]
        if not new_messages:
            return None

        conversation = [{"role": m["role"], "content": m["content"]} for m in new_messages]
        result = self._extract_and_store(conn, user_id, conversation, session_id=session_id)
        self._drain_offsets[key] = len(all_messages)
        return result

    def _maybe_consolidate(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
    ) -> None:
        """
        Check if consolidation is needed and run it in a background thread.

        Triggered when the number of new canonical memories since the last
        consolidation exceeds ``consolidation_trigger_threshold``.
        """
        consolidation_llm = self._get_llm("consolidation")
        if consolidation_llm is None:
            return

        threshold = self._settings.consolidation_trigger_threshold

        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(finished_at) FROM consolidation_log "
                "WHERE user_id = %s AND status = 'completed'",
                (user_id,),
            )
            row = cur.fetchone()
            last_finished = row[0] if row else None

            if last_finished:
                cur.execute(
                    "SELECT COUNT(*) FROM canonical_memories "
                    "WHERE user_id = %s AND status = 'active' AND created_at > %s",
                    (user_id, last_finished),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM canonical_memories "
                    "WHERE user_id = %s AND status = 'active'",
                    (user_id,),
                )
            new_count = cur.fetchone()[0]

        if new_count < threshold:
            return

        # Capture values for background thread (avoid closing over mutable state)
        dsn = self._settings.pg_dsn
        emb = self._emb
        sim_threshold = self._settings.consolidation_similarity_threshold
        max_cluster = self._settings.consolidation_max_cluster_size
        promotion_llm = self._get_llm("promotion") or consolidation_llm

        def _run() -> None:
            try:
                with psycopg.connect(dsn) as bg_conn:
                    # Advisory lock keyed on user_id hash to prevent concurrent
                    # consolidation for the same user.
                    lock_key = hash(user_id) % (2**31)
                    with bg_conn.cursor() as cur:
                        cur.execute(
                            "SELECT pg_try_advisory_lock(%s)", (lock_key,)
                        )
                        acquired = cur.fetchone()[0]
                    if not acquired:
                        logger.info("Consolidation already running for %s, skipping", user_id)
                        return

                    try:
                        consolidator = MemoryConsolidator(
                            emb, consolidation_llm,
                            similarity_threshold=sim_threshold,
                            max_cluster_size=max_cluster,
                        )
                        report = consolidator.consolidate(bg_conn, user_id)
                        promote_events_to_semantic(bg_conn, user_id, emb, promotion_llm)
                        bg_conn.commit()
                        logger.info(
                            "Consolidation for %s: scanned=%d merged=%d deleted=%d abstracted=%d",
                            user_id, report.memories_scanned, report.memories_merged,
                            report.memories_deleted, report.memories_abstracted,
                        )
                    finally:
                        with bg_conn.cursor() as cur:
                            cur.execute(
                                "SELECT pg_advisory_unlock(%s)", (lock_key,)
                            )
            except Exception:
                logger.exception("Background consolidation failed for %s", user_id)

        threading.Thread(target=_run, daemon=True).start()

    def _lazy_cleanup(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
    ) -> None:
        """
        Lightweight cleanup piggybacked on end_session.

        1. Remove orphaned working messages (stale sessions).
        2. Physically delete old superseded/deleted canonical memories.
        """
        timeout_hours = self._settings.orphan_session_timeout_hours
        cleanup_days = self._settings.superseded_cleanup_days

        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM working_messages "
                "WHERE user_id = %s "
                "AND created_at < NOW() - make_interval(hours => %s)",
                (user_id, timeout_hours),
            )
            cur.execute(
                "DELETE FROM canonical_memories "
                "WHERE user_id = %s AND status IN ('superseded', 'deleted') "
                "AND updated_at < NOW() - make_interval(days => %s)",
                (user_id, cleanup_days),
            )

    def _store_single(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        memory: ExtractedMemory,
        *,
        session_id: str | None = None,
        llm_fn: TokenTracker | None = None,
    ) -> str:
        """Store a single memory with conflict resolution."""
        # Use a savepoint so we can roll back to a clean state on failure
        # without aborting the outer transaction.
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT _store_single_sp")
        try:
            resolutions = resolve_conflict(
                conn, user_id, memory, self._emb,
                llm_fn=llm_fn or self._get_llm("conflict"),
            )
            embedding = coerce_pgvector_dims(
                self._emb.embed_query(memory.value or memory.content)
            )
            memory_id = ""
            for res in resolutions:
                result_id = apply_resolution(conn, user_id, res, embedding=embedding)
                if result_id:
                    memory_id = result_id
            with conn.cursor() as cur:
                cur.execute("RELEASE SAVEPOINT _store_single_sp")
            return memory_id
        except Exception:
            logger.warning("Conflict resolution failed, falling back to direct store")
            with conn.cursor() as cur:
                cur.execute("ROLLBACK TO SAVEPOINT _store_single_sp")
            tbl = table_for_memory(memory)
            if tbl == "episodic_memories":
                return store_episodic(conn, user_id, memory, self._emb, session_id=session_id)
            return store_semantic(conn, user_id, memory, self._emb)
