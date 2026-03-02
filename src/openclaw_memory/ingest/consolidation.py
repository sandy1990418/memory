"""
Sleep-time consolidation module for offline memory maintenance.

Periodically merges near-duplicate memories, detects stale entries, and
promotes recurring episodic patterns to semantic facts — inspired by
LightMem's sleep-time update mechanism.

All operations are scoped by user_id — no cross-user data is ever returned.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from ..embeddings import EmbeddingProvider

if TYPE_CHECKING:
    import psycopg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STALENESS_MONTHS = 6
_PROMOTION_REFERENCE_THRESHOLD = 3
_PROMOTION_AGE_DAYS = 90

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

MERGE_PROMPT = (
    "You are a memory consolidation assistant.\n"
    "Given these related memories belonging to one user, create a single "
    "consolidated memory that preserves all important information.\n\n"
    "Memories:\n{memories}\n\n"
    "If the memories should NOT be merged (they contain distinct facts), "
    'respond with exactly: {"action": "keep_separate"}\n\n'
    "Otherwise respond with a JSON object:\n"
    '  {{"action": "merge", "content": "...", "memory_type": "...", '
    '"confidence": 0.0-1.0, "memory_key": "..."}}\n\n'
    "Respond ONLY with valid JSON, no markdown fences."
)

STALENESS_PROMPT = (
    "You are a memory relevance assistant.\n"
    "Evaluate whether this memory is still relevant and useful:\n\n"
    "Memory: {memory}\n"
    "Memory type: {memory_type}\n"
    "Created: {created_at}\n"
    "Confidence: {confidence}\n\n"
    "Respond with a JSON object:\n"
    '  {{"action": "keep" | "delete" | "abstract", '
    '"reason": "brief explanation"}}\n'
    'If "abstract", also include "abstracted_content": "..." with a '
    "general fact derived from the memory.\n\n"
    "Respond ONLY with valid JSON, no markdown fences."
)

ABSTRACTION_PROMPT = (
    "You are a memory abstraction assistant.\n"
    "Convert this specific episodic memory into a general semantic fact or "
    "preference:\n\n"
    "Memory: {memory}\n"
    "Memory type: {memory_type}\n\n"
    "Respond with a JSON object:\n"
    '  {{"content": "...", "memory_type": "fact" or "preference", '
    '"memory_key": "...", "confidence": 0.0-1.0}}\n\n'
    "Respond ONLY with valid JSON, no markdown fences."
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationReport:
    """Summary of a single user's consolidation run."""

    user_id: str
    memories_scanned: int = 0
    memories_merged: int = 0
    memories_deleted: int = 0
    memories_abstracted: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of LLM JSON response, stripping markdown fences."""
    import re

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# Union-Find for clustering
# ---------------------------------------------------------------------------


class _UnionFind:
    """Simple union-find / disjoint-set structure."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            result.setdefault(root, []).append(x)
        return result


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """Offline memory consolidation engine.

    Args:
        pg_dsn:               PostgreSQL connection DSN.
        embedding_provider:   Embedding provider implementing EmbeddingProvider protocol.
        llm_fn:               LLM callable — takes a prompt string, returns a response string.
        similarity_threshold: Cosine similarity threshold for clustering (default 0.90).
        max_cluster_size:     Maximum number of memories per merge cluster (default 10).
    """

    def __init__(
        self,
        pg_dsn: str,
        embedding_provider: EmbeddingProvider,
        llm_fn: Callable[[str], str],
        *,
        similarity_threshold: float = 0.90,
        max_cluster_size: int = 10,
    ) -> None:
        self._pg_dsn = pg_dsn
        self._embedding_provider = embedding_provider
        self._llm_fn = llm_fn
        self._similarity_threshold = similarity_threshold
        self._max_cluster_size = max_cluster_size

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_conn(self) -> psycopg.Connection[Any]:
        from .pg_schema import get_pg_connection

        return get_pg_connection(self._pg_dsn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self, user_id: str) -> ConsolidationReport:
        """Run full consolidation for a single user."""
        start = time.monotonic()
        report = ConsolidationReport(user_id=user_id)

        conn = self._get_conn()
        log_id = self._start_log(conn, user_id)

        try:
            # Load all active canonical memories for user
            memories = self._load_active_memories(conn, user_id)
            report.memories_scanned = len(memories)

            # Step 1: Cluster and merge
            clusters = self._cluster_memories(memories)
            for cluster in clusters:
                try:
                    merged = self._merge_cluster(cluster)
                    if merged is not None:
                        self._apply_merge(conn, user_id, cluster, merged)
                        report.memories_merged += len(cluster)
                except Exception as exc:
                    report.errors.append(f"merge error: {exc}")

            # Step 2: Staleness detection
            try:
                stale_results = self._check_staleness(conn, user_id)
                for action, mem_row in stale_results:
                    if action == "delete":
                        self._mark_deleted(conn, mem_row["id"])
                        report.memories_deleted += 1
                    elif action == "abstract":
                        self._abstract_memory(conn, user_id, mem_row)
                        report.memories_abstracted += 1
            except Exception as exc:
                report.errors.append(f"staleness error: {exc}")

            # Step 3: Episodic -> Semantic promotion
            try:
                promoted = self._promote_to_semantic(conn, user_id)
                report.memories_abstracted += promoted
            except Exception as exc:
                report.errors.append(f"promotion error: {exc}")

            report.duration_seconds = time.monotonic() - start
            self._finish_log(conn, log_id, report, status="completed")

        except Exception as exc:
            report.errors.append(f"consolidation error: {exc}")
            report.duration_seconds = time.monotonic() - start
            self._finish_log(conn, log_id, report, status="error", error=str(exc))

        return report

    def consolidate_all(self, *, batch_size: int = 50) -> list[ConsolidationReport]:
        """Batch process all users with active canonical memories."""
        conn = self._get_conn()
        user_ids = self._get_all_user_ids(conn)

        reports: list[ConsolidationReport] = []
        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i : i + batch_size]
            for uid in batch:
                report = self.consolidate(uid)
                reports.append(report)
        return reports

    # ------------------------------------------------------------------
    # Memory loading
    # ------------------------------------------------------------------

    def _load_active_memories(
        self, conn: psycopg.Connection[Any], user_id: str
    ) -> list[dict[str, Any]]:
        """Load all active canonical memories for a user."""
        sql = """
            SELECT id::text, memory_key, value, memory_type, confidence,
                   event_time, embedding, metadata, created_at
            FROM canonical_memories
            WHERE user_id = %s AND status = 'active'
            ORDER BY created_at
        """
        with conn.cursor() as cur:
            cur.execute(sql, [user_id])
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            row_id, key, value, mtype, conf, etime, emb, meta, created = row
            results.append({
                "id": str(row_id),
                "memory_key": key,
                "value": value,
                "memory_type": mtype,
                "confidence": float(conf) if conf is not None else 0.8,
                "event_time": etime,
                "embedding": emb,
                "metadata": meta if isinstance(meta, dict) else {},
                "created_at": created,
            })
        return results

    def _get_all_user_ids(self, conn: psycopg.Connection[Any]) -> list[str]:
        """Get distinct user_ids from canonical_memories."""
        sql = """
            SELECT DISTINCT user_id
            FROM canonical_memories
            WHERE status = 'active'
            ORDER BY user_id
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_memories(
        self, memories: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Group similar memories using embedding cosine similarity."""
        if len(memories) < 2:
            return []

        # Build index by id
        by_id: dict[str, dict[str, Any]] = {m["id"]: m for m in memories}

        # Filter to memories with embeddings
        with_emb = [m for m in memories if m.get("embedding") is not None]
        if len(with_emb) < 2:
            return []

        # Find similar pairs using pairwise cosine
        uf = _UnionFind()
        for i, a in enumerate(with_emb):
            for j in range(i + 1, len(with_emb)):
                b = with_emb[j]
                sim = self._cosine_similarity(a["embedding"], b["embedding"])
                if sim >= self._similarity_threshold:
                    uf.union(a["id"], b["id"])

        # Build clusters, filter to 2+ members, cap at max_cluster_size
        clusters: list[list[dict[str, Any]]] = []
        for _root, member_ids in uf.groups().items():
            if len(member_ids) < 2:
                continue
            cluster = [by_id[mid] for mid in member_ids[:self._max_cluster_size]]
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        """Compute cosine similarity between two embedding vectors."""
        import math

        vec_a = list(a) if not isinstance(a, list) else a
        vec_b = list(b) if not isinstance(b, list) else b
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(x * x for x in vec_a))
        mag_b = math.sqrt(sum(y * y for y in vec_b))
        if mag_a < 1e-10 or mag_b < 1e-10:
            return 0.0
        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_cluster(self, memories: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Ask LLM to merge a cluster of related memories. Returns merged dict or None."""
        mem_texts = "\n".join(
            f"- [{m['memory_type']}] {m['value']} (key={m['memory_key']}, "
            f"confidence={m['confidence']})"
            for m in memories
        )
        prompt = MERGE_PROMPT.replace("{memories}", mem_texts)
        raw = self._llm_fn(prompt)
        parsed = _parse_llm_json(raw)
        if parsed is None:
            return None
        if parsed.get("action") == "keep_separate":
            return None
        if parsed.get("action") == "merge" and parsed.get("content"):
            return parsed
        return None

    def _apply_merge(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        originals: list[dict[str, Any]],
        merged: dict[str, Any],
    ) -> str:
        """Insert merged memory and mark originals as superseded."""
        new_id = str(uuid.uuid4())
        original_ids = [m["id"] for m in originals]

        # Embed the merged content
        embedding = self._embedding_provider.embed_query(merged["content"])

        # Get the max consolidation_round from originals
        max_round = 0
        for m in originals:
            meta = m.get("metadata", {})
            if isinstance(meta, dict):
                max_round = max(max_round, meta.get("consolidation_round", 0))

        with conn.cursor() as cur:
            # Mark originals as superseded
            for oid in original_ids:
                cur.execute(
                    """
                    UPDATE canonical_memories
                    SET status = 'superseded', updated_at = now()
                    WHERE id = %s::uuid
                    """,
                    [oid],
                )

            # Insert merged memory
            confidence = merged.get("confidence", 0.9)
            memory_type = merged.get("memory_type", "fact")
            memory_key = merged.get("memory_key", "")
            cur.execute(
                """
                INSERT INTO canonical_memories
                    (id, user_id, memory_key, value, memory_type, confidence,
                     status, embedding, metadata,
                     consolidated_from, consolidation_round, last_consolidated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'active', %s, %s,
                        %s, %s, now())
                """,
                [
                    new_id,
                    user_id,
                    memory_key,
                    merged["content"],
                    memory_type,
                    confidence,
                    embedding,
                    json.dumps({"source": "consolidation"}),
                    original_ids,
                    max_round + 1,
                ],
            )
        return new_id

    # ------------------------------------------------------------------
    # Staleness detection
    # ------------------------------------------------------------------

    def _check_staleness(
        self, conn: psycopg.Connection[Any], user_id: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Find and evaluate potentially stale memories."""
        cutoff = _now_utc() - timedelta(days=_STALENESS_MONTHS * 30)

        sql = """
            SELECT id::text, memory_key, value, memory_type, confidence,
                   event_time, created_at
            FROM canonical_memories
            WHERE user_id = %s
              AND status = 'active'
              AND confidence < 0.7
              AND (event_time IS NOT NULL AND event_time < %s
                   OR event_time IS NULL AND created_at < %s)
        """
        with conn.cursor() as cur:
            cur.execute(sql, [user_id, cutoff, cutoff])
            rows = cur.fetchall()

        results: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            row_id, key, value, mtype, conf, etime, created = row
            mem_row = {
                "id": str(row_id),
                "memory_key": key,
                "value": value,
                "memory_type": mtype,
                "confidence": float(conf) if conf is not None else 0.0,
                "event_time": etime,
                "created_at": created,
            }

            prompt = STALENESS_PROMPT.replace("{memory}", value or "").replace(
                "{memory_type}", mtype or "fact"
            ).replace(
                "{created_at}", str(created or "unknown")
            ).replace(
                "{confidence}", str(conf)
            )
            raw = self._llm_fn(prompt)
            parsed = _parse_llm_json(raw)
            if parsed is None:
                continue
            action = parsed.get("action", "keep")
            if action in ("delete", "abstract"):
                if action == "abstract":
                    mem_row["abstracted_content"] = parsed.get(
                        "abstracted_content", ""
                    )
                results.append((action, mem_row))
        return results

    def _mark_deleted(self, conn: psycopg.Connection[Any], memory_id: str) -> None:
        """Mark a memory as deleted (soft-delete)."""
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE canonical_memories
                SET status = 'deleted', updated_at = now()
                WHERE id = %s::uuid
                """,
                [memory_id],
            )

    def _abstract_memory(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        mem_row: dict[str, Any],
    ) -> str:
        """Replace a stale memory with its abstracted form."""
        abstracted_content = mem_row.get("abstracted_content", "")
        if not abstracted_content:
            return ""

        new_id = str(uuid.uuid4())
        embedding = self._embedding_provider.embed_query(abstracted_content)

        with conn.cursor() as cur:
            # Mark original as superseded
            cur.execute(
                """
                UPDATE canonical_memories
                SET status = 'superseded', updated_at = now()
                WHERE id = %s::uuid
                """,
                [mem_row["id"]],
            )
            # Insert abstracted memory
            cur.execute(
                """
                INSERT INTO canonical_memories
                    (id, user_id, memory_key, value, memory_type, confidence,
                     status, embedding, metadata,
                     consolidated_from, consolidation_round, last_consolidated_at)
                VALUES (%s, %s, %s, %s, 'fact', 0.85, 'active', %s, %s,
                        %s, 1, now())
                """,
                [
                    new_id,
                    user_id,
                    mem_row.get("memory_key", ""),
                    abstracted_content,
                    embedding,
                    json.dumps({"source": "staleness_abstraction"}),
                    [mem_row["id"]],
                ],
            )
        return new_id

    # ------------------------------------------------------------------
    # Episodic -> Semantic promotion
    # ------------------------------------------------------------------

    def _promote_to_semantic(
        self, conn: psycopg.Connection[Any], user_id: str
    ) -> int:
        """Promote recurring episodic patterns to semantic memories."""
        cutoff = _now_utc() - timedelta(days=_PROMOTION_AGE_DAYS)

        # Find event-type memories that are old or referenced frequently
        sql = """
            SELECT id::text, memory_key, value, memory_type, confidence,
                   created_at, metadata
            FROM canonical_memories
            WHERE user_id = %s
              AND status = 'active'
              AND memory_type = 'event'
              AND created_at < %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, [user_id, cutoff])
            rows = cur.fetchall()

        promoted_count = 0
        for row in rows:
            row_id, key, value, mtype, conf, created, meta = row
            prompt = ABSTRACTION_PROMPT.replace(
                "{memory}", value or ""
            ).replace("{memory_type}", mtype or "event")
            raw = self._llm_fn(prompt)
            parsed = _parse_llm_json(raw)
            if parsed is None or not parsed.get("content"):
                continue

            new_id = str(uuid.uuid4())
            embedding = self._embedding_provider.embed_query(parsed["content"])
            new_type = parsed.get("memory_type", "fact")
            new_key = parsed.get("memory_key", key or "")
            new_conf = parsed.get("confidence", 0.85)

            with conn.cursor() as cur2:
                cur2.execute(
                    """
                    UPDATE canonical_memories
                    SET status = 'superseded', updated_at = now()
                    WHERE id = %s::uuid
                    """,
                    [str(row_id)],
                )
                cur2.execute(
                    """
                    INSERT INTO canonical_memories
                        (id, user_id, memory_key, value, memory_type, confidence,
                         status, embedding, metadata,
                         consolidated_from, consolidation_round, last_consolidated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'active', %s, %s,
                            %s, 1, now())
                    """,
                    [
                        new_id,
                        user_id,
                        new_key,
                        parsed["content"],
                        new_type,
                        new_conf,
                        embedding,
                        json.dumps({"source": "episodic_promotion"}),
                        [str(row_id)],
                    ],
                )
            promoted_count += 1
        return promoted_count

    # ------------------------------------------------------------------
    # Consolidation log
    # ------------------------------------------------------------------

    def _start_log(
        self, conn: psycopg.Connection[Any], user_id: str
    ) -> int:
        """Create a consolidation_log entry and return its id."""
        sql = """
            INSERT INTO consolidation_log (user_id, status)
            VALUES (%s, 'running')
            RETURNING id
        """
        with conn.cursor() as cur:
            cur.execute(sql, [user_id])
            row = cur.fetchone()
        return row[0] if row else 0

    def _finish_log(
        self,
        conn: psycopg.Connection[Any],
        log_id: int,
        report: ConsolidationReport,
        *,
        status: str = "completed",
        error: str | None = None,
    ) -> None:
        """Update a consolidation_log entry with final results."""
        sql = """
            UPDATE consolidation_log
            SET finished_at = now(),
                memories_scanned = %s,
                memories_merged = %s,
                memories_deleted = %s,
                memories_abstracted = %s,
                status = %s,
                error = %s,
                metadata = %s
            WHERE id = %s
        """
        meta = {}
        if report.errors:
            meta["errors"] = report.errors
        meta["duration_seconds"] = report.duration_seconds

        with conn.cursor() as cur:
            cur.execute(
                sql,
                [
                    report.memories_scanned,
                    report.memories_merged,
                    report.memories_deleted,
                    report.memories_abstracted,
                    status,
                    error,
                    json.dumps(meta),
                    log_id,
                ],
            )
