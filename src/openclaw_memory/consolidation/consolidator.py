"""
Sleep-time consolidation engine for offline memory maintenance.

Clusters near-duplicate canonical memories, merges them via LLM,
and detects stale entries — inspired by LightMem's sleep-time mechanism.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

import psycopg

from ..core.embeddings import EmbeddingProvider
from ..core.types import ConsolidationReport
from ..utils.similarity import cosine_similarity
from ..utils.text import parse_llm_json

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

MERGE_PROMPT = (
    "You are a memory consolidation assistant.\n"
    "Given these related memories belonging to one user, create a single "
    "consolidated memory that preserves all important information.\n\n"
    "Memories:\n{memories}\n\n"
    "If the memories should NOT be merged (they contain distinct facts), "
    'respond with exactly: {{"action": "keep_separate"}}\n\n'
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
# Consolidator
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """
    Offline memory consolidation engine.

    Runs clustering, LLM-based merging, staleness detection, and
    episodic-to-semantic promotion.

    Args:
        embedding_provider: Embedding provider for content embedding.
        llm_fn:             LLM callable (prompt -> response).
        similarity_threshold: Cosine similarity for clustering (default 0.90).
        max_cluster_size:   Max memories per merge cluster (default 10).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_fn: Callable[[str], str],
        *,
        similarity_threshold: float = 0.90,
        max_cluster_size: int = 10,
    ) -> None:
        self._emb = embedding_provider
        self._llm_fn = llm_fn
        self._sim_threshold = similarity_threshold
        self._max_cluster = max_cluster_size

    def consolidate(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
    ) -> ConsolidationReport:
        """Run full consolidation for a single user."""
        start = time.monotonic()
        report = ConsolidationReport(user_id=user_id)

        memories = self._load_active(conn, user_id)
        report.memories_scanned = len(memories)

        # Step 1: Cluster and merge
        for cluster in self._cluster(memories):
            try:
                merged = self._merge_cluster(cluster)
                if merged is not None:
                    self._apply_merge(conn, user_id, cluster, merged)
                    report.memories_merged += len(cluster)
            except Exception as exc:
                report.errors.append(f"merge: {exc}")

        # Step 2: Staleness detection
        try:
            for action, mem_row in self._check_staleness(conn, user_id):
                if action == "delete":
                    self._soft_delete(conn, mem_row["id"])
                    report.memories_deleted += 1
                elif action == "abstract":
                    self._abstract(conn, user_id, mem_row)
                    report.memories_abstracted += 1
        except Exception as exc:
            report.errors.append(f"staleness: {exc}")

        report.duration_seconds = time.monotonic() - start
        return report

    # ------------------------------------------------------------------
    # Memory loading
    # ------------------------------------------------------------------

    def _load_active(
        self, conn: psycopg.Connection[Any], user_id: str,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT id::text, memory_key, value, memory_type, confidence,
                   embedding, metadata, created_at
            FROM canonical_memories
            WHERE user_id = %s AND status = 'active'
            ORDER BY created_at
        """
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "memory_key": r[1], "value": r[2],
                "memory_type": r[3],
                "confidence": float(r[4]) if r[4] is not None else 0.8,
                "embedding": r[5], "metadata": r[6] if isinstance(r[6], dict) else {},
                "created_at": r[7],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster(
        self, memories: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        if len(memories) < 2:
            return []

        by_id = {m["id"]: m for m in memories}
        with_emb = [m for m in memories if m.get("embedding") is not None]
        if len(with_emb) < 2:
            return []

        uf = _UnionFind()
        for i, a in enumerate(with_emb):
            emb_a = list(a["embedding"]) if not isinstance(a["embedding"], list) else a["embedding"]
            for j in range(i + 1, len(with_emb)):
                b = with_emb[j]
                emb_b = list(b["embedding"]) if not isinstance(b["embedding"], list) else b["embedding"]
                if cosine_similarity(emb_a, emb_b) >= self._sim_threshold:
                    uf.union(a["id"], b["id"])

        clusters: list[list[dict[str, Any]]] = []
        for member_ids in uf.groups().values():
            if len(member_ids) >= 2:
                clusters.append([by_id[mid] for mid in member_ids[:self._max_cluster]])
        return clusters

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_cluster(self, cluster: list[dict[str, Any]]) -> dict[str, Any] | None:
        mem_texts = "\n".join(
            f"- [{m['memory_type']}] {m['value']} (key={m['memory_key']}, "
            f"confidence={m['confidence']})"
            for m in cluster
        )
        prompt = MERGE_PROMPT.replace("{memories}", mem_texts)
        parsed = parse_llm_json(self._llm_fn(prompt))
        if parsed is None or parsed.get("action") != "merge" or not parsed.get("content"):
            return None
        return parsed

    def _apply_merge(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        originals: list[dict[str, Any]],
        merged: dict[str, Any],
    ) -> str:
        new_id = str(uuid.uuid4())
        original_ids = [m["id"] for m in originals]
        embedding = self._emb.embed_query(merged["content"])

        with conn.cursor() as cur:
            for oid in original_ids:
                cur.execute(
                    "UPDATE canonical_memories SET status = 'superseded', "
                    "updated_at = now() WHERE id = %s::uuid",
                    (oid,),
                )
            cur.execute(
                """
                INSERT INTO canonical_memories
                    (id, user_id, memory_key, value, memory_type, confidence,
                     status, embedding, metadata, consolidated_from)
                VALUES (%s, %s, %s, %s, %s, %s, 'active', %s, %s, %s)
                """,
                (
                    new_id, user_id,
                    merged.get("memory_key", ""),
                    merged["content"],
                    merged.get("memory_type", "fact"),
                    merged.get("confidence", 0.9),
                    embedding,
                    json.dumps({"source": "consolidation"}),
                    original_ids,
                ),
            )
        return new_id

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def _check_staleness(
        self, conn: psycopg.Connection[Any], user_id: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        sql = """
            SELECT id::text, memory_key, value, memory_type, confidence, created_at
            FROM canonical_memories
            WHERE user_id = %s AND status = 'active' AND confidence < 0.7
              AND created_at < now() - interval '180 days'
        """
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()

        results: list[tuple[str, dict[str, Any]]] = []
        for r in rows:
            mem = {
                "id": str(r[0]), "memory_key": r[1], "value": r[2],
                "memory_type": r[3], "confidence": float(r[4]) if r[4] else 0.0,
                "created_at": r[5],
            }
            prompt = STALENESS_PROMPT.format(
                memory=mem["value"] or "",
                memory_type=mem["memory_type"] or "fact",
                created_at=str(mem["created_at"] or "unknown"),
                confidence=str(mem["confidence"]),
            )
            parsed = parse_llm_json(self._llm_fn(prompt))
            if parsed and parsed.get("action") in ("delete", "abstract"):
                if parsed["action"] == "abstract":
                    mem["abstracted_content"] = parsed.get("abstracted_content", "")
                results.append((parsed["action"], mem))
        return results

    def _soft_delete(self, conn: psycopg.Connection[Any], memory_id: str) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE canonical_memories SET status = 'deleted', "
                "updated_at = now() WHERE id = %s::uuid",
                (memory_id,),
            )

    def _abstract(
        self, conn: psycopg.Connection[Any], user_id: str, mem: dict[str, Any],
    ) -> str:
        content = mem.get("abstracted_content", "")
        if not content:
            return ""
        new_id = str(uuid.uuid4())
        embedding = self._emb.embed_query(content)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE canonical_memories SET status = 'superseded', "
                "updated_at = now() WHERE id = %s::uuid",
                (mem["id"],),
            )
            cur.execute(
                """
                INSERT INTO canonical_memories
                    (id, user_id, memory_key, value, memory_type, confidence,
                     status, embedding, metadata, consolidated_from)
                VALUES (%s, %s, %s, %s, 'fact', 0.85, 'active', %s, %s, %s)
                """,
                (
                    new_id, user_id, mem.get("memory_key", ""),
                    content, embedding,
                    json.dumps({"source": "staleness_abstraction"}),
                    [mem["id"]],
                ),
            )
        return new_id
