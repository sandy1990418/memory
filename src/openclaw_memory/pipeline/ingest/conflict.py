"""
mem0-style CRUD conflict resolution for canonical memories.

Always active in the pipeline. Determines how to integrate a new
extracted memory against existing canonical memories:

  ADD       — genuinely new information
  UPDATE    — same key, updated value
  SUPERSEDE — same key, newer/better candidate
  DELETE    — existing memory made obsolete
  NOOP      — duplicate or stale candidate

Supports rule-based (fast, no LLM) and LLM-based (Mem0 CRUD) modes.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg

from ...core.embeddings import EmbeddingProvider
from ...core.types import ExtractedMemory
from ...utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)

# Action constants
ADD = "ADD"
UPDATE = "UPDATE"
SUPERSEDE = "SUPERSEDE"
DELETE = "DELETE"
NOOP = "NOOP"


@dataclass
class ResolutionResult:
    """Outcome of conflict resolution for a single candidate memory."""

    action: str
    candidate: ExtractedMemory
    existing_id: str | None = None
    reason: str = ""
    new_value: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_value(candidate: ExtractedMemory) -> str:
    return candidate.value or candidate.content


def _parse_event_time(candidate: ExtractedMemory) -> datetime | None:
    raw = candidate.event_time
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# DB lookups
# ---------------------------------------------------------------------------


def _find_by_key(
    conn: psycopg.Connection[Any], user_id: str, key: str,
) -> dict[str, Any] | None:
    """Find active canonical memory by exact key match."""
    sql = """
        SELECT id::text, memory_key, value, confidence, event_time, embedding
        FROM canonical_memories
        WHERE user_id = %s AND memory_key = %s AND status = 'active'
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id, key))
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "memory_key": row[1],
        "value": row[2],
        "confidence": float(row[3]) if row[3] is not None else 0.8,
        "event_time": row[4],
        "embedding": row[5],
    }


def _find_similar(
    conn: psycopg.Connection[Any],
    user_id: str,
    query_embedding: list[float],
    *,
    threshold: float = 0.80,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find similar active canonical memories via pgvector."""
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    sql = """
        SELECT id::text, memory_key, value, memory_type, confidence, event_time,
               1 - (embedding <=> %s::vector) AS similarity
        FROM canonical_memories
        WHERE user_id = %s AND status = 'active'
          AND embedding IS NOT NULL
          AND 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (emb_str, user_id, emb_str, threshold, emb_str, top_k))
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]), "memory_key": r[1] or "", "value": r[2] or "",
            "memory_type": r[3] or "", "confidence": float(r[4]) if r[4] else 0.8,
            "event_time": r[5], "similarity": float(r[6]),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Rule-based resolver
# ---------------------------------------------------------------------------


def resolve_rules(
    conn: psycopg.Connection[Any],
    user_id: str,
    candidate: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    similarity_threshold: float = 0.85,
) -> ResolutionResult:
    """Deterministic rule-based conflict resolution."""
    key = candidate.memory_key
    value = _get_value(candidate)
    event_time = _parse_event_time(candidate)

    # Step 1: exact key match
    if key:
        existing = _find_by_key(conn, user_id, key)
        if existing is not None:
            eid = existing["id"]
            et = existing["event_time"]
            ec = existing["confidence"]

            if event_time and et:
                if event_time > et:
                    return ResolutionResult(SUPERSEDE, candidate, eid, "Newer event_time")
                if event_time < et:
                    return ResolutionResult(NOOP, candidate, eid, "Existing is newer")

            if candidate.confidence > ec:
                return ResolutionResult(SUPERSEDE, candidate, eid, "Higher confidence")
            if candidate.confidence < ec:
                return ResolutionResult(NOOP, candidate, eid, "Existing has higher confidence")

            if value == existing["value"]:
                return ResolutionResult(NOOP, candidate, eid, "Identical value")

            return ResolutionResult(UPDATE, candidate, eid, "Same key, different value")

    # Step 2: embedding similarity fallback
    embedding = embedding_provider.embed_query(value)
    # Check against all active canonical memories
    sql = """
        SELECT id::text, embedding FROM canonical_memories
        WHERE user_id = %s AND status = 'active' AND embedding IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()

    for row_id, emb in rows:
        if emb is None:
            continue
        stored = list(emb) if not isinstance(emb, list) else emb
        if cosine_similarity(embedding, stored) >= similarity_threshold:
            return ResolutionResult(NOOP, candidate, str(row_id), "Near-duplicate")

    return ResolutionResult(ADD, candidate, None, "No existing match")


# ---------------------------------------------------------------------------
# LLM-based resolver
# ---------------------------------------------------------------------------

_CRUD_SYSTEM = """\
You are a memory manager. Decide how to handle a new memory given existing ones.
Respond with a JSON array of action objects:
- "action": "ADD" | "UPDATE" | "DELETE" | "NOOP"
- "memory_id": existing memory ID (for UPDATE/DELETE) or null
- "new_value": updated text (for UPDATE) or null
- "reason": brief explanation
Respond ONLY with valid JSON.\
"""


def resolve_llm(
    conn: psycopg.Connection[Any],
    user_id: str,
    candidate: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    llm_fn: Callable[[str], str],
    *,
    similarity_threshold: float = 0.80,
    top_k: int = 5,
) -> list[ResolutionResult]:
    """LLM-based CRUD conflict resolution (Mem0-style)."""
    value = _get_value(candidate)
    embedding = embedding_provider.embed_query(value)

    try:
        existing = _find_similar(
            conn, user_id, embedding,
            threshold=similarity_threshold, top_k=top_k,
        )
    except Exception:
        logger.warning("pgvector similarity search failed, proceeding empty")
        existing = []

    # Build prompt
    parts = [
        f"New memory: key={candidate.memory_key}, value={value}, "
        f"type={candidate.memory_type}, confidence={candidate.confidence}",
        "",
    ]
    if existing:
        parts.append("Existing memories:")
        for i, m in enumerate(existing):
            parts.append(
                f"  [{i+1}] id={m['id']} key={m['memory_key']} "
                f"value={m['value']} sim={m['similarity']:.3f}"
            )
    else:
        parts.append("Existing memories: none")

    try:
        response = llm_fn(_CRUD_SYSTEM + "\n\n" + "\n".join(parts))
    except Exception:
        logger.warning("LLM CRUD call failed, defaulting to ADD")
        return [ResolutionResult(ADD, candidate, None, "LLM call failed")]

    # Parse response
    text = re.sub(r"^```(?:json)?\s*", "", response.strip())
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        actions = json.loads(text)
    except json.JSONDecodeError:
        return [ResolutionResult(ADD, candidate, None, "Parse failure")]

    if not isinstance(actions, list):
        actions = [actions]

    existing_ids = {m["id"] for m in existing}
    results: list[ResolutionResult] = []
    for act in actions:
        if not isinstance(act, dict):
            continue
        action = str(act.get("action", "")).upper()
        mid = str(act.get("memory_id", "")) if act.get("memory_id") else None
        reason = str(act.get("reason", ""))

        if action == "ADD":
            results.append(ResolutionResult(ADD, candidate, None, reason))
        elif action == "UPDATE" and mid and mid in existing_ids:
            results.append(ResolutionResult(
                UPDATE, candidate, mid, reason,
                new_value=str(act.get("new_value", "")) or None,
            ))
        elif action == "DELETE" and mid and mid in existing_ids:
            results.append(ResolutionResult(DELETE, candidate, mid, reason))
        elif action == "NOOP":
            results.append(ResolutionResult(NOOP, candidate, mid, reason))

    return results or [ResolutionResult(ADD, candidate, None, "No valid actions")]


# ---------------------------------------------------------------------------
# Unified resolver
# ---------------------------------------------------------------------------


def resolve_conflict(
    conn: psycopg.Connection[Any],
    user_id: str,
    candidate: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    llm_fn: Callable[[str], str] | None = None,
    *,
    similarity_threshold: float = 0.85,
) -> list[ResolutionResult]:
    """Resolve conflict: LLM-based if llm_fn provided, else rules."""
    if llm_fn is not None:
        return resolve_llm(
            conn, user_id, candidate, embedding_provider, llm_fn,
            similarity_threshold=similarity_threshold,
        )
    return [resolve_rules(
        conn, user_id, candidate, embedding_provider, similarity_threshold,
    )]


# ---------------------------------------------------------------------------
# Apply resolution to DB
# ---------------------------------------------------------------------------


def apply_resolution(
    conn: psycopg.Connection[Any],
    user_id: str,
    result: ResolutionResult,
    embedding: list[float] | None = None,
) -> str | None:
    """Execute a resolution result against the database."""
    if result.action == NOOP:
        return None

    if result.action == DELETE:
        if result.existing_id:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE canonical_memories SET status = 'superseded', "
                    "updated_at = now() WHERE id = %s::uuid",
                    (result.existing_id,),
                )
        return None

    candidate = result.candidate
    key = candidate.memory_key or str(uuid.uuid4())
    value = _get_value(candidate)
    if result.action == UPDATE and result.new_value:
        value = result.new_value

    event_time = _parse_event_time(candidate)
    metadata = candidate.metadata or {}
    if candidate.source_refs:
        metadata = {**metadata, "source_refs": candidate.source_refs}

    with conn.cursor() as cur:
        if result.action == ADD:
            new_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO canonical_memories
                    (id, user_id, memory_key, value, memory_type, confidence,
                     event_time, status, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)
                """,
                (new_id, user_id, key, value, candidate.memory_type,
                 candidate.confidence, event_time, embedding, json.dumps(metadata)),
            )
            return new_id

        if result.action == UPDATE:
            cur.execute(
                """
                UPDATE canonical_memories
                SET value = %s, confidence = %s, event_time = %s,
                    metadata = %s, updated_at = now()
                WHERE id = %s::uuid
                """,
                (value, candidate.confidence, event_time,
                 json.dumps(metadata), result.existing_id),
            )
            return result.existing_id

        if result.action == SUPERSEDE:
            cur.execute(
                "UPDATE canonical_memories SET status = 'superseded', "
                "updated_at = now() WHERE id = %s::uuid",
                (result.existing_id,),
            )
            new_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO canonical_memories
                    (id, user_id, memory_key, value, memory_type, confidence,
                     event_time, status, supersedes_memory_id, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s::uuid, %s, %s)
                """,
                (new_id, user_id, key, value, candidate.memory_type,
                 candidate.confidence, event_time, result.existing_id,
                 embedding, json.dumps(metadata)),
            )
            return new_id

    return None
