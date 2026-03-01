"""
Action-based conflict resolution for canonical memories.

Compares an incoming ExtractedMemory candidate against existing canonical
memories in the database and decides how to handle it:

  ADD       — new key, no existing match → insert fresh row
  UPDATE    — same key, same confidence/time → in-place value update
  SUPERSEDE — same key, better candidate → mark old row inactive, insert new
  NOOP      — candidate is stale or duplicate → skip entirely
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .embeddings import EmbeddingProvider
from .extraction import ExtractedMemory

if TYPE_CHECKING:
    import psycopg

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

ADD = "ADD"
UPDATE = "UPDATE"
SUPERSEDE = "SUPERSEDE"
NOOP = "NOOP"

# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------


@dataclass
class ResolutionResult:
    """Outcome of conflict resolution for a single candidate memory."""

    action: str  # ADD / UPDATE / SUPERSEDE / NOOP
    candidate: ExtractedMemory
    existing_id: str | None = None  # UUID of matched existing row
    reason: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


def _get_candidate_key(candidate: ExtractedMemory) -> str | None:
    """Extract memory_key from candidate using getattr for forward-compat."""
    return getattr(candidate, "memory_key", None)


def _get_candidate_value(candidate: ExtractedMemory) -> str:
    """Extract value from candidate, falling back to content."""
    return getattr(candidate, "value", None) or candidate.content


def _get_candidate_event_time(candidate: ExtractedMemory) -> datetime | None:
    """
    Extract event_time from candidate and normalize to datetime.

    Accepts datetime objects and ISO-8601 strings.
    """
    raw = getattr(candidate, "event_time", None)
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


def _get_candidate_source_refs(candidate: ExtractedMemory) -> list[str]:
    """Extract source_refs from candidate using getattr for forward-compat."""
    return getattr(candidate, "source_refs", None) or []


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Key-match lookup
# ---------------------------------------------------------------------------


def _find_active_by_key(
    conn: psycopg.Connection[Any],
    user_id: str,
    memory_key: str,
) -> dict[str, Any] | None:
    """
    Return the active canonical memory row for (user_id, memory_key), or None.
    """
    sql = """
        SELECT id::text, memory_key, value, confidence, event_time, embedding
        FROM canonical_memories
        WHERE user_id = %s
          AND memory_key = %s
          AND status = 'active'
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, [user_id, memory_key])
        row = cur.fetchone()
    if row is None:
        return None
    row_id, key, value, confidence, event_time, embedding = row
    return {
        "id": str(row_id),
        "memory_key": key,
        "value": value,
        "confidence": float(confidence) if confidence is not None else 0.8,
        "event_time": event_time,
        "embedding": embedding,
    }


# ---------------------------------------------------------------------------
# Similarity fallback
# ---------------------------------------------------------------------------


def _find_similar_active(
    conn: psycopg.Connection[Any],
    user_id: str,
    query_embedding: list[float],
    threshold: float,
) -> str | None:
    """
    Return the id of any active canonical memory with cosine similarity >=
    threshold to query_embedding, or None.

    Falls back to in-Python cosine computation when pgvector is unavailable.
    """
    sql = """
        SELECT id::text, embedding
        FROM canonical_memories
        WHERE user_id = %s
          AND status = 'active'
          AND embedding IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, [user_id])
        rows = cur.fetchall()

    for row_id, embedding in rows:
        if embedding is None:
            continue
        # embedding may come back as a list or a pgvector type
        if not isinstance(embedding, list):
            try:
                embedding = list(embedding)
            except Exception:
                continue
        sim = _cosine_similarity(query_embedding, embedding)
        if sim >= threshold:
            return str(row_id)
    return None


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------


def resolve_conflict(
    conn: psycopg.Connection[Any],
    user_id: str,
    candidate: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    similarity_threshold: float = 0.85,
) -> ResolutionResult:
    """
    Determine the action required to integrate *candidate* into the
    canonical memory store for *user_id*.

    Policy order
    ------------
    1. Exact-key match by (user_id, memory_key) on active canonical memories.
       a. Prefer newer event_time.
       b. If time equal/unknown, prefer higher confidence.
       c. Otherwise NOOP (candidate is not better).
    2. No key match → similarity fallback via embedding cosine to prevent
       obvious duplicates.  NOOP if a near-duplicate exists.
    3. Truly new → ADD.
    """
    memory_key = _get_candidate_key(candidate)
    candidate_value = _get_candidate_value(candidate)
    candidate_event_time = _get_candidate_event_time(candidate)
    candidate_confidence = candidate.confidence

    # --- Step 1: exact key match ---
    if memory_key:
        existing = _find_active_by_key(conn, user_id, memory_key)
        if existing is not None:
            existing_id = existing["id"]
            existing_time: datetime | None = existing["event_time"]
            existing_confidence: float = existing["confidence"]

            # Compare by event_time first
            if candidate_event_time is not None and existing_time is not None:
                if candidate_event_time > existing_time:
                    return ResolutionResult(
                        action=SUPERSEDE,
                        candidate=candidate,
                        existing_id=existing_id,
                        reason="Candidate has newer event_time",
                    )
                if candidate_event_time < existing_time:
                    return ResolutionResult(
                        action=NOOP,
                        candidate=candidate,
                        existing_id=existing_id,
                        reason="Existing row has newer event_time",
                    )
                # Equal times → fall through to confidence comparison

            # Compare by confidence
            if candidate_confidence > existing_confidence:
                return ResolutionResult(
                    action=SUPERSEDE,
                    candidate=candidate,
                    existing_id=existing_id,
                    reason="Candidate has higher confidence",
                )
            if candidate_confidence < existing_confidence:
                return ResolutionResult(
                    action=NOOP,
                    candidate=candidate,
                    existing_id=existing_id,
                    reason="Existing row has higher confidence",
                )

            # Same confidence and same/unknown time
            if candidate_value == existing["value"]:
                return ResolutionResult(
                    action=NOOP,
                    candidate=candidate,
                    existing_id=existing_id,
                    reason="Identical value already active",
                )

            # Different value but equal priority → UPDATE
            return ResolutionResult(
                action=UPDATE,
                candidate=candidate,
                existing_id=existing_id,
                reason="Same key, equal priority, different value",
            )

    # --- Step 2: similarity fallback ---
    query_embedding = embedding_provider.embed_query(candidate_value)
    similar_id = _find_similar_active(conn, user_id, query_embedding, similarity_threshold)
    if similar_id is not None:
        return ResolutionResult(
            action=NOOP,
            candidate=candidate,
            existing_id=similar_id,
            reason=f"Near-duplicate found (cosine >= {similarity_threshold})",
        )

    # --- Step 3: genuinely new ---
    return ResolutionResult(
        action=ADD,
        candidate=candidate,
        existing_id=None,
        reason="No existing match found",
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def apply_resolution(
    conn: psycopg.Connection[Any],
    user_id: str,
    result: ResolutionResult,
    embedding: list[float] | None = None,
) -> str | None:
    """
    Execute *result* against the database.

    Returns the UUID of the affected row (new or updated), or None for NOOP.

    Parameters
    ----------
    conn:      Open psycopg3 connection (autocommit handled by caller).
    user_id:   Tenant identifier.
    result:    Resolution result from resolve_conflict().
    embedding: Pre-computed embedding for the candidate value, if available.
               Used to populate the embedding column on INSERT.
    """
    if result.action == NOOP:
        return None

    candidate = result.candidate
    memory_key = _get_candidate_key(candidate) or str(uuid.uuid4())
    value = _get_candidate_value(candidate)
    event_time = _get_candidate_event_time(candidate)
    metadata = candidate.metadata or {}
    source_refs = _get_candidate_source_refs(candidate)
    if source_refs:
        metadata = {**metadata, "source_refs": source_refs}
    confidence = candidate.confidence
    memory_type = candidate.memory_type

    # Serialise embedding list to a string pgvector understands
    embedding_val: Any = None
    if embedding is not None:
        embedding_val = embedding  # psycopg adapters handle list[float]→vector

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
                [
                    new_id,
                    user_id,
                    memory_key,
                    value,
                    memory_type,
                    confidence,
                    event_time,
                    embedding_val,
                    json.dumps(metadata),
                ],
            )
            return new_id

        if result.action == UPDATE:
            existing_id = result.existing_id
            cur.execute(
                """
                UPDATE canonical_memories
                SET value = %s,
                    confidence = %s,
                    event_time = %s,
                    metadata = %s,
                    updated_at = now()
                WHERE id = %s::uuid
                """,
                [value, confidence, event_time, json.dumps(metadata), existing_id],
            )
            return existing_id

        if result.action == SUPERSEDE:
            existing_id = result.existing_id
            new_id = str(uuid.uuid4())

            # Mark old row superseded
            cur.execute(
                """
                UPDATE canonical_memories
                SET status = 'superseded', updated_at = now()
                WHERE id = %s::uuid
                """,
                [existing_id],
            )

            # Insert new active row with lineage
            cur.execute(
                """
                INSERT INTO canonical_memories
                    (id, user_id, memory_key, value, memory_type, confidence,
                     event_time, status, supersedes_memory_id, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s::uuid, %s, %s)
                """,
                [
                    new_id,
                    user_id,
                    memory_key,
                    value,
                    memory_type,
                    confidence,
                    event_time,
                    existing_id,
                    embedding_val,
                    json.dumps(metadata),
                ],
            )
            return new_id

    return None
