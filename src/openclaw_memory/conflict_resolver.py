"""
Action-based conflict resolution for canonical memories.

Compares an incoming ExtractedMemory candidate against existing canonical
memories in the database and decides how to handle it:

  ADD       — new key, no existing match → insert fresh row
  UPDATE    — same key, same confidence/time → in-place value update
  SUPERSEDE — same key, better candidate → mark old row inactive, insert new
  DELETE    — LLM decides an existing memory should be removed
  NOOP      — candidate is stale or duplicate → skip entirely

Supports two modes:
  - Rule-based (resolve_conflict_rules): deterministic, no LLM required
  - LLM-based (resolve_conflict_llm): Mem0-style CRUD via tool-calling
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .embeddings import EmbeddingProvider
from .extraction import ExtractedMemory

if TYPE_CHECKING:
    import psycopg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

ADD = "ADD"
UPDATE = "UPDATE"
SUPERSEDE = "SUPERSEDE"
DELETE = "DELETE"
NOOP = "NOOP"

# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------


@dataclass
class ResolutionResult:
    """Outcome of conflict resolution for a single candidate memory."""

    action: str  # ADD / UPDATE / SUPERSEDE / DELETE / NOOP
    candidate: ExtractedMemory
    existing_id: str | None = None  # UUID of matched existing row
    reason: str = ""
    new_value: str | None = None  # For UPDATE: LLM-suggested merged value


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
# Similarity search (pgvector)
# ---------------------------------------------------------------------------


def _find_similar_active_pgvector(
    conn: psycopg.Connection[Any],
    user_id: str,
    query_embedding: list[float],
    *,
    threshold: float = 0.80,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Return top-K active canonical memories with cosine similarity >= threshold
    using pgvector's <=> operator for server-side computation.

    Returns a list of dicts with id, memory_key, value, memory_type,
    confidence, event_time, and similarity.
    """
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
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
        cur.execute(sql, [embedding_str, user_id, embedding_str, threshold,
                          embedding_str, top_k])
        rows = cur.fetchall()

    results = []
    for row in rows:
        row_id, key, value, mem_type, confidence, event_time, similarity = row
        results.append({
            "id": str(row_id),
            "memory_key": key or "",
            "value": value or "",
            "memory_type": mem_type or "",
            "confidence": float(confidence) if confidence is not None else 0.8,
            "event_time": event_time,
            "similarity": float(similarity),
        })
    return results


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
# LLM CRUD prompt
# ---------------------------------------------------------------------------

_CRUD_SYSTEM_PROMPT = """\
You are a memory manager. Your job is to decide how to handle a new memory \
given a list of existing memories.

For the new memory and existing memories provided, decide what actions to take. \
You must respond with a JSON array of action objects.

Each action object must have:
- "action": one of "ADD", "UPDATE", "DELETE", "NOOP"
- "memory_id": the ID of the existing memory to act on (required for UPDATE and DELETE, null for ADD and NOOP)
- "new_value": the updated value text (required for UPDATE, null otherwise)
- "reason": brief explanation of why this action was chosen

Rules:
- ADD: The new memory contains genuinely new information not covered by any existing memory. Return one ADD action.
- UPDATE: An existing memory covers the same topic but the new memory has updated or more complete information. Provide the merged/updated value in "new_value".
- DELETE: An existing memory is now contradicted or made obsolete by the new memory. The new memory itself should also be ADDed or used to UPDATE another memory.
- NOOP: The new memory is a duplicate or already covered by existing memories. Return one NOOP action.

If the new memory contradicts an existing memory, UPDATE the existing one with the new information (or DELETE + ADD if the topic is different enough).

Respond ONLY with a valid JSON array. No markdown fences, no extra text.\
"""


def _build_crud_prompt(
    candidate: ExtractedMemory,
    existing_memories: list[dict[str, Any]],
) -> str:
    """Build the user message for the CRUD LLM call."""
    candidate_value = _get_candidate_value(candidate)
    candidate_key = _get_candidate_key(candidate) or ""
    candidate_event_time = _get_candidate_event_time(candidate)
    event_time_str = candidate_event_time.isoformat() if candidate_event_time else "unknown"

    parts = [
        "New memory to process:",
        f"  key: {candidate_key}",
        f"  value: {candidate_value}",
        f"  type: {candidate.memory_type}",
        f"  confidence: {candidate.confidence}",
        f"  event_time: {event_time_str}",
        "",
    ]

    if existing_memories:
        parts.append("Existing memories:")
        for i, mem in enumerate(existing_memories):
            parts.append(
                f"  [{i+1}] id={mem['id']} | key={mem.get('memory_key', '')} "
                f"| value={mem.get('value', '')} | type={mem.get('memory_type', '')} "
                f"| confidence={mem.get('confidence', '')} "
                f"| similarity={mem.get('similarity', ''):.3f}"
            )
    else:
        parts.append("Existing memories: none")

    parts.append("")
    parts.append("Respond with a JSON array of actions.")
    return "\n".join(parts)


def _parse_crud_response(
    response: str,
    candidate: ExtractedMemory,
    existing_memories: list[dict[str, Any]],
) -> list[ResolutionResult]:
    """Parse the LLM CRUD response into a list of ResolutionResult."""
    # Strip markdown fences if present
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        actions = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM CRUD response, falling back to ADD: %s", text[:200])
        return [ResolutionResult(
            action=ADD,
            candidate=candidate,
            existing_id=None,
            reason="LLM response parse failure, defaulting to ADD",
        )]

    if not isinstance(actions, list):
        actions = [actions]

    # Build lookup of existing memory IDs for validation
    existing_ids = {mem["id"] for mem in existing_memories}

    results: list[ResolutionResult] = []
    for act in actions:
        if not isinstance(act, dict):
            continue

        action_str = str(act.get("action", "")).upper().strip()
        memory_id = act.get("memory_id")
        new_value = act.get("new_value")
        reason = str(act.get("reason", ""))

        if memory_id is not None:
            memory_id = str(memory_id)

        if action_str == "ADD":
            results.append(ResolutionResult(
                action=ADD,
                candidate=candidate,
                existing_id=None,
                reason=reason or "LLM decided to add new memory",
            ))
        elif action_str == "UPDATE" and memory_id and memory_id in existing_ids:
            results.append(ResolutionResult(
                action=UPDATE,
                candidate=candidate,
                existing_id=memory_id,
                reason=reason or "LLM decided to update existing memory",
                new_value=str(new_value) if new_value else None,
            ))
        elif action_str == "DELETE" and memory_id and memory_id in existing_ids:
            results.append(ResolutionResult(
                action=DELETE,
                candidate=candidate,
                existing_id=memory_id,
                reason=reason or "LLM decided to delete existing memory",
            ))
        elif action_str == "NOOP":
            results.append(ResolutionResult(
                action=NOOP,
                candidate=candidate,
                existing_id=memory_id if memory_id in existing_ids else None,
                reason=reason or "LLM decided no action needed",
            ))

    if not results:
        # Empty or fully invalid response: default to ADD
        results.append(ResolutionResult(
            action=ADD,
            candidate=candidate,
            existing_id=None,
            reason="LLM returned no valid actions, defaulting to ADD",
        ))

    return results


# ---------------------------------------------------------------------------
# LLM-based resolver
# ---------------------------------------------------------------------------


def resolve_conflict_llm(
    conn: psycopg.Connection[Any],
    user_id: str,
    candidate: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    llm_fn: Callable[[str], str],
    *,
    similarity_threshold: float = 0.80,
    top_k: int = 5,
) -> list[ResolutionResult]:
    """
    LLM-based CRUD conflict resolution (Mem0-style).

    Pipeline:
      1. Embed the candidate memory
      2. Find top-K similar active canonical memories via pgvector
      3. Build a CRUD prompt with candidate + existing memories
      4. Call llm_fn for JSON action decisions
      5. Parse response into list[ResolutionResult]
    """
    candidate_value = _get_candidate_value(candidate)
    query_embedding = embedding_provider.embed_query(candidate_value)

    # Step 2: find similar memories via pgvector
    try:
        existing_memories = _find_similar_active_pgvector(
            conn, user_id, query_embedding,
            threshold=similarity_threshold,
            top_k=top_k,
        )
    except Exception:
        # pgvector query failed (e.g. extension not installed), fall back to empty
        logger.warning("pgvector similarity search failed, proceeding with no existing memories")
        existing_memories = []

    # Step 3: build prompt
    user_prompt = _build_crud_prompt(candidate, existing_memories)
    full_prompt = _CRUD_SYSTEM_PROMPT + "\n\n" + user_prompt

    # Step 4: call LLM
    try:
        response = llm_fn(full_prompt)
    except Exception:
        logger.warning("LLM call failed in CRUD resolver, falling back to ADD")
        return [ResolutionResult(
            action=ADD,
            candidate=candidate,
            existing_id=None,
            reason="LLM call failed, defaulting to ADD",
        )]

    # Step 5: parse response
    return _parse_crud_response(response, candidate, existing_memories)


# ---------------------------------------------------------------------------
# Rule-based resolver (original logic)
# ---------------------------------------------------------------------------


def resolve_conflict_rules(
    conn: psycopg.Connection[Any],
    user_id: str,
    candidate: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    similarity_threshold: float = 0.85,
) -> ResolutionResult:
    """
    Deterministic rule-based conflict resolution (original logic).

    Policy order
    ------------
    1. Exact-key match by (user_id, memory_key) on active canonical memories.
       a. Prefer newer event_time.
       b. If time equal/unknown, prefer higher confidence.
       c. Otherwise NOOP (candidate is not better).
    2. No key match -> similarity fallback via embedding cosine to prevent
       obvious duplicates.  NOOP if a near-duplicate exists.
    3. Truly new -> ADD.
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
                # Equal times -> fall through to confidence comparison

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

            # Different value but equal priority -> UPDATE
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
# Unified dispatcher
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
    """
    Determine the action(s) required to integrate *candidate* into the
    canonical memory store for *user_id*.

    When llm_fn is provided, uses LLM-based CRUD resolution which can
    return multiple actions (e.g. UPDATE one memory AND DELETE another).

    When llm_fn is None, falls back to deterministic rule-based resolution
    (single result wrapped in a list for consistent return type).
    """
    if llm_fn is not None:
        return resolve_conflict_llm(
            conn, user_id, candidate, embedding_provider, llm_fn,
            similarity_threshold=similarity_threshold,
        )
    return [resolve_conflict_rules(
        conn, user_id, candidate, embedding_provider,
        similarity_threshold=similarity_threshold,
    )]


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

    Returns the UUID of the affected row (new or updated), or None for NOOP/DELETE.

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

    if result.action == DELETE:
        existing_id = result.existing_id
        if existing_id:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE canonical_memories
                    SET status = 'superseded', updated_at = now()
                    WHERE id = %s::uuid
                    """,
                    [existing_id],
                )
        return None

    candidate = result.candidate
    memory_key = _get_candidate_key(candidate) or str(uuid.uuid4())
    value = _get_candidate_value(candidate)
    # If LLM provided a new_value for UPDATE, use it
    if result.action == UPDATE and result.new_value:
        value = result.new_value
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
        embedding_val = embedding  # psycopg adapters handle list[float]->vector

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
