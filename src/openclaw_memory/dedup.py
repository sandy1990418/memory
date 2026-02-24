"""
Deduplication logic for extracted memories using cosine similarity.

The DB connection is intentionally kept abstract — the caller passes a Protocol-
conforming object (or any object with an ``execute`` / ``fetchall`` interface).
This avoids a hard dependency on any specific PostgreSQL driver (psycopg2,
psycopg3, asyncpg shims, etc.).
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from .embeddings import EmbeddingProvider
from .extraction import ExtractedMemory


# ---------------------------------------------------------------------------
# Abstract DB protocol — callers can pass psycopg2/3, SQLite, or mocks
# ---------------------------------------------------------------------------


@runtime_checkable
class DBConnection(Protocol):
    """Minimal synchronous DB connection interface needed for dedup."""

    def execute(self, query: str, params: Any = None) -> Any: ...
    def fetchall(self) -> list[Any]: ...
    def fetchone(self) -> Any | None: ...


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def store_with_dedup(
    conn: DBConnection,
    user_id: str,
    memory: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    similarity_threshold: float = 0.85,
) -> str:
    """
    Embed *memory*, check for near-duplicates, then INSERT or UPDATE.

    Returns one of:
      "inserted" — no duplicate found, new row written
      "updated"  — existing similar memory merged/updated
      "skipped"  — would be identical (similarity == 1.0 exactly; reserved for future use)
    """
    new_embedding = embedding_provider.embed_query(memory.content)

    # Fetch all existing embeddings for this user
    cursor = conn.execute(
        "SELECT id, content, embedding FROM memories WHERE user_id = %s",
        (user_id,),
    )
    rows = cursor.fetchall()

    best_id: Any = None
    best_sim: float = -1.0

    for row in rows:
        row_id, _content, stored_embedding = row[0], row[1], row[2]

        # stored_embedding may come back as a list, string, or bytes — normalise
        if isinstance(stored_embedding, str):
            import json as _json
            stored_vec: list[float] = _json.loads(stored_embedding)
        elif isinstance(stored_embedding, (bytes, bytearray)):
            import json as _json
            stored_vec = _json.loads(stored_embedding.decode())
        else:
            stored_vec = list(stored_embedding)

        sim = _cosine_similarity(new_embedding, stored_vec)
        if sim > best_sim:
            best_sim = sim
            best_id = row_id

    import json as _json_mod

    embedding_json = _json_mod.dumps(new_embedding)

    if best_id is not None and best_sim >= similarity_threshold:
        # Merge: update content and embedding with the newer version
        conn.execute(
            "UPDATE memories "
            "SET content = %s, embedding = %s, updated_at = NOW() "
            "WHERE id = %s",
            (memory.content, embedding_json, best_id),
        )
        return "updated"

    # Insert new memory
    conn.execute(
        "INSERT INTO memories (user_id, content, memory_type, confidence, metadata, embedding) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (
            user_id,
            memory.content,
            memory.memory_type,
            memory.confidence,
            _json_mod.dumps(memory.metadata),
            embedding_json,
        ),
    )
    return "inserted"
