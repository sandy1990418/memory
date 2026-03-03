"""
Deduplication logic for the three-layer memory system.

Provides embedding-based deduplication to prevent storing
near-duplicate memories across episodic and semantic tables.
"""

from __future__ import annotations

from typing import Any

import psycopg

from ..utils.similarity import cosine_similarity


def is_duplicate(
    conn: psycopg.Connection[Any],
    user_id: str,
    embedding: list[float],
    *,
    threshold: float = 0.95,
    table: str = "episodic_memories",
) -> bool:
    """
    Check if a memory with this embedding already exists.

    Returns True if any active memory in the table has cosine similarity
    above the threshold.
    """
    sql = f"""
        SELECT embedding
        FROM {table}
        WHERE user_id = %s AND embedding IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id,))
        for (stored_emb,) in cur:
            if stored_emb is None:
                continue
            stored = list(stored_emb) if not isinstance(stored_emb, list) else stored_emb
            if cosine_similarity(embedding, stored) >= threshold:
                return True
    return False


def dedup_memories(
    conn: psycopg.Connection[Any],
    user_id: str,
    embeddings_with_ids: list[tuple[str, list[float]]],
    *,
    threshold: float = 0.95,
    table: str = "canonical_memories",
) -> list[str]:
    """
    Find duplicate memory IDs within a list of (id, embedding) pairs.

    Returns a list of IDs that are duplicates of earlier entries
    (keeps the first occurrence).
    """
    duplicates: list[str] = []
    seen: list[tuple[str, list[float]]] = []

    for mem_id, emb in embeddings_with_ids:
        is_dup = False
        for _, seen_emb in seen:
            if cosine_similarity(emb, seen_emb) >= threshold:
                is_dup = True
                break
        if is_dup:
            duplicates.append(mem_id)
        else:
            seen.append((mem_id, emb))

    return duplicates
