"""
Episodic -> Semantic promotion logic.

Promotes old event-type canonical memories into semantic facts
via LLM abstraction, preventing valuable patterns from decaying.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import psycopg

from ..core.embeddings import EmbeddingProvider
from ..utils.text import parse_llm_json

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


def promote_events_to_semantic(
    conn: psycopg.Connection[Any],
    user_id: str,
    embedding_provider: EmbeddingProvider,
    llm_fn: Callable[[str], str],
    *,
    age_days: int = 90,
) -> int:
    """
    Promote old event-type canonical memories to semantic facts.

    Returns count of promoted memories.
    """
    sql = """
        SELECT id::text, memory_key, value, memory_type
        FROM canonical_memories
        WHERE user_id = %s AND status = 'active' AND memory_type = 'event'
          AND created_at < now() - make_interval(days := %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id, age_days))
        rows = cur.fetchall()

    promoted = 0
    for row_id, key, value, mtype in rows:
        prompt = ABSTRACTION_PROMPT.format(
            memory=value or "", memory_type=mtype or "event",
        )
        parsed = parse_llm_json(llm_fn(prompt))
        if not parsed or not parsed.get("content"):
            continue

        new_id = str(uuid.uuid4())
        embedding = embedding_provider.embed_query(parsed["content"])

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE canonical_memories SET status = 'superseded', "
                "updated_at = now() WHERE id = %s::uuid",
                (str(row_id),),
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
                    parsed.get("memory_key", key or ""),
                    parsed["content"],
                    parsed.get("memory_type", "fact"),
                    parsed.get("confidence", 0.85),
                    embedding,
                    json.dumps({"source": "episodic_promotion"}),
                    [str(row_id)],
                ),
            )
        promoted += 1
    return promoted
