"""
L2: Episodic memory — time-decayed, recent events and facts.

Stores memories extracted from conversations with temporal decay
applied during search (half-life ~30 days by default). Events and
recent facts naturally fade unless consolidated into semantic memory.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

from ..core.embeddings import EmbeddingProvider, coerce_pgvector_dims, embedding_to_pg_literal
from ..core.types import ExtractedMemory


def store_episodic(
    conn: psycopg.Connection[Any],
    user_id: str,
    memory: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
    *,
    session_id: str | None = None,
) -> str:
    """Store an extracted memory in the episodic_memories table."""
    embedding = coerce_pgvector_dims(embedding_provider.embed_query(memory.content))
    emb_literal = embedding_to_pg_literal(embedding)

    sql = """
        INSERT INTO episodic_memories
            (user_id, session_id, content, embedding, memory_type, metadata)
        VALUES (%s, %s, %s, %s::vector, %s, %s)
        RETURNING id::text
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            user_id,
            session_id,
            memory.content,
            emb_literal,
            memory.memory_type,
            json.dumps(memory.metadata),
        ))
        row = cur.fetchone()
    return str(row[0]) if row else ""


def store_session_episode(
    conn: psycopg.Connection[Any],
    user_id: str,
    session_id: str,
    content: str,
    embedding_provider: EmbeddingProvider,
) -> str:
    """Store a raw or summarized session as a single episodic row."""
    embedding = coerce_pgvector_dims(embedding_provider.embed_query(content[:2000]))
    emb_literal = embedding_to_pg_literal(embedding)

    sql = """
        INSERT INTO episodic_memories
            (user_id, session_id, content, embedding, memory_type)
        VALUES (%s, %s, %s, %s::vector, 'session')
        RETURNING id::text
    """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id, session_id, content, emb_literal))
        row = cur.fetchone()
    return str(row[0]) if row else ""
