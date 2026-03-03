"""
L3: Semantic memory — consolidated, evergreen knowledge.

Stores long-term facts, preferences, and decisions that have been
promoted from episodic memory or directly extracted with high
confidence. No temporal decay is applied to semantic memories.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

from ..core.embeddings import EmbeddingProvider, coerce_pgvector_dims, embedding_to_pg_literal
from ..core.types import ExtractedMemory


def store_semantic(
    conn: psycopg.Connection[Any],
    user_id: str,
    memory: ExtractedMemory,
    embedding_provider: EmbeddingProvider,
) -> str:
    """Store an extracted memory in the semantic_memories table."""
    embedding = coerce_pgvector_dims(embedding_provider.embed_query(memory.content))
    emb_literal = embedding_to_pg_literal(embedding)

    sql = """
        INSERT INTO semantic_memories
            (user_id, content, embedding, memory_type, metadata)
        VALUES (%s, %s, %s::vector, %s, %s)
        RETURNING id::text
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            user_id,
            memory.content,
            emb_literal,
            memory.memory_type,
            json.dumps(memory.metadata),
        ))
        row = cur.fetchone()
    return str(row[0]) if row else ""


def table_for_memory(memory: ExtractedMemory) -> str:
    """Determine target table based on memory type."""
    if memory.memory_type == "event":
        return "episodic_memories"
    return "semantic_memories"
