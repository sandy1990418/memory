"""
Unified search across the three-layer memory system.

Orchestrates search across L1 (working), L2 (episodic), and L3 (semantic)
memories, applying hybrid merge and ranking pipeline. Implements
claude-mem-style progressive disclosure with three tiers:

  Tier 1 (compact):   IDs + scores + short titles  (~50 tokens/result)
  Tier 2 (timeline):  Temporal context around a memory
  Tier 3 (detail):    Full content for selected IDs
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import psycopg

from ...core.embeddings import EmbeddingProvider
from ...core.types import MemoryContext, MemoryIndex, MemorySearchResult
from ...db import queries
from .hybrid import merge_hybrid_results
from .ranking import (
    MMRConfig,
    TemporalDecayConfig,
    apply_ranking_pipeline,
)

# ---------------------------------------------------------------------------
# Query expansion (keyword extraction)
# ---------------------------------------------------------------------------

_STOP_WORDS_EN: frozenset[str] = frozenset([
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over",
    "and", "or", "but", "if", "then", "because", "as", "while",
    "when", "where", "what", "which", "who", "how", "why",
    "yesterday", "today", "tomorrow", "earlier", "later", "recently",
    "ago", "just", "now",
    "thing", "things", "stuff", "something", "anything", "everything", "nothing",
    "please", "help", "find", "show", "get", "tell", "give",
])

_STOP_WORDS_ZH: frozenset[str] = frozenset([
    "我", "我们", "你", "你们", "他", "她", "它", "他们",
    "这", "那", "这个", "那个", "这些", "那些",
    "的", "了", "着", "过", "得", "地", "吗", "呢", "吧", "啊", "呀", "嘛", "啦",
    "是", "有", "在", "被", "把", "给", "让", "用", "到", "去", "来",
    "做", "说", "看", "找", "想", "要", "能", "会", "可以",
    "和", "与", "或", "但", "但是", "因为", "所以", "如果", "虽然",
    "而", "也", "都", "就", "还", "又", "再", "才", "只",
    "之前", "以前", "之后", "以后", "刚才", "现在",
    "昨天", "今天", "明天", "最近",
    "东西", "事情", "事", "什么", "哪个", "哪些", "怎么", "为什么", "多少",
    "请", "帮", "帮忙", "告诉",
])

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a conversational query (EN + ZH)."""
    normalized = query.lower().strip()
    segments = re.split(r"[\s.,!?;:\"'()\[\]{}<>/\\|@#$%^&*=+~`]+", normalized)
    tokens: list[str] = []

    for seg in segments:
        if not seg:
            continue
        if _CJK_RE.search(seg):
            chars = [c for c in seg if _CJK_RE.match(c)]
            tokens.extend(chars)
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
        else:
            tokens.append(seg)

    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if token in _STOP_WORDS_EN or token in _STOP_WORDS_ZH:
            continue
        if not token or (token.isascii() and len(token) < 3):
            continue
        if token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)

    return keywords


# ---------------------------------------------------------------------------
# Tier 1: Compact search
# ---------------------------------------------------------------------------


def search_compact(
    conn: psycopg.Connection[Any],
    user_id: str,
    query: str,
    embedding_provider: EmbeddingProvider,
    *,
    limit: int = 20,
    tables: list[str] | None = None,
) -> list[MemoryIndex]:
    """
    Tier 1: Return lightweight index entries across memory layers.

    Each result is ~50 tokens — suitable for showing in a search index
    before the user selects specific memories to expand.
    """
    query_vec = embedding_provider.embed_query(query)
    raw = queries.search_compact(conn, user_id, query_vec, limit=limit, tables=tables)
    return [
        MemoryIndex(
            id=r["id"],
            title=r["title"],
            memory_type=r.get("memory_type", ""),
            score=r["score"],
            created_at=str(r.get("created_at", "")),
            source=r.get("source", ""),
        )
        for r in raw
    ]


# ---------------------------------------------------------------------------
# Tier 2: Timeline context
# ---------------------------------------------------------------------------


def search_timeline(
    conn: psycopg.Connection[Any],
    user_id: str,
    memory_id: str,
    *,
    depth_before: int = 3,
    depth_after: int = 3,
) -> MemoryContext | None:
    """Tier 2: Get temporal context around a specific memory."""
    raw = queries.get_timeline(
        conn, user_id, memory_id,
        depth_before=depth_before,
        depth_after=depth_after,
    )
    if raw is None:
        return None
    return MemoryContext(
        id=raw["id"],
        content=raw["content"],
        memory_type=raw.get("memory_type", ""),
        score=raw.get("score", 1.0),
        created_at=str(raw.get("created_at", "")),
        source=raw.get("source", ""),
        neighbors=raw.get("neighbors", []),
    )


# ---------------------------------------------------------------------------
# Tier 3: Full content by IDs
# ---------------------------------------------------------------------------


def search_detail(
    conn: psycopg.Connection[Any],
    memory_ids: list[str],
) -> list[MemorySearchResult]:
    """Tier 3: Fetch full memory content for selected IDs."""
    raw = queries.get_memories_by_ids(conn, memory_ids)
    return [
        MemorySearchResult(
            id=r["id"],
            content=r["content"],
            memory_type=r.get("memory_type", ""),
            score=1.0,
            created_at=str(r.get("created_at", "")),
            source=r.get("source", ""),
            metadata=r.get("metadata", {}),
        )
        for r in raw
    ]


# ---------------------------------------------------------------------------
# Full search (hybrid + ranking pipeline)
# ---------------------------------------------------------------------------


def search(
    conn: psycopg.Connection[Any],
    user_id: str,
    query: str,
    embedding_provider: EmbeddingProvider,
    *,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    temporal_decay: TemporalDecayConfig | None = None,
    mmr: MMRConfig | None = None,
    llm_fn: Callable[[str], str] | None = None,
    top_k: int = 10,
    tables: list[str] | None = None,
) -> list[MemorySearchResult]:
    """
    Full search pipeline: vector + keyword -> hybrid merge -> ranking.

    Searches across L2 (episodic) and L3 (semantic/canonical) layers,
    applies temporal decay, MMR, and optional LLM re-ranking.

    Args:
        conn:               PostgreSQL connection.
        user_id:            User to search for.
        query:              Natural language query.
        embedding_provider: Provider for query embedding.
        vector_weight:      Weight for vector similarity.
        text_weight:        Weight for BM25 keyword matching.
        temporal_decay:     Temporal decay config (None uses defaults).
        mmr:                MMR config (None uses defaults).
        llm_fn:             Optional LLM callable for re-ranking.
        top_k:              Max results to return.
        tables:             Tables to search (default: all).

    Returns:
        Ranked list of MemorySearchResult.
    """
    target_tables = tables or ["episodic_memories", "semantic_memories", "canonical_memories"]
    query_vec = embedding_provider.embed_query(query)

    # Gather vector + keyword results from all target tables
    all_vector: list[dict[str, Any]] = []
    all_keyword: list[dict[str, Any]] = []

    for tbl in target_tables:
        all_vector.extend(queries.search_vector(
            conn, user_id, query_vec, limit=top_k * 2, table=tbl,
        ))
        all_keyword.extend(queries.search_keyword(
            conn, user_id, query, limit=top_k * 2, table=tbl,
        ))

    # Hybrid merge
    merged = merge_hybrid_results(
        vector=all_vector,
        keyword=all_keyword,
        vector_weight=vector_weight,
        text_weight=text_weight,
    )

    # Apply ranking pipeline
    ranked = apply_ranking_pipeline(
        merged,
        temporal_decay=temporal_decay,
        mmr=mmr,
        llm_fn=llm_fn,
        query=query,
        top_k=top_k,
    )

    return [
        MemorySearchResult(
            id=r["id"],
            content=r.get("snippet", r.get("content", "")),
            memory_type=r.get("memory_type", ""),
            score=r["score"],
            created_at=str(r.get("created_at", "")),
            source=r.get("source", ""),
            metadata=r.get("metadata", {}),
        )
        for r in ranked
    ]
