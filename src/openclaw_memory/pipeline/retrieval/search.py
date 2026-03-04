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
from ...utils.similarity import jaccard_similarity, tokenize
from .hybrid import merge_hybrid_results
from .ranking import (
    MMRConfig,
    TemporalDecayConfig,
    apply_ranking_pipeline,
)

# Default table quotas: canonical gets the lion's share, episodic supplements.
_DEFAULT_TABLES = ["canonical_memories", "episodic_memories"]
_CANONICAL_RATIO = 0.6  # overridden by config via canonical_ratio param

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


def _dedup_by_content(
    results: list[dict[str, Any]],
    similarity_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Remove near-duplicate results across tables, keeping first (higher-priority)."""
    kept: list[dict[str, Any]] = []
    for r in results:
        r_tokens = tokenize(r.get("snippet", r.get("content", "")))
        is_dup = any(
            jaccard_similarity(r_tokens, tokenize(k.get("snippet", k.get("content", ""))))
            >= similarity_threshold
            for k in kept
        )
        if not is_dup:
            kept.append(r)
    return kept


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
    canonical_ratio: float = _CANONICAL_RATIO,
) -> list[MemorySearchResult]:
    """
    Quota-based layered search: each layer gets an independent quota.

    Default searches canonical_memories (60% quota) + episodic_memories
    (40% quota). Each layer runs its own hybrid merge + ranking pipeline,
    then results are cross-layer deduped and merged.

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
        tables:             Tables to search (default: canonical + episodic).
        canonical_ratio:    Fraction of top_k allocated to canonical_memories.

    Returns:
        Ranked list of MemorySearchResult.
    """
    target_tables = tables or _DEFAULT_TABLES
    query_vec = embedding_provider.embed_query(query)

    # Compute per-table quotas
    n_tables = len(target_tables)
    canonical_quota = max(1, int(top_k * canonical_ratio)) if n_tables > 1 else top_k
    other_quota = max(1, top_k - canonical_quota) if n_tables > 1 else top_k

    all_ranked: list[dict[str, Any]] = []

    for tbl in target_tables:
        quota = canonical_quota if tbl == "canonical_memories" else other_quota
        fetch_limit = quota * 3  # over-fetch for ranking headroom

        vec_hits = queries.search_vector(
            conn, user_id, query_vec, limit=fetch_limit, table=tbl,
        )
        kw_hits = queries.search_keyword(
            conn, user_id, query, limit=fetch_limit, table=tbl,
        )

        merged = merge_hybrid_results(
            vector=vec_hits,
            keyword=kw_hits,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )

        ranked = apply_ranking_pipeline(
            merged,
            temporal_decay=temporal_decay,
            mmr=mmr,
            llm_fn=llm_fn,
            query=query,
            top_k=quota,
        )
        all_ranked.extend(ranked)

    # Cross-layer content dedup (canonical added first → kept on ties)
    if len(target_tables) > 1:
        all_ranked = _dedup_by_content(all_ranked)

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
        for r in all_ranked[:top_k]
    ]
