"""
Hybrid BM25 + vector search merging for PostgreSQL-backed memory.

Merges vector (pgvector cosine) and keyword (tsvector BM25) results into a
unified ranked list, weighted by configurable vector_weight and text_weight.
"""

from __future__ import annotations

import re
from typing import Any


def build_fts_query(raw: str) -> str | None:
    """Build a PostgreSQL plainto_tsquery-compatible query from raw input."""
    tokens = re.findall(r"[\w]+", raw, re.UNICODE)
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        return None
    return " & ".join(tokens)


def merge_hybrid_results(
    *,
    vector: list[dict[str, Any]],
    keyword: list[dict[str, Any]],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Merge vector and keyword search results into a single ranked list.

    Each input dict should have at minimum: id, snippet, and either
    vector_score or text_score. Output dicts carry a combined ``score``.

    Args:
        vector:        Results from pgvector cosine search.
        keyword:       Results from tsvector keyword search.
        vector_weight: Weight for vector similarity scores.
        text_weight:   Weight for BM25 keyword scores.

    Returns:
        Merged and sorted list of result dicts with combined ``score``.
    """
    by_id: dict[str, dict[str, Any]] = {}

    for r in vector:
        rid = r["id"]
        by_id[rid] = {
            **r,
            "vector_score": r.get("vector_score", r.get("score", 0.0)),
            "text_score": 0.0,
        }

    for r in keyword:
        rid = r["id"]
        existing = by_id.get(rid)
        if existing:
            existing["text_score"] = r.get("text_score", r.get("score", 0.0))
        else:
            by_id[rid] = {
                **r,
                "vector_score": 0.0,
                "text_score": r.get("text_score", r.get("score", 0.0)),
            }

    merged: list[dict[str, Any]] = []
    for entry in by_id.values():
        combined = (
            vector_weight * entry["vector_score"]
            + text_weight * entry["text_score"]
        )
        merged.append({**entry, "score": combined})

    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged
