"""
Hybrid BM25 + vector search merging.
Mirrors: src/memory/hybrid.ts
"""
from __future__ import annotations

import re

from .mmr import MMRConfig, apply_mmr_to_hybrid_results
from .temporal_decay import TemporalDecayConfig, apply_temporal_decay_to_results


def build_fts_query(raw: str) -> str | None:
    """
    Build an FTS5 MATCH query from a raw search string.
    Mirrors: hybrid.ts::buildFtsQuery
    """
    tokens = re.findall(r"[\w]+", raw, re.UNICODE)
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        return None
    quoted = ['"' + t.replace('"', '') + '"' for t in tokens]
    return " AND ".join(quoted)


def bm25_rank_to_score(rank: float) -> float:
    """
    Convert BM25 rank (lower = better) to a 0..1 score.
    Mirrors: hybrid.ts::bm25RankToScore
    """
    normalized = max(0.0, rank) if (isinstance(rank, float) and rank == rank) else 999.0
    return 1.0 / (1.0 + normalized)


def merge_hybrid_results(
    *,
    vector: list[dict],
    keyword: list[dict],
    vector_weight: float,
    text_weight: float,
    mmr: dict | None = None,
    temporal_decay: dict | None = None,
    workspace_dir: str | None = None,
    now_ms: float | None = None,
) -> list[dict]:
    """
    Merge vector and keyword results into a unified ranked list.

    vector items: {id, path, start_line, end_line, source, snippet, vector_score}
    keyword items: {id, path, start_line, end_line, source, snippet, text_score}

    Returns dicts: {path, start_line, end_line, score, snippet, source}

    Mirrors: hybrid.ts::mergeHybridResults
    """
    by_id: dict[str, dict] = {}

    for r in vector:
        by_id[r["id"]] = {
            "id": r["id"],
            "path": r["path"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "source": r["source"],
            "snippet": r["snippet"],
            "vector_score": r.get("vector_score", r.get("score", 0.0)),
            "text_score": 0.0,
        }

    for r in keyword:
        existing = by_id.get(r["id"])
        if existing:
            existing["text_score"] = r.get("text_score", r.get("score", 0.0))
            if r.get("snippet"):
                existing["snippet"] = r["snippet"]
        else:
            by_id[r["id"]] = {
                "id": r["id"],
                "path": r["path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "source": r["source"],
                "snippet": r["snippet"],
                "vector_score": 0.0,
                "text_score": r.get("text_score", r.get("score", 0.0)),
            }

    merged = [
        {
            "path": entry["path"],
            "start_line": entry["start_line"],
            "end_line": entry["end_line"],
            "score": vector_weight * entry["vector_score"] + text_weight * entry["text_score"],
            "snippet": entry["snippet"],
            "source": entry["source"],
        }
        for entry in by_id.values()
    ]

    # Temporal decay
    td_config: TemporalDecayConfig | None = None
    if temporal_decay:
        td_config = TemporalDecayConfig(
            enabled=temporal_decay.get("enabled", False),
            half_life_days=temporal_decay.get("half_life_days", 30.0),
        )
    decayed = apply_temporal_decay_to_results(
        merged, config=td_config, workspace_dir=workspace_dir, now_ms=now_ms
    )

    # Sort by score descending
    sorted_results = sorted(decayed, key=lambda r: r["score"], reverse=True)

    # MMR
    mmr_config: MMRConfig | None = None
    if mmr and mmr.get("enabled"):
        mmr_config = MMRConfig(
            enabled=True,
            lambda_=mmr.get("lambda_", 0.7),
        )
        return apply_mmr_to_hybrid_results(sorted_results, mmr_config)

    return sorted_results
