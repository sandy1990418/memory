"""
Post-retrieval ranking pipeline: temporal decay, MMR, and LLM re-ranking.

Consolidates the three old modules (decay.py, mmr.py, rerank.py) into a
single ranking module with a unified ``apply_ranking_pipeline`` entry point.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from ...utils.similarity import jaccard_similarity, tokenize

# ---------------------------------------------------------------------------
# Temporal Decay
# ---------------------------------------------------------------------------

_DAY_MS = 86_400_000  # 24 * 60 * 60 * 1000


@dataclass
class TemporalDecayConfig:
    enabled: bool = True
    half_life_days: float = 30.0


def _to_decay_lambda(half_life_days: float) -> float:
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        return 0.0
    return math.log(2) / half_life_days


def calculate_temporal_decay(age_days: float, half_life_days: float) -> float:
    """Return the exponential decay multiplier for a given age."""
    lam = _to_decay_lambda(half_life_days)
    clamped = max(0.0, age_days)
    if lam <= 0 or not math.isfinite(clamped):
        return 1.0
    return math.exp(-lam * clamped)


def apply_temporal_decay(
    results: list[dict[str, Any]],
    *,
    config: TemporalDecayConfig | None = None,
    now_ms: float | None = None,
) -> list[dict[str, Any]]:
    """
    Apply temporal decay to results using their ``created_at`` field.

    Each dict must have ``score`` and ``created_at`` (datetime or ISO string).
    Returns a new list with decayed scores.
    """
    cfg = config or TemporalDecayConfig()
    if not cfg.enabled:
        return list(results)

    now_ms_actual = now_ms if now_ms is not None else time.time() * 1000.0
    out: list[dict[str, Any]] = []

    for entry in results:
        created_at = entry.get("created_at")
        if created_at is None:
            out.append(entry)
            continue

        if isinstance(created_at, str):
            try:
                ts_dt = datetime.fromisoformat(created_at)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            except (ValueError, OverflowError):
                out.append(entry)
                continue
        elif isinstance(created_at, datetime):
            ts_dt = created_at
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        else:
            out.append(entry)
            continue

        age_ms = max(0.0, now_ms_actual - ts_dt.timestamp() * 1000.0)
        age_days = age_ms / _DAY_MS
        multiplier = calculate_temporal_decay(age_days, cfg.half_life_days)
        out.append({**entry, "score": entry["score"] * multiplier})

    return out


# ---------------------------------------------------------------------------
# Maximal Marginal Relevance (MMR)
# ---------------------------------------------------------------------------


@dataclass
class MMRConfig:
    enabled: bool = True
    lambda_: float = 0.7  # 0 = max diversity, 1 = max relevance


T = TypeVar("T")


def mmr_rerank(
    items: list[T],
    *,
    score_fn: Callable[[T], float],
    content_fn: Callable[[T], str],
    config: MMRConfig | None = None,
) -> list[T]:
    """Generic MMR re-ranking using Jaccard text similarity."""
    cfg = config or MMRConfig()
    if not cfg.enabled or len(items) <= 1:
        return list(items)

    lam = max(0.0, min(1.0, cfg.lambda_))
    if lam == 1.0:
        return sorted(items, key=score_fn, reverse=True)

    token_cache = {i: tokenize(content_fn(item)) for i, item in enumerate(items)}
    scores = [score_fn(item) for item in items]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score

    def normalize(s: float) -> float:
        return (s - min_score) / score_range if score_range > 0 else 1.0

    selected: list[int] = []
    remaining = set(range(len(items)))

    while remaining:
        best_idx = -1
        best_mmr = float("-inf")

        for idx in remaining:
            norm_rel = normalize(scores[idx])
            max_sim = 0.0
            for sel_idx in selected:
                sim = jaccard_similarity(token_cache[idx], token_cache[sel_idx])
                if sim > max_sim:
                    max_sim = sim
            mmr_score = lam * norm_rel - (1.0 - lam) * max_sim
            if mmr_score > best_mmr or (mmr_score == best_mmr and scores[idx] > scores[best_idx]):
                best_mmr = mmr_score
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [items[i] for i in selected]


def apply_mmr(
    results: list[dict[str, Any]],
    config: MMRConfig | None = None,
) -> list[dict[str, Any]]:
    """Apply MMR to result dicts (must have ``score`` and ``snippet``)."""
    if not results:
        return results
    return mmr_rerank(
        results,
        score_fn=lambda r: r["score"],
        content_fn=lambda r: r.get("snippet", ""),
        config=config,
    )


# ---------------------------------------------------------------------------
# LLM Re-ranking
# ---------------------------------------------------------------------------

_RERANK_PROMPT = """\
You are a memory relevance scoring assistant.

Given a user query and a list of memory candidates, score each memory based on
how relevant it is to the query.

Query: {query}

Candidates:
{candidates}

Return a JSON array of objects with:
  "index": the 0-based index of the candidate
  "score": float 0.0-1.0 (1.0 = perfectly relevant)

Only include candidates you want to keep (score > 0).
Respond ONLY with valid JSON, no markdown fences.
"""


def llm_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    llm_fn: Callable[[str], str],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """
    Re-rank top candidates using LLM scoring (cost-controlled: top 20 only).

    Falls back to original order on LLM failure.
    """
    if not candidates:
        return []

    pool = candidates[:20]
    rest = candidates[20:]

    lines = [f"[{i}] {c.get('snippet', '')}" for i, c in enumerate(pool)]
    prompt = _RERANK_PROMPT.format(query=query, candidates="\n".join(lines))

    try:
        response = llm_fn(prompt)
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

        scores: dict[int, float] = {}
        if text and text not in ("[]", "null"):
            data = json.loads(text)
            if not isinstance(data, list):
                data = [data]
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index", -1))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(pool):
                    try:
                        scores[idx] = max(0.0, min(1.0, float(item.get("score", 0.0))))
                    except (TypeError, ValueError):
                        pass
    except Exception:
        return candidates[:top_k]

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, item in enumerate(pool):
        llm_score = scores.get(i)
        if llm_score is None:
            scored.append((item.get("score", 0.0) * 0.1, i, item))
        else:
            scored.append((llm_score, i, item))

    scored.sort(key=lambda t: (-t[0], t[1]))
    reranked = [item for _, _, item in scored]
    return (reranked + rest)[:top_k]


# ---------------------------------------------------------------------------
# Unified ranking pipeline
# ---------------------------------------------------------------------------


def apply_ranking_pipeline(
    results: list[dict[str, Any]],
    *,
    temporal_decay: TemporalDecayConfig | None = None,
    mmr: MMRConfig | None = None,
    llm_fn: Callable[[str], str] | None = None,
    query: str = "",
    top_k: int = 10,
    now_ms: float | None = None,
) -> list[dict[str, Any]]:
    """
    Apply the full post-retrieval ranking pipeline in order:
    1. Temporal decay (exponential half-life)
    2. MMR diversity re-ranking
    3. Optional LLM re-ranking (precision boost)

    Args:
        results:        Raw search results (must have ``score``, ``snippet``, ``created_at``).
        temporal_decay: Config for temporal decay. None uses defaults (enabled).
        mmr:            Config for MMR. None uses defaults (enabled).
        llm_fn:         Optional LLM callable for re-ranking.
        query:          User query (required if llm_fn is provided).
        top_k:          Max results to return.
        now_ms:         Current time in ms (for testing).

    Returns:
        Ranked and trimmed result list.
    """
    # Stage 1: Temporal decay
    ranked = apply_temporal_decay(results, config=temporal_decay, now_ms=now_ms)

    # Sort by decayed score
    ranked.sort(key=lambda r: r["score"], reverse=True)

    # Stage 2: MMR diversity
    ranked = apply_mmr(ranked, config=mmr)

    # Stage 3: Optional LLM re-ranking
    if llm_fn is not None and query:
        ranked = llm_rerank(query, ranked, llm_fn, top_k=top_k)
    else:
        ranked = ranked[:top_k]

    return ranked
