"""
LLM-based re-ranking for memory search results (QMD-inspired).

Provides:
  - llm_rerank(query, candidates, llm_fn, top_k) — re-rank via LLM scoring
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

_RERANK_PROMPT = """\
You are a memory relevance scoring assistant.

Given a user query and a list of memory candidates, score each memory based on
how relevant it is to the query.

Query: {query}

Candidates:
{candidates}

Return a JSON array of objects with:
  "index": the 0-based index of the candidate
  "score": float 0.0–1.0 (1.0 = perfectly relevant)

Only include candidates you want to keep (score > 0).
Respond ONLY with valid JSON, no markdown fences.
"""


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates):
        snippet = c.get("snippet", "")
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)


def _parse_scores(response: str, num_candidates: int) -> dict[int, float]:
    """Parse LLM scoring response into {index: score} map."""
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if not text or text in ("[]", "null"):
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, list):
        return {}

    scores: dict[int, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= num_candidates:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scores[idx] = max(0.0, min(1.0, score))

    return scores


def llm_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    llm_fn: Callable[[str], str],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """
    Re-rank top candidates using LLM for precision.

    Only re-ranks top 20 candidates (cost control).
    Returns the same dict format as input, sorted by LLM score descending.
    Falls back to original order if LLM fails.

    Args:
        query:      User search query.
        candidates: List of result dicts (must have 'snippet' field).
        llm_fn:     Callable that takes a prompt string and returns a string response.
        top_k:      Number of results to return after re-ranking.
    """
    if not candidates:
        return []

    # Cost control: only re-rank top 20
    pool = candidates[:20]
    rest = candidates[20:]

    prompt = _RERANK_PROMPT.format(
        query=query,
        candidates=_format_candidates(pool),
    )

    try:
        response = llm_fn(prompt)
        scores = _parse_scores(response, len(pool))
    except Exception:
        # Graceful fallback: return original order sliced to top_k
        return candidates[:top_k]

    # Assign LLM scores; fall back to original position-based score for unscored items
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, item in enumerate(pool):
        llm_score = scores.get(i)
        if llm_score is None:
            # Not included in LLM output → low priority, use original score * 0.1
            fallback = item.get("score", 0.0) * 0.1
            scored.append((fallback, i, item))
        else:
            scored.append((llm_score, i, item))

    # Sort by LLM score descending, then by original index for ties
    scored.sort(key=lambda t: (-t[0], t[1]))

    reranked = [item for _, _, item in scored]

    # Append the un-pooled rest at the end (they are already low-scored)
    combined = reranked + rest

    return combined[:top_k]
