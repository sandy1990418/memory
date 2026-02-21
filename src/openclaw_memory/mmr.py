"""
Maximal Marginal Relevance (MMR) re-ranking.
Mirrors: src/memory/mmr.ts
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class MMRConfig:
    enabled: bool = False
    lambda_: float = 0.7  # 0 = max diversity, 1 = max relevance


DEFAULT_MMR_CONFIG = MMRConfig(enabled=False, lambda_=0.7)


def tokenize(text: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return frozenset(tokens)


def jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def text_similarity(content_a: str, content_b: str) -> float:
    return jaccard_similarity(tokenize(content_a), tokenize(content_b))


def compute_mmr_score(relevance: float, max_similarity: float, lambda_: float) -> float:
    return lambda_ * relevance - (1.0 - lambda_) * max_similarity


T = TypeVar("T")


def mmr_rerank(items: list[T], *, score_fn, content_fn, config: MMRConfig | None = None) -> list[T]:
    """
    Generic MMR re-ranking.
    score_fn(item) -> float
    content_fn(item) -> str
    """
    cfg = config or DEFAULT_MMR_CONFIG
    if not cfg.enabled or len(items) <= 1:
        return list(items)

    lam = max(0.0, min(1.0, cfg.lambda_))
    if lam == 1.0:
        return sorted(items, key=score_fn, reverse=True)

    # Pre-tokenize
    token_cache: dict[int, frozenset[str]] = {
        i: tokenize(content_fn(item)) for i, item in enumerate(items)
    }

    scores = [score_fn(item) for item in items]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score

    def normalize(s: float) -> float:
        return (s - min_score) / score_range if score_range > 0 else 1.0

    selected_indices: list[int] = []
    remaining = set(range(len(items)))

    while remaining:
        best_idx = -1
        best_mmr = float("-inf")

        for idx in remaining:
            norm_rel = normalize(scores[idx])
            max_sim = 0.0
            for sel_idx in selected_indices:
                sim = jaccard_similarity(token_cache[idx], token_cache[sel_idx])
                if sim > max_sim:
                    max_sim = sim
            mmr = compute_mmr_score(norm_rel, max_sim, lam)
            if mmr > best_mmr or (mmr == best_mmr and scores[idx] > scores[best_idx]):
                best_mmr = mmr
                best_idx = idx

        if best_idx < 0:
            break
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [items[i] for i in selected_indices]


def apply_mmr_to_hybrid_results(results: list[dict], config: MMRConfig | None = None) -> list[dict]:
    """
    Apply MMR re-ranking to hybrid search result dicts.
    Each dict must have: score, snippet, path, start_line.
    Mirrors: mmr.ts::applyMMRToHybridResults
    """
    if not results:
        return results
    return mmr_rerank(
        results,
        score_fn=lambda r: r["score"],
        content_fn=lambda r: r.get("snippet", ""),
        config=config,
    )
