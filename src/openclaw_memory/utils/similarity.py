"""
Shared similarity functions — single source of truth.

Consolidates duplicate cosine_similarity / jaccard implementations
that previously appeared in conflict.py, consolidation.py, dedup.py, mmr.py.
"""

from __future__ import annotations

import math
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_]+", flags=re.IGNORECASE)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


def cosine_similarity_any(a: Any, b: Any) -> float:
    """Cosine similarity accepting non-list types (e.g. pgvector)."""
    vec_a = list(a) if not isinstance(a, list) else a
    vec_b = list(b) if not isinstance(b, list) else b
    return cosine_similarity(vec_a, vec_b)


def tokenize(text: str) -> frozenset[str]:
    """Tokenize text into a set of lowercase alphanumeric tokens."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def text_similarity(content_a: str, content_b: str) -> float:
    """Text similarity via Jaccard over token sets."""
    return jaccard_similarity(tokenize(content_a), tokenize(content_b))
