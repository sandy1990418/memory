"""
Mock embedding provider for tests.
Produces deterministic, low-dimensional vectors based on word frequency.
"""
from __future__ import annotations

import math


# Vocabulary used for deterministic embeddings (alpha-side vs beta-side)
_ALPHA_WORDS = frozenset([
    "memory", "index", "chunk", "search", "file", "hash", "embed",
    "vector", "cosine", "similarity", "query", "result", "path",
])
_BETA_WORDS = frozenset([
    "session", "daily", "note", "decision", "cli", "sync", "dirty",
    "manager", "status", "provider", "model", "database", "sqlite",
])

# Dimension of mock embeddings
MOCK_DIMS = 16


def _make_vector(text: str) -> list[float]:
    """
    Produce a deterministic MOCK_DIMS-dimensional unit vector.

    Dimensions 0..7  respond to alpha-side words.
    Dimensions 8..15 respond to beta-side words.
    Within each group the weight is spread across dims by word-character sum.
    """
    words = text.lower().split()
    vec = [0.0] * MOCK_DIMS

    for word in words:
        char_sum = sum(ord(c) for c in word)
        if word in _ALPHA_WORDS:
            slot = char_sum % 8            # dims 0-7
            vec[slot] += 1.0
        elif word in _BETA_WORDS:
            slot = 8 + (char_sum % 8)     # dims 8-15
            vec[slot] += 1.0
        else:
            # Generic: spread across all dims by hash
            slot = char_sum % MOCK_DIMS
            vec[slot] += 0.5

    # Normalize to unit vector
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude < 1e-10:
        # Return a stable non-zero vector for empty/unknown text
        vec[0] = 1.0
        return vec
    return [v / magnitude for v in vec]


class MockEmbeddingProvider:
    """
    Deterministic mock embedding provider.

    Attributes
    ----------
    embed_batch_calls:
        Tracks every list of texts passed to embed_batch, for assertion.
    """

    id: str = "mock"
    model: str = "mock-v1"

    def __init__(self) -> None:
        self.embed_batch_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        return _make_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls.append(list(texts))
        return [_make_vector(t) for t in texts]
