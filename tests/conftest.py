"""
Shared test helpers for openclaw-memory tests.
"""

from __future__ import annotations

from tests.helpers.embeddings_mock import MockEmbeddingProvider


def make_mock_embedding_provider() -> MockEmbeddingProvider:
    """Create a fresh MockEmbeddingProvider with empty call tracking."""
    return MockEmbeddingProvider()
