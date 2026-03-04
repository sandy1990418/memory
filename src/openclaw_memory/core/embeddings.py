"""
Embedding provider abstraction and adapters.

Supports OpenAI, Gemini, Voyage AI, and local llama-cpp-python.
"""

from __future__ import annotations

import math
import os
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    id: str
    model: str

    def embed_query(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

PGVECTOR_DIMS = 1536  # Default; overridden at startup via configure_pgvector_dims()


def configure_pgvector_dims(dims: int) -> None:
    """Set the global pgvector dimension from config (called at startup)."""
    global PGVECTOR_DIMS
    PGVECTOR_DIMS = dims


def sanitize_and_normalize(vec: list[float]) -> list[float]:
    """Sanitize NaN/Inf values and L2-normalize."""
    sanitized = [v if math.isfinite(v) else 0.0 for v in vec]
    magnitude = math.sqrt(sum(v * v for v in sanitized))
    if magnitude < 1e-10:
        return sanitized
    return [v / magnitude for v in sanitized]


def coerce_pgvector_dims(
    embedding: list[float], expected_dims: int | None = None
) -> list[float]:
    """Pad or truncate embedding to match pgvector column dimensions."""
    target = expected_dims if expected_dims is not None else PGVECTOR_DIMS
    if len(embedding) == target:
        return embedding
    if len(embedding) > target:
        # Truncate and re-normalize (safe for MRL-capable models)
        truncated = embedding[:target]
        return sanitize_and_normalize(truncated)
    return embedding + ([0.0] * (target - len(embedding)))


def embedding_to_pg_literal(embedding: list[float]) -> str:
    """Convert embedding list to PostgreSQL vector literal string."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class OpenAIEmbeddingProvider:
    id = "openai"

    def __init__(
        self, model: str, api_key: str, base_url: str | None, headers: dict[str, str]
    ):
        import openai

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if headers:
            client_kwargs["default_headers"] = headers
        self._client = openai.OpenAI(**client_kwargs)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self.model)
        return sanitize_and_normalize(resp.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self.model)
        return [sanitize_and_normalize(d.embedding) for d in resp.data]


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


class GeminiEmbeddingProvider:
    id = "gemini"

    def __init__(self, model: str, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        result = self._genai.embed_content(model=self.model, content=text)
        return sanitize_and_normalize(result["embedding"])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


# ---------------------------------------------------------------------------
# Voyage adapter
# ---------------------------------------------------------------------------


class VoyageEmbeddingProvider:
    id = "voyage"

    def __init__(self, model: str, api_key: str):
        import voyageai

        self._client = voyageai.Client(api_key=api_key)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self.model)
        return sanitize_and_normalize(result.embeddings[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self.model)
        return [sanitize_and_normalize(v) for v in result.embeddings]


# ---------------------------------------------------------------------------
# Local (llama-cpp-python) adapter
# ---------------------------------------------------------------------------


class LocalEmbeddingProvider:
    id = "local"

    def __init__(self, model: str, model_path: str, model_cache_dir: str | None = None):
        try:
            from llama_cpp import Llama  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require llama-cpp-python: "
                "pip install 'openclaw-memory[local]'"
            ) from exc
        from llama_cpp import Llama

        self._llama = Llama(model_path=model_path, embedding=True, verbose=False)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        vec = self._llama.embed(text)
        return sanitize_and_normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
_DEFAULT_VOYAGE_MODEL = "voyage-4-large"
_REMOTE_PROVIDER_IDS = ("openai", "gemini", "voyage")


def _resolve_key(provider_id: str) -> str | None:
    """Resolve API key from environment for a given provider."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "voyage": "VOYAGE_API_KEY",
    }
    env_var = env_map.get(provider_id)
    return os.environ.get(env_var) if env_var else None


def _create_for_id(provider_id: str, model: str) -> EmbeddingProvider:
    """Create embedding provider by ID. Raises RuntimeError if key missing."""
    if provider_id == "openai":
        key = _resolve_key("openai")
        if not key:
            raise RuntimeError("No API key found for provider openai (OPENAI_API_KEY)")
        return OpenAIEmbeddingProvider(model=model, api_key=key, base_url=None, headers={})

    if provider_id == "gemini":
        key = _resolve_key("gemini")
        if not key:
            raise RuntimeError("No API key found for provider gemini (GEMINI_API_KEY)")
        return GeminiEmbeddingProvider(model=model, api_key=key)

    if provider_id == "voyage":
        key = _resolve_key("voyage")
        if not key:
            raise RuntimeError("No API key found for provider voyage (VOYAGE_API_KEY)")
        return VoyageEmbeddingProvider(model=model, api_key=key)

    raise ValueError(f"Unknown provider id: {provider_id!r}")


def create_embedding_provider(
    provider: str = "auto",
    model: str | None = None,
) -> EmbeddingProvider:
    """
    Create an embedding provider with auto-detection support.

    Args:
        provider: Provider ID ("openai", "gemini", "voyage", "auto").
        model:    Model name override. Defaults based on provider.

    Raises:
        RuntimeError: If no provider can be created (missing API keys).
    """
    model_map = {
        "openai": _DEFAULT_OPENAI_MODEL,
        "gemini": _DEFAULT_GEMINI_MODEL,
        "voyage": _DEFAULT_VOYAGE_MODEL,
    }

    if provider != "auto":
        resolved_model = model or model_map.get(provider, _DEFAULT_OPENAI_MODEL)
        return _create_for_id(provider, resolved_model)

    # Auto-detect: try providers in order
    errors: list[str] = []
    for pid in _REMOTE_PROVIDER_IDS:
        resolved_model = model or model_map.get(pid, _DEFAULT_OPENAI_MODEL)
        try:
            return _create_for_id(pid, resolved_model)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue

    raise RuntimeError(
        "No embedding provider available. Set one of: "
        "OPENAI_API_KEY, GEMINI_API_KEY, or VOYAGE_API_KEY.\n"
        + "\n".join(errors)
    )
