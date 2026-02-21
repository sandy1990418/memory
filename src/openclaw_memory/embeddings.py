"""
Embedding provider abstraction and adapters.
Mirrors: src/memory/embeddings.ts, embeddings-openai.ts, embeddings-gemini.ts, embeddings-voyage.ts
"""
from __future__ import annotations

import math
import os
from typing import Protocol, runtime_checkable

from .config import LocalConfig, RemoteConfig


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


def _sanitize_and_normalize(vec: list[float]) -> list[float]:
    sanitized = [v if math.isfinite(v) else 0.0 for v in vec]
    magnitude = math.sqrt(sum(v * v for v in sanitized))
    if magnitude < 1e-10:
        return sanitized
    return [v / magnitude for v in sanitized]


def _missing_key_error(provider_id: str) -> str:
    return f"No API key found for provider {provider_id}"


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class OpenAIEmbeddingProvider:
    id = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None, headers: dict[str, str]):
        import openai  # lazy import

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if headers:
            # openai SDK uses httpx, pass extra headers via default_headers
            client_kwargs["default_headers"] = headers
        self._client = openai.OpenAI(**client_kwargs)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self.model)
        return _sanitize_and_normalize(resp.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self.model)
        return [_sanitize_and_normalize(d.embedding) for d in resp.data]


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


class GeminiEmbeddingProvider:
    id = "gemini"

    def __init__(self, model: str, api_key: str):
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model

    def _embed_one(self, text: str) -> list[float]:
        result = self._genai.embed_content(model=self.model, content=text)
        vec = result["embedding"]
        return _sanitize_and_normalize(vec)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Voyage adapter
# ---------------------------------------------------------------------------


class VoyageEmbeddingProvider:
    id = "voyage"

    def __init__(self, model: str, api_key: str):
        import voyageai  # lazy import

        self._client = voyageai.Client(api_key=api_key)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self.model)
        vec = result.embeddings[0]
        return _sanitize_and_normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self.model)
        return [_sanitize_and_normalize(v) for v in result.embeddings]


# ---------------------------------------------------------------------------
# Local (llama-cpp-python) adapter — deferred, stub only
# ---------------------------------------------------------------------------


class LocalEmbeddingProvider:
    id = "local"

    def __init__(self, model: str, model_path: str, model_cache_dir: str | None = None):
        try:
            from llama_cpp import Llama  # lazy import  # noqa: F401
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
        return _sanitize_and_normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_REMOTE_PROVIDER_IDS = ("openai", "gemini", "voyage")
DEFAULT_LOCAL_MODEL = "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf"


def _resolve_openai_key(remote: RemoteConfig | None) -> str | None:
    if remote and remote.api_key:
        return remote.api_key
    return os.environ.get("OPENAI_API_KEY")


def _resolve_gemini_key(remote: RemoteConfig | None) -> str | None:
    if remote and remote.api_key:
        return remote.api_key
    return os.environ.get("GEMINI_API_KEY")


def _resolve_voyage_key(remote: RemoteConfig | None) -> str | None:
    if remote and remote.api_key:
        return remote.api_key
    return os.environ.get("VOYAGE_API_KEY")


def _create_for_id(
    provider_id: str,
    model: str,
    remote: RemoteConfig | None,
    local: LocalConfig | None,
) -> EmbeddingProvider:
    if provider_id == "openai":
        key = _resolve_openai_key(remote)
        if not key:
            raise RuntimeError(_missing_key_error("openai"))
        return OpenAIEmbeddingProvider(
            model=model,
            api_key=key,
            base_url=remote.base_url if remote else None,
            headers=remote.headers if remote else {},
        )
    if provider_id == "gemini":
        key = _resolve_gemini_key(remote)
        if not key:
            raise RuntimeError(_missing_key_error("gemini"))
        return GeminiEmbeddingProvider(model=model, api_key=key)
    if provider_id == "voyage":
        key = _resolve_voyage_key(remote)
        if not key:
            raise RuntimeError(_missing_key_error("voyage"))
        return VoyageEmbeddingProvider(model=model, api_key=key)
    if provider_id == "local":
        model_path = (local and local.model_path) or DEFAULT_LOCAL_MODEL
        cache_dir = local.model_cache_dir if local else None
        return LocalEmbeddingProvider(model=model, model_path=model_path, model_cache_dir=cache_dir)
    raise ValueError(f"Unknown provider id: {provider_id!r}")


def _is_missing_key_error(exc: Exception) -> bool:
    return "No API key found for provider" in str(exc)


def create_embedding_provider(
    *,
    provider: str,
    model: str,
    fallback: str = "none",
    remote: RemoteConfig | None = None,
    local: LocalConfig | None = None,
) -> tuple[EmbeddingProvider | None, dict]:
    """
    Create an embedding provider following the same auto-selection logic as TS.
    Returns (provider_or_None, meta_dict).
    meta_dict keys: requested_provider, fallback_from, fallback_reason, unavailable_reason.
    """
    meta: dict = {"requested_provider": provider}

    if provider == "auto":
        # 1. local if model path configured and file exists
        if local and local.model_path:
            path = local.model_path
            if not path.startswith("hf:") and not path.startswith("http"):
                import os as _os
                if _os.path.isfile(_os.path.expanduser(path)):
                    try:
                        p = _create_for_id("local", model, remote, local)
                        return p, meta
                    except Exception:
                        pass

        # 2. Try remote providers in order
        missing_key_errors: list[str] = []
        for pid in _REMOTE_PROVIDER_IDS:
            try:
                p = _create_for_id(pid, model, remote, local)
                return p, meta
            except RuntimeError as exc:
                if _is_missing_key_error(exc):
                    missing_key_errors.append(str(exc))
                    continue
                raise

        # All failed due to missing keys → FTS-only
        reason = "\n\n".join(missing_key_errors) if missing_key_errors else "No embeddings provider available."
        meta["unavailable_reason"] = reason
        return None, meta

    # Specific provider requested
    try:
        p = _create_for_id(provider, model, remote, local)
        return p, meta
    except Exception as primary_err:
        reason = str(primary_err)
        if fallback and fallback != "none" and fallback != provider:
            try:
                p = _create_for_id(fallback, model, remote, local)
                meta["fallback_from"] = provider
                meta["fallback_reason"] = reason
                return p, meta
            except Exception as fallback_err:
                combined = f"{reason}\n\nFallback to {fallback} failed: {fallback_err}"
                if _is_missing_key_error(primary_err) and _is_missing_key_error(fallback_err):
                    meta["unavailable_reason"] = combined
                    meta["fallback_from"] = provider
                    meta["fallback_reason"] = reason
                    return None, meta
                raise RuntimeError(combined) from fallback_err

        if _is_missing_key_error(primary_err):
            meta["unavailable_reason"] = reason
            return None, meta
        raise
