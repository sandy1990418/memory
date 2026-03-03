"""
LLM client abstraction and adapters.

Supports OpenAI, Anthropic, and Google Gemini.
Follows the same factory pattern as core/embeddings.py.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

_PROVIDER_IDS = ("openai", "anthropic", "gemini")


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class OpenAILLMClient:
    """LLM client using the OpenAI Chat Completions API."""

    def __init__(self, model: str, api_key: str) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


class AnthropicLLMClient:
    """LLM client using the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


class GeminiLLMClient:
    """LLM client using the Google Generative AI API."""

    def __init__(self, model: str, api_key: str) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def _resolve_key(provider_id: str) -> str | None:
    """Resolve API key from environment for a given provider."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_map.get(provider_id)
    return os.environ.get(env_var) if env_var else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _create_for_id(provider_id: str, model: str) -> Callable[[str], str]:
    """Create LLM client by provider ID. Raises RuntimeError if key missing."""
    if provider_id == "openai":
        key = _resolve_key("openai")
        if not key:
            raise RuntimeError("No API key found for provider openai (OPENAI_API_KEY)")
        return OpenAILLMClient(model=model, api_key=key)

    if provider_id == "anthropic":
        key = _resolve_key("anthropic")
        if not key:
            raise RuntimeError("No API key found for provider anthropic (ANTHROPIC_API_KEY)")
        return AnthropicLLMClient(model=model, api_key=key)

    if provider_id == "gemini":
        key = _resolve_key("gemini")
        if not key:
            raise RuntimeError("No API key found for provider gemini (GEMINI_API_KEY)")
        return GeminiLLMClient(model=model, api_key=key)

    raise ValueError(f"Unknown LLM provider: {provider_id!r}")


def create_llm_client(
    provider: str = "auto",
    model: str | None = None,
) -> Callable[[str], str] | None:
    """
    Create an LLM client with auto-detection support.

    Args:
        provider: Provider ID ("openai", "anthropic", "gemini", "auto").
        model:    Model name override. Defaults based on provider.

    Returns:
        A callable ``(prompt: str) -> str``, or ``None`` if no provider
        could be created (auto mode with no API keys).

    Raises:
        RuntimeError: If an explicit provider is specified but its API key
            is missing.
    """
    model_map = {
        "openai": _DEFAULT_OPENAI_MODEL,
        "anthropic": _DEFAULT_ANTHROPIC_MODEL,
        "gemini": _DEFAULT_GEMINI_MODEL,
    }

    if provider != "auto":
        resolved_model = model or model_map.get(provider, _DEFAULT_OPENAI_MODEL)
        return _create_for_id(provider, resolved_model)

    # Auto-detect: try providers in order, return first with a valid key
    for pid in _PROVIDER_IDS:
        resolved_model = model or model_map.get(pid, _DEFAULT_OPENAI_MODEL)
        try:
            client = _create_for_id(pid, resolved_model)
            logger.info("Auto-detected LLM provider: %s (model=%s)", pid, resolved_model)
            return client
        except RuntimeError:
            continue

    logger.warning(
        "No LLM provider available. Set one of: "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY. "
        "LLM-dependent features (extraction, answer generation) will be disabled."
    )
    return None
