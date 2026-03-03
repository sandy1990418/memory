"""
LLM-based memory extraction and classification (mem0-style).

Extracts structured memories from conversations:
  {content, memory_type, confidence, memory_key, value, event_time}
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from ...core.types import ExtractedMemory
from ...utils.text import coerce_text

EXTRACTION_PROMPT = (
    "You are a memory extraction assistant. "
    "Analyze the following conversation and extract facts worth remembering long-term.\n\n"
    "For each memory, classify it as one of:\n"
    '  - "preference": user preferences, likes/dislikes, settings\n'
    '  - "fact": factual information about the user or their context\n'
    '  - "decision": decisions made by the user\n'
    '  - "event": notable events or actions that occurred\n\n'
    "Return a JSON array. Each item must have:\n"
    '  "content": string — the memory text\n'
    '  "memory_type": one of preference/fact/decision/event\n'
    '  "confidence": float 0.0–1.0 — how confident you are this is worth storing\n'
    '  "metadata": object — optional extra context (e.g. {"language": "zh"})\n'
    '  "memory_key": string — canonical dotted-path key (e.g. "profile.favorite_cuisine")\n'
    '  "value": string — normalized value or compact JSON string\n'
    '  "event_time": string | null — ISO 8601 timestamp if applicable, else null\n'
    '  "source_refs": array of strings — message IDs or timestamps this memory came from\n\n'
    "If nothing is worth storing, return an empty array [].\n"
    "Respond ONLY with valid JSON, no markdown fences.\n\n"
    "Conversation:\n{conversation}"
)

VALID_MEMORY_TYPES = frozenset({"preference", "fact", "decision", "event"})

# ISO 8601 pattern
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def _normalize_key(raw: str) -> str:
    """Normalize memory key to lowercase dotted-path."""
    if not raw:
        return ""
    key = raw.strip().lower()
    key = re.sub(r"[ \t-]+", "_", key)
    key = re.sub(r"\.{2,}", ".", key)
    key = re.sub(r"[^a-z0-9_.À-ÿ]", "", key)
    return key


def _parse_event_time(raw: Any) -> str | None:
    """Validate and return ISO 8601 timestamp or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text and _ISO_RE.match(text):
        return text
    return None


def _apply_type_fallback(memory_type: str) -> str:
    return memory_type if memory_type in VALID_MEMORY_TYPES else "fact"


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in conversation
    )


def _parse_response(response: str) -> list[ExtractedMemory]:
    """Parse LLM JSON response into ExtractedMemory objects."""
    text = coerce_text(response)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if not text or text in ("[]", "null"):
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    memories: list[ExtractedMemory] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = coerce_text(item.get("content", ""))
        if not content:
            continue

        memory_type = _apply_type_fallback(
            coerce_text(item.get("memory_type", "fact")).lower()
        )
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.8))))
        except (TypeError, ValueError):
            confidence = 0.8

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        memory_key = _normalize_key(
            item.get("memory_key", "") if isinstance(item.get("memory_key"), str) else ""
        )
        value_text = coerce_text(item.get("value", ""))
        value = value_text.strip() if value_text else content

        raw_refs = item.get("source_refs", [])
        source_refs = [str(r) for r in raw_refs] if isinstance(raw_refs, list) else []

        memories.append(ExtractedMemory(
            content=content,
            memory_type=memory_type,
            confidence=confidence,
            metadata=metadata,
            memory_key=memory_key,
            value=value,
            event_time=_parse_event_time(item.get("event_time")),
            source_refs=source_refs,
        ))
    return memories


def extract_memories(
    conversation: list[dict[str, str]],
    llm_fn: Callable[[str], str],
) -> list[ExtractedMemory]:
    """
    Extract and classify memories from a conversation via LLM.

    Returns a list of ExtractedMemory with type classification and
    confidence scores. Supports Chinese and English conversations.
    """
    if not conversation:
        return []

    formatted = _format_conversation(conversation)
    prompt = EXTRACTION_PROMPT.replace("{conversation}", formatted)
    response = llm_fn(prompt)
    return _parse_response(response)
