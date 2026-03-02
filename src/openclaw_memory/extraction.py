"""
LLM-based memory extraction and classification from conversations.
Extends memory_flush.py with structured JSON output, classification, and confidence scoring.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .memory_normalize import (
    apply_type_fallback,
    normalize_memory_key,
    normalize_value,
    parse_event_time,
)

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


@dataclass
class ExtractedMemory:
    content: str
    memory_type: str  # "preference" | "fact" | "decision" | "event"
    confidence: float  # 0.0–1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_key: str = ""
    value: str = ""
    event_time: str | None = None
    source_refs: list[str] = field(default_factory=list)


def _coerce_text(value: Any) -> str:
    """
    Best-effort conversion of non-string JSON fields into text.

    LLM structured outputs occasionally emit arrays/objects for fields that are
    expected to be strings. We normalize those to avoid parser crashes.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_coerce_text(v) for v in value]
        joined = " ".join(p for p in parts if p)
        if joined:
            return joined.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value).strip()
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value).strip()
    return str(value).strip()


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _parse_llm_response(response: str) -> list[ExtractedMemory]:
    """Parse the LLM JSON response into ExtractedMemory objects."""
    text = _coerce_text(response)

    # Strip markdown code fences if present
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
        content = _coerce_text(item.get("content", ""))
        if not content:
            continue
        memory_type_raw = _coerce_text(item.get("memory_type", "fact")).lower()
        memory_type = apply_type_fallback(memory_type_raw or "fact")
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        memory_key_raw = item.get("memory_key", "")
        memory_key = normalize_memory_key(memory_key_raw if isinstance(memory_key_raw, str) else "")
        raw_value = item.get("value", "")
        value_text = _coerce_text(raw_value)
        value = normalize_value(value_text) if value_text else content
        event_time = parse_event_time(item.get("event_time"))
        raw_refs = item.get("source_refs", [])
        source_refs = [str(r) for r in raw_refs] if isinstance(raw_refs, list) else []
        memories.append(
            ExtractedMemory(
                content=content,
                memory_type=memory_type,
                confidence=confidence,
                metadata=metadata,
                memory_key=memory_key,
                value=value,
                event_time=event_time,
                source_refs=source_refs,
            )
        )
    return memories


def extract_memories(
    conversation: list[dict[str, str]],
    llm_fn: Callable[[str], str],
    *,
    prompt_template: str = EXTRACTION_PROMPT,
) -> list[ExtractedMemory]:
    """
    Extract and classify memories from *conversation* via an LLM call.

    Returns a list of ExtractedMemory with type classification and confidence scores.
    Supports both Chinese and English conversations (handled by the LLM).
    """
    if not conversation:
        return []

    formatted = _format_conversation(conversation)
    prompt = prompt_template.replace("{conversation}", formatted)
    response = llm_fn(prompt)
    return _parse_llm_response(response)
