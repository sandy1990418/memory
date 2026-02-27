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
    '  "metadata": object — optional extra context (e.g. {"language": "zh"})\n\n'
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


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _parse_llm_response(response: str) -> list[ExtractedMemory]:
    """Parse the LLM JSON response into ExtractedMemory objects."""
    text = response.strip()

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
        content = item.get("content", "").strip()
        if not content:
            continue
        memory_type = item.get("memory_type", "fact")
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = "fact"
        confidence = float(item.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        memories.append(
            ExtractedMemory(
                content=content,
                memory_type=memory_type,
                confidence=confidence,
                metadata=metadata,
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
