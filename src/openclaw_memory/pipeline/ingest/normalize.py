"""Normalization helpers for structured memory fields."""

from __future__ import annotations

import re

VALID_MEMORY_TYPES = frozenset({"preference", "fact", "decision", "event"})

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def normalize_memory_key(raw: str) -> str:
    """Normalize a raw memory key to lowercase dotted-path."""
    if not raw:
        return ""
    key = raw.strip().lower()
    key = re.sub(r"[ \t-]+", "_", key)
    key = re.sub(r"\.{2,}", ".", key)
    key = re.sub(r"[^a-z0-9_.À-ÿ]", "", key)
    return key


def normalize_value(raw: str) -> str:
    """Trim whitespace from a value string."""
    return raw.strip()


def parse_event_time(raw: str | None) -> str | None:
    """Return stripped ISO 8601 string if valid, else None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text and _ISO_RE.match(text):
        return text
    return None


def apply_type_fallback(memory_type: str) -> str:
    """Validate memory_type; default to 'fact'."""
    return memory_type if memory_type in VALID_MEMORY_TYPES else "fact"
