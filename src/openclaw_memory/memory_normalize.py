"""
Normalization helpers for structured memory fields.
"""

from __future__ import annotations

import re

VALID_MEMORY_TYPES = frozenset({"preference", "fact", "decision", "event"})

# ISO 8601 basic pattern: matches YYYY-MM-DD with optional time portion
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"          # date
    r"(?:[ T]\d{2}:\d{2}"          # optional time HH:MM
    r"(?::\d{2}(?:\.\d+)?)?"       # optional :SS.fff
    r"(?:Z|[+-]\d{2}:?\d{2})?"     # optional timezone
    r")?$"
)


def normalize_memory_key(raw: str) -> str:
    """
    Normalize a raw memory key to a canonical lowercase dotted-path.

    Examples:
        "Profile.Favorite Cuisine" -> "profile.favorite_cuisine"
        "  user.city  "            -> "user.city"
    """
    if not raw:
        return ""
    # Strip surrounding whitespace, lowercase
    key = raw.strip().lower()
    # Replace spaces/hyphens within segments with underscores
    key = re.sub(r"[ \t-]+", "_", key)
    # Collapse multiple dots
    key = re.sub(r"\.{2,}", ".", key)
    # Remove characters that are not alphanumeric, underscore, or dot
    key = re.sub(r"[^a-z0-9_.À-ÿ]", "", key)
    return key


def normalize_value(raw: str) -> str:
    """Trim whitespace and perform basic cleanup on a value string."""
    return raw.strip()


def parse_event_time(raw: str | None) -> str | None:
    """
    Parse an ISO 8601 timestamp string.

    Returns the stripped string if it looks like a valid ISO timestamp,
    otherwise returns None.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if _ISO_RE.match(text):
        return text
    return None


def apply_type_fallback(memory_type: str) -> str:
    """
    Validate *memory_type* against VALID_MEMORY_TYPES.

    Returns the type unchanged if valid, otherwise returns "fact".
    """
    if memory_type in VALID_MEMORY_TYPES:
        return memory_type
    return "fact"
