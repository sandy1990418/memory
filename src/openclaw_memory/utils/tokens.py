"""Token estimation utilities."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4). Used across the memory subsystem."""
    if not text:
        return 0
    return max(1, len(text) // 4)
