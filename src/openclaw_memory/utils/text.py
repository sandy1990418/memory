"""Shared text processing helpers."""

from __future__ import annotations

import json
import re
from typing import Any

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return _WS_RE.sub(" ", text).strip()


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of LLM JSON response, stripping markdown fences."""
    text = strip_markdown_fences(raw)
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def coerce_text(value: Any) -> str:
    """Best-effort conversion of non-string JSON fields into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [coerce_text(v) for v in value]
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


def truncate_utf16_safe(text: str, max_chars: int) -> str:
    """Truncate string safely, respecting UTF-16 surrogate pairs."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if truncated and ord(truncated[-1]) >= 0xD800 and ord(truncated[-1]) <= 0xDBFF:
        truncated = truncated[:-1]
    return truncated
