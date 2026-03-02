"""
LightMem-inspired preprocessing helpers.

These helpers reduce distillation prompt size by:
  1) selecting only useful roles (for example user messages),
  2) pre-compressing verbose turns,
  3) segmenting long conversations into topic blocks,
  4) enforcing a hard token budget before sending to the extractor LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_]+", flags=re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WS_RE = re.compile(r"\s+")

_VALID_MESSAGES_USE = frozenset({"all", "user_only"})


@dataclass(frozen=True)
class DistillPrepConfig:
    """
    Configuration for distillation-time prompt shrinking.
    """

    pre_compress: bool = False
    messages_use: str = "all"
    topic_segment: bool = False
    max_input_tokens: int = 4096
    topic_token_threshold: int = 800
    topic_similarity_threshold: float = 0.18
    per_message_char_limit: int = 320
    segment_char_limit: int = 900


def estimate_tokens(text: str) -> int:
    """Rough token estimate used across the memory subsystem (chars / 4)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def normalize_messages_use(value: str) -> str:
    """Return a safe role-selection mode."""
    v = (value or "all").strip().lower()
    if v in _VALID_MESSAGES_USE:
        return v
    return "all"


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _signature(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _compress_text(text: str, limit: int) -> str:
    """Compress long text while keeping head/tail cues."""
    if limit <= 0:
        return ""
    txt = _normalize_text(text)
    if len(txt) <= limit:
        return txt

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(txt) if p.strip()]
    if len(parts) >= 3:
        middle = parts[len(parts) // 2]
        candidate = " ".join([parts[0], middle, parts[-1]])
        if len(candidate) <= limit:
            return candidate

    head = max(1, int(limit * 0.65))
    tail = max(1, limit - head - 5)
    return f"{txt[:head].rstrip()} ... {txt[-tail:].lstrip()}"


def _select_messages(
    conversation: list[dict[str, Any]],
    *,
    messages_use: str,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    mode = normalize_messages_use(messages_use)
    for raw in conversation:
        role = str(raw.get("role", "unknown")).strip().lower() or "unknown"
        content = _normalize_text(str(raw.get("content", "")))
        if not content:
            continue
        if mode == "user_only" and role not in {"user", "system"}:
            continue
        selected.append({"role": role, "content": content})
    return selected


def _segment_messages(
    messages: list[dict[str, str]],
    *,
    cfg: DistillPrepConfig,
) -> list[dict[str, str]]:
    """
    Segment long transcripts by topical drift.

    When topic segmentation is enabled, each segment is collapsed into a
    synthetic "system" message to reduce extractor input size.
    """
    if not cfg.topic_segment or len(messages) <= 1:
        return messages

    segments: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_tokens = 0
    current_sig: frozenset[str] = frozenset()
    split_threshold = max(64, cfg.topic_token_threshold)

    for msg in messages:
        msg_sig = _signature(msg["content"])
        msg_tokens = estimate_tokens(msg["content"])
        sim = _jaccard(current_sig, msg_sig) if current else 1.0

        drift_split = (
            bool(current)
            and current_tokens >= split_threshold
            and sim < cfg.topic_similarity_threshold
        )
        hard_split = bool(current) and current_tokens + msg_tokens > split_threshold * 2
        if drift_split or hard_split:
            segments.append(current)
            current = []
            current_tokens = 0
            current_sig = frozenset()

        current.append(msg)
        current_tokens += msg_tokens
        current_sig = current_sig | msg_sig

    if current:
        segments.append(current)

    # No real split happened.
    if len(segments) <= 1:
        return messages

    out: list[dict[str, str]] = []
    for idx, seg in enumerate(segments, start=1):
        block = " ".join(f"{m['role']}: {m['content']}" for m in seg)
        compact = _compress_text(block, max(80, cfg.segment_char_limit))
        out.append({"role": "system", "content": f"[topic_{idx}] {compact}"})
    return out


def _trim_to_budget(messages: list[dict[str, str]], *, max_tokens: int) -> list[dict[str, str]]:
    if max_tokens <= 0:
        return []
    out = list(messages)
    total = sum(estimate_tokens(m["content"]) for m in out)
    while total > max_tokens and len(out) > 1:
        pop_index = 1 if out[0]["role"] == "system" else 0
        removed = out.pop(pop_index)
        total -= estimate_tokens(removed["content"])

    if total > max_tokens and out:
        max_chars = max(32, max_tokens * 4)
        out[-1] = {
            "role": out[-1]["role"],
            "content": _compress_text(out[-1]["content"], max_chars),
        }
    return out


def prepare_messages_for_distill(
    conversation: list[dict[str, Any]],
    *,
    config: DistillPrepConfig,
) -> list[dict[str, str]]:
    """
    Build an extractor-ready conversation under a token budget.
    """
    selected = _select_messages(conversation, messages_use=config.messages_use)
    if not selected:
        return []

    if config.pre_compress:
        selected = [
            {
                "role": m["role"],
                "content": _compress_text(m["content"], max(40, config.per_message_char_limit)),
            }
            for m in selected
        ]

    segmented = _segment_messages(selected, cfg=config)
    return _trim_to_budget(segmented, max_tokens=max(64, config.max_input_tokens))


def build_compact_session_text(
    conversation: list[dict[str, Any]],
    *,
    config: DistillPrepConfig,
    max_chars: int = 1800,
) -> str:
    """
    Build a compact archival session string without another LLM call.
    """
    prepared = prepare_messages_for_distill(conversation, config=config)
    if not prepared:
        return ""
    lines = [f"{m['role']}: {m['content']}" for m in prepared]
    joined = "\n".join(lines)
    return _compress_text(joined, max(80, max_chars))

