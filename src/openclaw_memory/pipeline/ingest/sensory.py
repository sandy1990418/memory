"""
LightMem Stage 1: Sensory filtering and compression.

Always active in the pipeline (not a feature flag). Reduces input
size before LLM extraction by:
  1. Selecting relevant messages (filtering low-info acknowledgments)
  2. Pre-compressing verbose turns via TF-IDF sentence scoring
  3. Segmenting by topic drift
  4. Trimming to token budget
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ...utils.tokens import estimate_tokens

_TOKEN_RE = re.compile(r"[a-z0-9_]+", flags=re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WS_RE = re.compile(r"\s+")
_KEYWORD_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\"[^\"]+\"|'[^']+'")
_ACK_PATTERNS = frozenset({
    "ok", "okay", "sure", "thanks", "thank you", "yes", "no",
    "yep", "nope", "got it", "k", "yea", "yeah", "alright",
})


@dataclass(frozen=True)
class SensoryConfig:
    """Configuration for sensory-stage preprocessing."""

    pre_compress: bool = True
    messages_use: str = "all"  # "all" | "user_only"
    topic_segment: bool = True
    max_input_tokens: int = 4096
    topic_token_threshold: int = 800
    topic_similarity_threshold: float = 0.18
    per_message_char_limit: int = 320
    segment_char_limit: int = 900


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _signature(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_keywords(text: str) -> frozenset[str]:
    return frozenset(
        m.strip("\"'").lower()
        for m in _KEYWORD_RE.findall(text)
        if len(m) > 1
    )


def _is_low_info(content: str) -> bool:
    """Check if a message is a short acknowledgment."""
    stripped = content.strip().rstrip(".!,").lower()
    if len(content) < 20 and "?" not in content:
        if stripped in _ACK_PATTERNS:
            return True
    return False


def _compute_sentence_scores(sentences: list[str]) -> list[float]:
    """TF-IDF importance scoring for sentence selection."""
    if not sentences:
        return []

    doc_freq: dict[str, int] = {}
    sentence_tokens = []
    for sent in sentences:
        tokens = [t.lower() for t in _TOKEN_RE.findall(sent)]
        sentence_tokens.append(tokens)
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    n_docs = len(sentences)
    scores: list[float] = []
    for tokens in sentence_tokens:
        if not tokens:
            scores.append(0.0)
            continue
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            tf[t] /= len(tokens)
        score = sum(
            freq * (math.log((n_docs + 1) / (doc_freq.get(t, 0) + 1)) + 1)
            for t, freq in tf.items()
        )
        scores.append(score)
    return scores


def compress_text(text: str, limit: int) -> str:
    """Compress text using TF-IDF sentence scoring to preserve key content."""
    if limit <= 0:
        return ""
    txt = _normalize_text(text)
    if len(txt) <= limit:
        return txt

    sentences = [p.strip() for p in _SENTENCE_SPLIT_RE.split(txt) if p.strip()]

    if len(sentences) <= 2:
        head = max(1, int(limit * 0.65))
        tail = max(1, limit - head - 5)
        return f"{txt[:head].rstrip()} ... {txt[-tail:].lstrip()}"

    scores = _compute_sentence_scores(sentences)
    first, last = sentences[0], sentences[-1]
    base_len = len(first) + len(last) + 1

    if base_len > limit:
        head = max(1, int(limit * 0.65))
        tail = max(1, limit - head - 5)
        return f"{txt[:head].rstrip()} ... {txt[-tail:].lstrip()}"

    remaining = limit - base_len
    middle = [(i, sentences[i], scores[i]) for i in range(1, len(sentences) - 1)]
    middle.sort(key=lambda x: x[2], reverse=True)

    selected = {0, len(sentences) - 1}
    for idx, sent, _score in middle:
        cost = len(sent) + 1
        if cost <= remaining:
            selected.add(idx)
            remaining -= cost

    return " ".join(sentences[i] for i in sorted(selected))


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _select_messages(
    conversation: list[dict[str, Any]],
    messages_use: str,
) -> list[dict[str, str]]:
    """Filter messages by role and information density."""
    selected: list[dict[str, str]] = []
    for raw in conversation:
        role = str(raw.get("role", "unknown")).strip().lower() or "unknown"
        content = _normalize_text(str(raw.get("content", "")))
        if not content:
            continue
        if messages_use == "user_only" and role not in {"user", "system"}:
            continue
        if _is_low_info(content):
            continue
        selected.append({"role": role, "content": content})
    return selected


def _segment_by_topic(
    messages: list[dict[str, str]],
    cfg: SensoryConfig,
) -> list[dict[str, str]]:
    """Segment long transcripts by topical drift."""
    if not cfg.topic_segment or len(messages) <= 1:
        return messages

    segments: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_tokens = 0
    current_sig: frozenset[str] = frozenset()
    current_kw: frozenset[str] = frozenset()
    threshold = max(64, cfg.topic_token_threshold)

    for msg in messages:
        msg_sig = _signature(msg["content"])
        msg_tokens = estimate_tokens(msg["content"])
        msg_kw = _extract_keywords(msg["content"])

        if current:
            sim = 0.6 * _jaccard(current_sig, msg_sig) + 0.4 * _jaccard(current_kw, msg_kw)
        else:
            sim = 1.0

        drift = bool(current) and current_tokens >= threshold and sim < cfg.topic_similarity_threshold
        hard = bool(current) and current_tokens + msg_tokens > threshold * 2
        if drift or hard:
            segments.append(current)
            current, current_tokens, current_sig, current_kw = [], 0, frozenset(), frozenset()

        current.append(msg)
        current_tokens += msg_tokens
        current_sig |= msg_sig
        current_kw |= msg_kw

    if current:
        segments.append(current)
    if len(segments) <= 1:
        return messages

    out: list[dict[str, str]] = []
    for idx, seg in enumerate(segments, 1):
        block = " ".join(f"{m['role']}: {m['content']}" for m in seg)
        compact = compress_text(block, max(80, cfg.segment_char_limit))
        out.append({"role": "system", "content": f"[topic_{idx}] {compact}"})
    return out


def _trim_to_budget(
    messages: list[dict[str, str]],
    max_tokens: int,
) -> list[dict[str, str]]:
    """Trim message list to fit within token budget."""
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
            "content": compress_text(out[-1]["content"], max_chars),
        }
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_for_extraction(
    conversation: list[dict[str, Any]],
    config: SensoryConfig | None = None,
) -> list[dict[str, str]]:
    """
    Full sensory pipeline: select -> compress -> segment -> trim.

    Always active — this IS the first stage of the ingest pipeline.
    """
    cfg = config or SensoryConfig()
    selected = _select_messages(conversation, cfg.messages_use)
    if not selected:
        return []

    if cfg.pre_compress:
        selected = [
            {
                "role": m["role"],
                "content": compress_text(m["content"], max(40, cfg.per_message_char_limit)),
            }
            for m in selected
        ]

    segmented = _segment_by_topic(selected, cfg)
    return _trim_to_budget(segmented, max(64, cfg.max_input_tokens))


def build_session_summary(
    conversation: list[dict[str, Any]],
    config: SensoryConfig | None = None,
    max_chars: int = 1800,
) -> str:
    """Build a compact archival session summary without LLM calls."""
    prepared = prepare_for_extraction(conversation, config)
    if not prepared:
        return ""
    lines = [f"{m['role']}: {m['content']}" for m in prepared]
    return compress_text("\n".join(lines), max(80, max_chars))
