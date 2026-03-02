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

import math

_TOKEN_RE = re.compile(r"[a-z0-9_]+", flags=re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WS_RE = re.compile(r"\s+")
_KEYWORD_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\"[^\"]+\"|'[^']+'")
_ACK_PATTERNS = frozenset({"ok", "okay", "sure", "thanks", "thank you", "yes", "no", "yep", "nope", "got it", "k", "yea", "yeah", "alright"})

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


def _extract_keywords(text: str) -> frozenset[str]:
    """Extract capitalized phrases and quoted strings as keywords."""
    return frozenset(m.strip("\"'").lower() for m in _KEYWORD_RE.findall(text) if len(m) > 1)


def _keyword_overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute overlap ratio between two keyword sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _is_low_info_message(content: str) -> bool:
    """Check if a message is a short acknowledgment with low information density."""
    stripped = content.strip().rstrip(".!,").lower()
    if len(content) < 20 and "?" not in content:
        if stripped in _ACK_PATTERNS:
            return True
        # Also catch very short messages with no real content
        words = _TOKEN_RE.findall(stripped)
        if len(words) <= 2 and stripped in _ACK_PATTERNS:
            return True
    return False


def _compute_sentence_scores(sentences: list[str]) -> list[float]:
    """Score sentences by TF-IDF importance relative to the full text."""
    if not sentences:
        return []

    # Build document frequency (how many sentences contain each term)
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
        # Normalize TF by sentence length
        for t in tf:
            tf[t] /= len(tokens)
        # TF-IDF score for the sentence
        score = 0.0
        for t, freq in tf.items():
            idf = math.log((n_docs + 1) / (doc_freq.get(t, 0) + 1)) + 1
            score += freq * idf
        scores.append(score)
    return scores


def _compress_text(text: str, limit: int) -> str:
    """Compress long text using TF-IDF sentence scoring to preserve important content."""
    if limit <= 0:
        return ""
    txt = _normalize_text(text)
    if len(txt) <= limit:
        return txt

    sentences = [p.strip() for p in _SENTENCE_SPLIT_RE.split(txt) if p.strip()]

    # For very short texts or single sentences, fall back to truncation
    if len(sentences) <= 2:
        head = max(1, int(limit * 0.65))
        tail = max(1, limit - head - 5)
        return f"{txt[:head].rstrip()} ... {txt[-tail:].lstrip()}"

    # Score sentences by importance
    scores = _compute_sentence_scores(sentences)

    # Always keep first and last sentence (positional importance)
    # Then fill remaining budget with highest-scored middle sentences
    first = sentences[0]
    last = sentences[-1]
    base_len = len(first) + len(last) + 1  # +1 for space

    if base_len > limit:
        # Even first+last exceeds limit, truncate
        head = max(1, int(limit * 0.65))
        tail = max(1, limit - head - 5)
        return f"{txt[:head].rstrip()} ... {txt[-tail:].lstrip()}"

    remaining = limit - base_len
    # Rank middle sentences by score
    middle = [(i, sentences[i], scores[i]) for i in range(1, len(sentences) - 1)]
    middle.sort(key=lambda x: x[2], reverse=True)

    # Greedily add highest-scored sentences that fit
    selected_indices = {0, len(sentences) - 1}
    for idx, sent, _score in middle:
        cost = len(sent) + 1  # +1 for space
        if cost <= remaining:
            selected_indices.add(idx)
            remaining -= cost

    # Reassemble in original order
    result_parts = [sentences[i] for i in sorted(selected_indices)]
    return " ".join(result_parts)


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
        # Deprioritize short acknowledgments with low information density
        if _is_low_info_message(content):
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

    current_keywords: frozenset[str] = frozenset()

    for msg in messages:
        msg_sig = _signature(msg["content"])
        msg_tokens = estimate_tokens(msg["content"])
        msg_keywords = _extract_keywords(msg["content"])

        if current:
            jaccard_sim = _jaccard(current_sig, msg_sig)
            kw_sim = _keyword_overlap(current_keywords, msg_keywords)
            sim = 0.6 * jaccard_sim + 0.4 * kw_sim
        else:
            sim = 1.0

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
            current_keywords = frozenset()

        current.append(msg)
        current_tokens += msg_tokens
        current_sig = current_sig | msg_sig
        current_keywords = current_keywords | msg_keywords

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

