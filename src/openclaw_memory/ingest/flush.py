"""
LLM-based memory extraction from conversations.
Mirrors: src/auto-reply/reply/memory-flush.ts (simplified: no agent, direct LLM call).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_MEMORY_FLUSH_PROMPT = (
    "Pre-compaction memory flush.\n"
    "以下是一段對話歷史。請萃取值得長期記住的事實、偏好、決定。\n"
    "格式：每條一行 markdown bullet point。\n"
    "如果沒有值得記的，回覆「無」。\n\n"
    "{conversation}"
)


@dataclass
class MemoryFlushConfig:
    enabled: bool = True
    threshold_ratio: float = 0.75
    context_window: int = 128_000
    prompt: str = DEFAULT_MEMORY_FLUSH_PROMPT


def should_flush(
    conversation: list[dict[str, str]],
    context_window: int = 128_000,
    threshold_ratio: float = 0.75,
) -> bool:
    """Return True when the conversation is large enough to warrant a memory flush."""
    total_chars = sum(len(msg.get("content", "")) for msg in conversation)
    total_tokens = total_chars // 4  # rough estimate
    return total_tokens >= context_window * threshold_ratio


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def memory_flush(
    conversation: list[dict[str, str]],
    workspace_dir: str,
    llm_fn: Callable[[str], str],
    *,
    date_str: str | None = None,
    prompt_template: str = DEFAULT_MEMORY_FLUSH_PROMPT,
) -> str | None:
    """
    Extract durable memories from *conversation* via an LLM call and append
    them to the daily memory file ``memory/YYYY-MM-DD.md``.

    Returns the absolute path of the written file, or ``None`` when the LLM
    indicates there is nothing worth storing.
    """
    formatted = _format_conversation(conversation)
    prompt = prompt_template.replace("{conversation}", formatted)
    result = llm_fn(prompt)

    # If the LLM says nothing to store, bail out.
    if _is_empty_result(result):
        return None

    now = datetime.now(timezone.utc)
    if date_str is None:
        date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    memory_dir = os.path.join(workspace_dir, "memory")
    os.makedirs(memory_dir, exist_ok=True)

    file_path = os.path.join(memory_dir, f"{date_str}.md")
    section = f"\n## {time_str}\n\n{result.rstrip()}\n"

    with open(file_path, "a", encoding="utf-8") as fh:
        fh.write(section)

    return file_path


def _is_empty_result(text: str) -> bool:
    """Return True when the LLM response indicates nothing to store."""
    stripped = text.strip()
    if not stripped:
        return True
    if "無" in stripped and len(stripped) < 20:
        return True
    return False
