"""
Save conversation turns as markdown files for the memory sync pipeline.
Mirrors: src/hooks/bundled/session-memory/handler.ts
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import datetime, timezone

from .internal import ensure_dir

__all__ = [
    "save_session_to_memory",
    "generate_session_slug",
    "sanitize_slug",
]


def sanitize_slug(raw: str) -> str:
    """Make a string filesystem-safe: lowercase, hyphens only, max 40 chars."""
    lowered = raw.lower().strip()
    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    # Collapse multiple hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:40]


def generate_session_slug(
    conversation: list[dict[str, str]],
    *,
    session_id: str = "",
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """Generate a slug for the session file name.

    If *llm_fn* is provided, call it with recent conversation text to produce a
    descriptive slug.  Otherwise fall back to the first 8 characters of
    *session_id*.
    """
    if llm_fn is not None:
        recent_text = "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
            for turn in conversation[-5:]
        )
        try:
            raw_slug = llm_fn(recent_text)
            slug = sanitize_slug(raw_slug)
            if slug:
                return slug
        except Exception:
            pass

    # Fallback
    fallback = session_id[:8] if session_id else "session"
    return sanitize_slug(fallback)


def save_session_to_memory(
    conversation: list[dict[str, str]],
    session_id: str,
    workspace_dir: str,
    *,
    max_turns: int = 15,
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """Write the last *max_turns* conversation turns to a dated markdown file.

    Returns the absolute path of the written file.
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    slug = generate_session_slug(
        conversation, session_id=session_id, llm_fn=llm_fn,
    )
    filename = f"{date_str}-{slug}.md"

    sessions_dir = os.path.join(workspace_dir, "memory", "sessions")
    ensure_dir(sessions_dir)

    dest = os.path.join(sessions_dir, filename)

    # Truncate to last max_turns
    turns = conversation[-max_turns:] if max_turns > 0 else []

    lines: list[str] = [
        f"# Session: {date_str} {time_str} UTC",
        "",
        f"- **Session ID**: {session_id}",
        "",
        "## Conversation",
        "",
    ]
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        lines.append(f"**{role}**: {content}")
        lines.append("")

    content_str = "\n".join(lines)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content_str)

    return dest
