"""
Short-term memory buffer for topic-aware message accumulation.

Buffers incoming messages in PostgreSQL before triggering LLM extraction,
reducing extraction calls by batching messages until a flush threshold is
reached (token count, message count, or age).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .lightmem import estimate_tokens

if TYPE_CHECKING:
    import psycopg


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferConfig:
    """Thresholds that control when the short-term buffer should flush."""

    flush_token_threshold: int = 1500
    flush_message_threshold: int = 15
    max_age_seconds: int = 3600
    topic_similarity_threshold: float = 0.3


# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_SQL_UPSERT = """
INSERT INTO short_term_buffer (id, user_id, thread_id, messages, topic_summary, token_count, created_at, updated_at)
VALUES (%s, %s, %s, %s::jsonb, %s, %s, now(), now())
"""

_SQL_SELECT_USER = """
SELECT id, messages, token_count, created_at, updated_at
  FROM short_term_buffer
 WHERE user_id = %s
 ORDER BY updated_at DESC
 LIMIT 1
"""

_SQL_UPDATE_BUFFER = """
UPDATE short_term_buffer
   SET messages = %s::jsonb,
       token_count = %s,
       topic_summary = %s,
       updated_at = now()
 WHERE id = %s
"""

_SQL_DELETE_BY_ID = """
DELETE FROM short_term_buffer WHERE id = %s
"""

_SQL_DELETE_BY_USER = """
DELETE FROM short_term_buffer WHERE user_id = %s
"""

_SQL_SELECT_EXPIRED = """
SELECT id, user_id, messages, token_count
  FROM short_term_buffer
 WHERE created_at < (now() - make_interval(secs => %s))
 ORDER BY created_at ASC
"""


# ---------------------------------------------------------------------------
# ShortTermBuffer
# ---------------------------------------------------------------------------


class ShortTermBuffer:
    """Topic-aware short-term memory buffer (LightMem Light2 stage)."""

    def __init__(
        self,
        conn_factory: Any,
        *,
        config: BufferConfig | None = None,
    ) -> None:
        self._conn_factory = conn_factory
        self._config = config or BufferConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        *,
        thread_id: str | None = None,
    ) -> None:
        """Add messages to the user's buffer, creating or updating a row."""
        if not messages:
            return

        new_text = " ".join(m.get("content", "") for m in messages)
        new_tokens = estimate_tokens(new_text)

        conn = self._conn_factory()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(_SQL_SELECT_USER, (user_id,))
                row = cur.fetchone()

                if row is not None:
                    buf_id, existing_msgs_raw, existing_tokens, _, _ = row
                    existing_msgs = (
                        json.loads(existing_msgs_raw)
                        if isinstance(existing_msgs_raw, str)
                        else existing_msgs_raw
                    )
                    merged = existing_msgs + messages
                    total_tokens = existing_tokens + new_tokens
                    topic = self._derive_topic(merged)
                    cur.execute(
                        _SQL_UPDATE_BUFFER,
                        (json.dumps(merged), total_tokens, topic, buf_id),
                    )
                else:
                    buf_id = str(uuid.uuid4())
                    topic = self._derive_topic(messages)
                    cur.execute(
                        _SQL_UPSERT,
                        (
                            buf_id,
                            user_id,
                            thread_id,
                            json.dumps(messages),
                            topic,
                            new_tokens,
                        ),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def should_flush(self, user_id: str) -> bool:
        """Check if buffer should be flushed based on thresholds."""
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(_SQL_SELECT_USER, (user_id,))
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return False

        _, msgs_raw, token_count, created_at, _ = row
        msgs = (
            json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
        )

        cfg = self._config

        if token_count >= cfg.flush_token_threshold:
            return True

        if len(msgs) >= cfg.flush_message_threshold:
            return True

        if created_at is not None:
            if hasattr(created_at, "timestamp"):
                age = time.time() - created_at.timestamp()
            else:
                age = 0.0
            if age >= cfg.max_age_seconds:
                return True

        return False

    def flush(self, user_id: str) -> list[dict[str, Any]]:
        """Extract and clear buffered messages for *user_id*."""
        conn = self._conn_factory()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(_SQL_SELECT_USER, (user_id,))
                row = cur.fetchone()
                if row is None:
                    conn.commit()
                    return []

                buf_id, msgs_raw, _, _, _ = row
                msgs = (
                    json.loads(msgs_raw)
                    if isinstance(msgs_raw, str)
                    else msgs_raw
                )
                cur.execute(_SQL_DELETE_BY_ID, (buf_id,))
            conn.commit()
            return msgs
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def peek(self, user_id: str) -> tuple[list[dict[str, Any]], int]:
        """Check buffer contents without flushing. Returns (messages, token_count)."""
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(_SQL_SELECT_USER, (user_id,))
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return [], 0

        _, msgs_raw, token_count, _, _ = row
        msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
        return msgs, token_count

    def flush_all_expired(self) -> dict[str, list[dict[str, Any]]]:
        """Flush all buffers that have exceeded age threshold. Returns {user_id: messages}."""
        cfg = self._config
        result: dict[str, list[dict[str, Any]]] = {}

        conn = self._conn_factory()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(_SQL_SELECT_EXPIRED, (cfg.max_age_seconds,))
                rows = cur.fetchall()
                for buf_id, uid, msgs_raw, _ in rows:
                    msgs = (
                        json.loads(msgs_raw)
                        if isinstance(msgs_raw, str)
                        else msgs_raw
                    )
                    if uid in result:
                        result[uid].extend(msgs)
                    else:
                        result[uid] = list(msgs)
                    cur.execute(_SQL_DELETE_BY_ID, (buf_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_topic(messages: list[dict[str, Any]]) -> str:
        """Build a simple topic summary from the first few messages."""
        parts: list[str] = []
        for m in messages[:3]:
            content = m.get("content", "")
            if content:
                parts.append(content[:80])
        return " | ".join(parts) if parts else ""
