"""
Working memory backends for short-term, per-user context.

Provides:
  - WorkingMemory     — Redis-backed, per-user/thread, TTL-based.
    Gracefully degrades (returns None / no-ops) if Redis is unavailable.
  - DBWorkingMemory   — PostgreSQL-backed, per-user, max N messages.
    Inserts messages, retrieves the most recent N in chronological order,
    and prunes older messages when the per-user cap is exceeded.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

_MAX_WORKING_MESSAGES = 20


class WorkingMemory:
    """
    Redis-backed working memory with TTL.

    Keys are scoped as: ``wm:{user_id}:{thread_id}``

    Graceful degradation: if Redis is unavailable at construction time or
    during any operation, the operation silently returns None / no-ops
    instead of raising.
    """

    def __init__(self, redis_url: str, ttl: int = 1800) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: Any = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            import redis  # lazy import

            client = redis.Redis.from_url(self._redis_url, decode_responses=True)
            # Verify connection
            client.ping()
            self._client = client
            self._available = True
        except Exception:
            self._client = None
            self._available = False

    @staticmethod
    def _key(user_id: str, thread_id: str) -> str:
        return f"wm:{user_id}:{thread_id}"

    def get(self, user_id: str, thread_id: str) -> list[dict[str, Any]] | None:
        """
        Retrieve messages for a given user/thread.

        Returns None if unavailable or no data stored.
        """
        if not self._available or self._client is None:
            return None
        try:
            raw = self._client.get(self._key(user_id, thread_id))
            if raw is None:
                return None
            result: list[dict[str, Any]] = json.loads(raw)
            return result
        except Exception:
            return None

    def set(self, user_id: str, thread_id: str, messages: list[dict[str, Any]]) -> None:
        """
        Store messages for a given user/thread with TTL.

        No-ops silently if Redis is unavailable.
        """
        if not self._available or self._client is None:
            return
        try:
            self._client.setex(
                self._key(user_id, thread_id),
                self._ttl,
                json.dumps(messages),
            )
        except Exception:
            pass

    def delete(self, user_id: str, thread_id: str) -> None:
        """
        Delete messages for a given user/thread.

        No-ops silently if Redis is unavailable.
        """
        if not self._available or self._client is None:
            return
        try:
            self._client.delete(self._key(user_id, thread_id))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DB-backed working memory (PostgreSQL)
# ---------------------------------------------------------------------------

_SQL_INSERT = """
    INSERT INTO working_messages (user_id, role, content)
    VALUES (%s, %s, %s)
"""

_SQL_PRUNE = """
    DELETE FROM working_messages
    WHERE user_id = %s
      AND id NOT IN (
          SELECT id FROM working_messages
          WHERE user_id = %s
          ORDER BY created_at DESC
          LIMIT %s
      )
"""

_SQL_SELECT_RECENT = """
    SELECT role, content, created_at
    FROM (
        SELECT role, content, created_at
        FROM working_messages
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    ) sub
    ORDER BY created_at ASC
"""

_SQL_DELETE_USER = """
    DELETE FROM working_messages
    WHERE user_id = %s
"""


class DBWorkingMemory:
    """
    PostgreSQL-backed working memory scoped by user_id only.

    Stores the most recent messages and returns them in chronological order
    (oldest to newest) for prompt injection.  Per-user message count is capped
    at ``max_messages`` (default 20); older messages are pruned on each append.

    Args:
        conn_factory: A zero-argument callable that returns an open
                      psycopg3 ``Connection``.  Typically a lambda wrapping
                      ``get_pg_connection(dsn)``.
        max_messages: Maximum messages to keep per user (default 20).
    """

    def __init__(
        self,
        conn_factory: Any,
        max_messages: int = _MAX_WORKING_MESSAGES,
    ) -> None:
        self._conn_factory = conn_factory
        self._max_messages = max_messages

    def append(self, user_id: str, role: str, content: str) -> None:
        """
        Insert a message for *user_id* and prune any excess rows.

        Pruning keeps the most recent ``max_messages`` rows; older rows
        are deleted in the same connection.
        """
        conn = self._conn_factory()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(_SQL_INSERT, (user_id, role, content))
                cur.execute(_SQL_PRUNE, (user_id, user_id, self._max_messages))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_recent(self, user_id: str, limit: int = _MAX_WORKING_MESSAGES) -> list[dict[str, Any]]:
        """
        Return up to *limit* most recent messages in chronological order.

        Each item is a dict with keys ``role``, ``content``, and ``created_at``.
        """
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(_SQL_SELECT_RECENT, (user_id, limit))
                rows = cur.fetchall()
        finally:
            conn.close()

        return [
            {"role": row[0], "content": row[1], "created_at": row[2]}
            for row in rows
        ]

    def delete(self, user_id: str) -> None:
        """Delete all working-memory messages for *user_id*."""
        conn = self._conn_factory()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(_SQL_DELETE_USER, (user_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
