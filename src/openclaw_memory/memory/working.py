"""
L1: Working memory — session-scoped, PostgreSQL-backed.

Stores the most recent messages for the current conversation session,
scoped by (user_id, session_id). Provides the immediate context that
the chatbot needs for coherent responses.

Replaces the old Redis + DB dual backend with a single PG-backed
implementation using the connection pool.
"""

from __future__ import annotations

from typing import Any

import psycopg

_MAX_MESSAGES = 20

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_SQL_INSERT = """
    INSERT INTO working_messages (user_id, session_id, role, content)
    VALUES (%s, %s, %s, %s)
"""

_SQL_PRUNE = """
    DELETE FROM working_messages
    WHERE user_id = %s AND session_id = %s
      AND id NOT IN (
          SELECT id FROM working_messages
          WHERE user_id = %s AND session_id = %s
          ORDER BY created_at DESC
          LIMIT %s
      )
"""

_SQL_SELECT = """
    SELECT role, content, created_at
    FROM (
        SELECT role, content, created_at
        FROM working_messages
        WHERE user_id = %s AND session_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    ) sub
    ORDER BY created_at ASC
"""

_SQL_DELETE_SESSION = """
    DELETE FROM working_messages
    WHERE user_id = %s AND session_id = %s
"""

_SQL_DELETE_USER = """
    DELETE FROM working_messages
    WHERE user_id = %s
"""

_SQL_COUNT = """
    SELECT COUNT(*) FROM working_messages
    WHERE user_id = %s AND session_id = %s
"""


# ---------------------------------------------------------------------------
# Working Memory Manager
# ---------------------------------------------------------------------------


class WorkingMemory:
    """
    PostgreSQL-backed working memory scoped by (user_id, session_id).

    Stores the most recent messages and returns them in chronological
    order for prompt injection. Per-session message count is capped
    at max_messages; older messages are pruned on each append.
    """

    def __init__(self, max_messages: int = _MAX_MESSAGES) -> None:
        self._max_messages = max_messages

    def append(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Insert a message and prune excess rows for this session."""
        with conn.cursor() as cur:
            cur.execute(_SQL_INSERT, (user_id, session_id, role, content))
            cur.execute(
                _SQL_PRUNE,
                (user_id, session_id, user_id, session_id, self._max_messages),
            )

    def get_recent(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent messages in chronological order (oldest first)."""
        effective_limit = limit or self._max_messages
        with conn.cursor() as cur:
            cur.execute(_SQL_SELECT, (user_id, session_id, effective_limit))
            rows = cur.fetchall()
        return [
            {"role": r[0], "content": r[1], "created_at": r[2]}
            for r in rows
        ]

    def clear_session(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
    ) -> None:
        """Delete all working-memory messages for a specific session."""
        with conn.cursor() as cur:
            cur.execute(_SQL_DELETE_SESSION, (user_id, session_id))

    def count(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
        session_id: str,
    ) -> int:
        """Return the number of messages for a session."""
        with conn.cursor() as cur:
            cur.execute(_SQL_COUNT, (user_id, session_id))
            return cur.fetchone()[0]

    def clear_user(
        self,
        conn: psycopg.Connection[Any],
        user_id: str,
    ) -> None:
        """Delete all working-memory messages for a user."""
        with conn.cursor() as cur:
            cur.execute(_SQL_DELETE_USER, (user_id,))

    def to_search_results(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Convert working-memory messages to search result dicts."""
        results: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue
            results.append({
                "id": f"wm:{user_id}:{session_id}:{i}",
                "snippet": content,
                "source": "working_memory",
                "memory_type": "working",
                "vector_score": 1.0,
                "text_score": 1.0,
                "score": 1.0,
                "created_at": msg.get("created_at"),
            })
        return results
