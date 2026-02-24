"""
Redis-based L1 working memory (short-term, per-session context).

Provides:
  - WorkingMemory — wraps Redis with per-user/thread key scoping and TTL.
    Gracefully degrades (returns None / no-ops) if Redis is unavailable.
"""

from __future__ import annotations

import json
from typing import Any


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

    def get(self, user_id: str, thread_id: str) -> list[dict] | None:
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
            return json.loads(raw)
        except Exception:
            return None

    def set(self, user_id: str, thread_id: str, messages: list[dict]) -> None:
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
