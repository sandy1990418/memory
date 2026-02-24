"""
Buffered batch processor for memory extraction and storage.
Inspired by Memobase's buffered-flush pattern.
"""

from __future__ import annotations

from collections.abc import Callable

from .dedup import DBConnection, store_with_dedup
from .embeddings import EmbeddingProvider
from .extraction import ExtractedMemory, extract_memories


class MemoryBatchProcessor:
    """
    Buffer incoming conversation messages per user and flush in batches.

    Extraction + classification + dedup + store are all triggered on flush.
    """

    def __init__(
        self,
        buffer_size: int = 10,
        *,
        llm_fn: Callable[[str], str] | None = None,
        conn: DBConnection | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.buffer_size = buffer_size
        self.llm_fn = llm_fn
        self.conn = conn
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold

        # user_id -> list of message dicts
        self._buffer: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Buffering
    # ------------------------------------------------------------------

    def buffer_conversation(self, user_id: str, messages: list[dict]) -> None:
        """
        Add *messages* to the per-user buffer.
        Automatically flushes when the buffer reaches *buffer_size*.
        """
        if user_id not in self._buffer:
            self._buffer[user_id] = []
        self._buffer[user_id].extend(messages)

        if len(self._buffer[user_id]) >= self.buffer_size:
            self.flush(user_id)

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def flush(self, user_id: str) -> list[ExtractedMemory]:
        """
        Extract, classify, dedup, and store all buffered messages for *user_id*.

        Returns the list of extracted memories (empty if buffer was empty or
        no memories were found). The buffer for this user is cleared afterwards.
        """
        messages = self._buffer.pop(user_id, [])
        if not messages:
            return []

        if self.llm_fn is None:
            raise RuntimeError("llm_fn must be set before flushing")

        memories = extract_memories(messages, self.llm_fn)

        if memories and self.conn is not None and self.embedding_provider is not None:
            for mem in memories:
                store_with_dedup(
                    self.conn,
                    user_id,
                    mem,
                    self.embedding_provider,
                    self.similarity_threshold,
                )

        return memories

    def flush_all(self) -> dict[str, list[ExtractedMemory]]:
        """
        Flush all users with buffered messages.

        Returns a mapping of user_id -> extracted memories.
        Suitable for shutdown / periodic cron jobs.
        """
        user_ids = list(self._buffer.keys())
        results: dict[str, list[ExtractedMemory]] = {}
        for user_id in user_ids:
            results[user_id] = self.flush(user_id)
        return results

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def buffer_size_for(self, user_id: str) -> int:
        """Return the current number of buffered messages for *user_id*."""
        return len(self._buffer.get(user_id, []))

    def buffered_users(self) -> list[str]:
        """Return the list of user IDs that have buffered messages."""
        return list(self._buffer.keys())
