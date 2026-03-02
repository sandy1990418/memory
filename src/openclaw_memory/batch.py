"""
Buffered batch processor for memory extraction and storage.
Inspired by Memobase's buffered-flush pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .dedup import DBConnection, store_with_dedup
from .embeddings import EmbeddingProvider
from .extraction import ExtractedMemory, extract_memories
from .lightmem import (
    DistillPrepConfig,
    estimate_tokens,
    normalize_messages_use,
    prepare_messages_for_distill,
)


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
        token_buffer_threshold: int | None = None,
        distill_pre_compress: bool = False,
        distill_messages_use: str = "all",
        distill_topic_segment: bool = False,
        distill_max_tokens: int = 2200,
        distill_topic_threshold: int = 600,
    ) -> None:
        self.buffer_size = buffer_size
        self.llm_fn = llm_fn
        self.conn = conn
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.token_buffer_threshold = (
            max(64, int(token_buffer_threshold))
            if token_buffer_threshold is not None
            else None
        )

        # user_id -> list of message dicts
        self._buffer: dict[str, list[dict[str, Any]]] = {}
        self._buffer_tokens: dict[str, int] = {}

        # LightMem-style extraction input preparation for batched flushes.
        self._distill_config = DistillPrepConfig(
            pre_compress=distill_pre_compress,
            messages_use=normalize_messages_use(distill_messages_use),
            topic_segment=distill_topic_segment,
            max_input_tokens=max(64, int(distill_max_tokens)),
            topic_token_threshold=max(64, int(distill_topic_threshold)),
        )

    # ------------------------------------------------------------------
    # Buffering
    # ------------------------------------------------------------------

    def buffer_conversation(self, user_id: str, messages: list[dict[str, Any]]) -> None:
        """
        Add *messages* to the per-user buffer.

        Auto flush triggers when either:
          - buffered message count reaches ``buffer_size``, or
          - estimated token count reaches ``token_buffer_threshold`` (if set).
        """
        if user_id not in self._buffer:
            self._buffer[user_id] = []
            self._buffer_tokens[user_id] = 0

        self._buffer[user_id].extend(messages)
        added_tokens = sum(estimate_tokens(str(msg.get("content", ""))) for msg in messages)
        self._buffer_tokens[user_id] = self._buffer_tokens.get(user_id, 0) + added_tokens

        over_size = len(self._buffer[user_id]) >= self.buffer_size
        over_tokens = (
            self.token_buffer_threshold is not None
            and self._buffer_tokens.get(user_id, 0) >= self.token_buffer_threshold
        )
        if over_size or over_tokens:
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
        self._buffer_tokens.pop(user_id, None)
        if not messages:
            return []

        if self.llm_fn is None:
            raise RuntimeError("llm_fn must be set before flushing")

        # Build source refs from message timestamps or positional indices
        source_refs: list[str] = []
        for idx, msg in enumerate(messages):
            ts = msg.get("timestamp") or msg.get("ts")
            source_refs.append(str(ts) if ts is not None else str(idx))

        prepared = prepare_messages_for_distill(messages, config=self._distill_config)
        extraction_input: list[dict[str, Any]] = prepared if prepared else messages
        memories = extract_memories(extraction_input, self.llm_fn)

        # Propagate source_refs into memories that don't already have them
        for mem in memories:
            if not mem.source_refs:
                mem.source_refs = list(source_refs)

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

    def buffered_tokens_for(self, user_id: str) -> int:
        """Return the estimated buffered token count for *user_id*."""
        return self._buffer_tokens.get(user_id, 0)

    def buffered_users(self) -> list[str]:
        """Return the list of user IDs that have buffered messages."""
        return list(self._buffer.keys())
