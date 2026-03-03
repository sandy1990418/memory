"""
Token-tracking wrapper for LLM calls.

Wraps any ``Callable[[str], str]`` to record input/output token counts,
timing, and operation type — without changing the call signature.

Usage::

    tracker = TokenTracker(my_llm_fn, model="gpt-4o-mini")
    # Pass tracker wherever llm_fn is expected:
    result = extract_memories(conversation, llm_fn=tracker)
    # Inspect usage:
    print(tracker.total_input_tokens, tracker.total_output_tokens)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .tokens import estimate_tokens


@dataclass
class LLMCallRecord:
    """Single LLM call record."""

    input_tokens: int
    output_tokens: int
    operation: str = ""
    duration_ms: float = 0.0


class TokenTracker:
    """
    Transparent wrapper around ``llm_fn: Callable[[str], str]``.

    Intercepts each call to count tokens and record timing.
    Compatible with all existing call sites since it preserves
    the ``Callable[[str], str]`` signature.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        *,
        model: str = "gpt-4o-mini",
        operation: str = "",
    ) -> None:
        self._llm_fn = llm_fn
        self.model = model
        self.operation = operation
        self.calls: list[LLMCallRecord] = []

    def __call__(self, prompt: str) -> str:
        input_tokens = estimate_tokens(prompt)
        start = time.monotonic()
        try:
            response = self._llm_fn(prompt)
        except Exception:
            # Record even failed calls (0 output tokens)
            elapsed = (time.monotonic() - start) * 1000
            self.calls.append(LLMCallRecord(
                input_tokens=input_tokens,
                output_tokens=0,
                operation=self.operation,
                duration_ms=elapsed,
            ))
            raise
        elapsed = (time.monotonic() - start) * 1000
        output_tokens = estimate_tokens(response)
        self.calls.append(LLMCallRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=self.operation,
            duration_ms=elapsed,
        ))
        return response

    def with_operation(self, operation: str) -> TokenTracker:
        """Return a new tracker sharing the same call log but with a different operation label."""
        t = TokenTracker(self._llm_fn, model=self.model, operation=operation)
        t.calls = self.calls  # shared list
        return t

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def summary(self) -> dict:
        """Return usage summary grouped by operation."""
        by_op: dict[str, dict] = {}
        for c in self.calls:
            op = c.operation or "unknown"
            if op not in by_op:
                by_op[op] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "duration_ms": 0.0}
            by_op[op]["calls"] += 1
            by_op[op]["input_tokens"] += c.input_tokens
            by_op[op]["output_tokens"] += c.output_tokens
            by_op[op]["duration_ms"] += c.duration_ms
        return {
            "model": self.model,
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "by_operation": by_op,
        }

    def reset(self) -> None:
        """Clear all recorded calls."""
        self.calls.clear()
