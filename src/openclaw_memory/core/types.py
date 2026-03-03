"""
Core domain types for the memory system.

Replaces the old types.py with Pydantic-compatible dataclasses
and cleaner session-scoped semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# -- Search result types (tiered: compact -> context -> full) --


@dataclass
class MemoryIndex:
    """Tier 1: Lightweight index entry (~50 tokens)."""

    id: str
    title: str
    memory_type: str  # preference / fact / decision / event
    score: float
    created_at: str
    source: str  # episodic / semantic / canonical


@dataclass
class MemoryContext:
    """Tier 2: Timeline context (~200 tokens)."""

    id: str
    content: str
    memory_type: str
    score: float
    created_at: str
    source: str
    neighbors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MemorySearchResult:
    """Tier 3: Full memory content."""

    id: str
    content: str
    memory_type: str
    score: float
    created_at: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


# -- Extracted memory (from LLM) --


@dataclass
class ExtractedMemory:
    """A memory extracted by the LLM from a conversation."""

    content: str
    memory_type: str  # preference / fact / decision / event
    confidence: float  # 0.0 - 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_key: str = ""
    value: str = ""
    event_time: str | None = None
    source_refs: list[str] = field(default_factory=list)


# -- Consolidation --


@dataclass
class ConsolidationReport:
    """Summary of a single user's consolidation run."""

    user_id: str
    memories_scanned: int = 0
    memories_merged: int = 0
    memories_deleted: int = 0
    memories_abstracted: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
