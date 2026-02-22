"""Core types for the OpenClaw memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

MemorySource = str  # "memory" | "sessions"


@dataclass
class MemorySearchResult:
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: MemorySource
    citation: str | None = None


@dataclass
class MemorySyncProgressUpdate:
    completed: int
    total: int
    label: str | None = None


@dataclass
class MemoryProviderStatus:
    backend: str  # "builtin"
    provider: str
    files: int
    chunks: int
    dirty: bool
    model: str | None = None
    requested_provider: str | None = None
    workspace_dir: str | None = None
    db_path: str | None = None
    extra_paths: list[str] = field(default_factory=list)
    sources: list[MemorySource] = field(default_factory=list)
    source_counts: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[str, Any] | None = None
    fts: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    vector: dict[str, Any] | None = None
    batch: dict[str, Any] | None = None
    custom: dict[str, Any] | None = None


@dataclass
class MemoryEmbeddingProbeResult:
    ok: bool
    error: str | None = None


@dataclass
class MemoryFileEntry:
    path: str  # workspace-relative
    abs_path: str
    mtime_ms: float
    size: int
    hash: str


@dataclass
class MemoryChunk:
    start_line: int
    end_line: int
    text: str
    hash: str


@runtime_checkable
class MemorySearchManagerProtocol(Protocol):
    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        min_score: float | None = None,
        session_key: str | None = None,
    ) -> list[MemorySearchResult]: ...

    def read_file(
        self,
        rel_path: str,
        *,
        from_line: int | None = None,
        lines: int | None = None,
    ) -> dict[str, str]: ...

    def status(self) -> MemoryProviderStatus: ...

    def sync(
        self,
        *,
        reason: str | None = None,
        force: bool = False,
        progress: Any | None = None,
    ) -> None: ...

    def close(self) -> None: ...
