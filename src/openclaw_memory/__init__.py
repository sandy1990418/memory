"""
openclaw-memory: Standalone Python implementation of the OpenClaw memory subsystem.

Public API:
    MemoryIndexManager — main class for search, sync, and file access
    MemorySearchResult — search result dataclass
    MemoryProviderStatus — index status dataclass
    resolve_memory_search_config — config resolution from env vars
"""

from .config import ResolvedMemorySearchConfig, resolve_memory_search_config
from .manager import MemoryIndexManager
from .types import (
    MemoryChunk,
    MemoryEmbeddingProbeResult,
    MemoryFileEntry,
    MemoryProviderStatus,
    MemorySearchResult,
    MemorySyncProgressUpdate,
)

__all__ = [
    "MemoryIndexManager",
    "MemorySearchResult",
    "MemoryProviderStatus",
    "MemoryChunk",
    "MemoryFileEntry",
    "MemoryEmbeddingProbeResult",
    "MemorySyncProgressUpdate",
    "ResolvedMemorySearchConfig",
    "resolve_memory_search_config",
]
