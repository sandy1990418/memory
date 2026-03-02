"""
openclaw-memory: Standalone Python implementation of the OpenClaw memory subsystem.

Public API:
    MemoryIndexManager — main class for search, sync, and file access
    MemorySearchResult — search result dataclass
    MemoryProviderStatus — index status dataclass
    resolve_memory_search_config — config resolution from env vars
"""

from .config import ResolvedMemorySearchConfig, resolve_memory_search_config
from .consolidation import ConsolidationReport, MemoryConsolidator
from .manager import MemoryIndexManager
from .short_term_buffer import BufferConfig, ShortTermBuffer
from .types import (
    MemoryChunk,
    MemoryContext,
    MemoryEmbeddingProbeResult,
    MemoryFileEntry,
    MemoryIndex,
    MemoryProviderStatus,
    MemorySearchResult,
    MemorySyncProgressUpdate,
)

__all__ = [
    "MemoryIndexManager",
    "MemorySearchResult",
    "MemoryProviderStatus",
    "MemoryChunk",
    "MemoryContext",
    "MemoryFileEntry",
    "MemoryEmbeddingProbeResult",
    "MemoryIndex",
    "MemorySyncProgressUpdate",
    "ResolvedMemorySearchConfig",
    "resolve_memory_search_config",
    "BufferConfig",
    "ShortTermBuffer",
    "ConsolidationReport",
    "MemoryConsolidator",
]
