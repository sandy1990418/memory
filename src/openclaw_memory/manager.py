"""
MemoryIndexManager — core search, sync, and file read operations.
Mirrors: src/memory/manager.ts
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ResolvedMemorySearchConfig, resolve_memory_search_config
from .embeddings import EmbeddingProvider, create_embedding_provider
from .hybrid import merge_hybrid_results
from .internal import (
    is_memory_path,
    normalize_extra_memory_paths,
)
from .manager_search import search_keyword, search_vector
from .query_expansion import extract_keywords
from .schema import ensure_memory_index_schema
from .sync import sync_memory_files
from .types import MemoryEmbeddingProbeResult, MemoryProviderStatus, MemorySearchResult

_SNIPPET_MAX_CHARS = 700
_FTS_TABLE = "chunks_fts"
_EMBEDDING_CACHE_TABLE = "embedding_cache"


class MemoryIndexManager:
    """
    Manages the SQLite-backed memory index.
    Thread-safe for reads; single-writer sync via a lock.
    Mirrors: src/memory/manager.ts::MemoryIndexManager
    """

    def __init__(
        self,
        workspace_dir: str,
        db_path: str,
        *,
        config: ResolvedMemorySearchConfig | None = None,
        provider: EmbeddingProvider | None = None,
        provider_meta: dict[str, Any] | None = None,
    ) -> None:
        # Resolve symlinks so path comparisons work consistently (e.g. macOS /tmp → /private/tmp)
        self.workspace_dir = os.path.realpath(workspace_dir)
        self.db_path = db_path
        self._config = config or resolve_memory_search_config(
            workspace_dir=workspace_dir, overrides={"db_path": db_path}
        )
        self._provider = provider
        self._provider_meta = provider_meta or {}

        # Determine FTS enabled from config
        self._fts_enabled = self._config.query.hybrid.enabled

        # Open DB
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")

        schema_result = ensure_memory_index_schema(
            self._db,
            embedding_cache_table=_EMBEDDING_CACHE_TABLE,
            fts_table=_FTS_TABLE,
            fts_enabled=self._fts_enabled,
        )
        self._fts_available: bool = bool(schema_result.get("fts_available", False))
        self._fts_error: str | None = schema_result.get("fts_error")  # type: ignore[assignment]

        self._dirty = True
        self._sync_lock = threading.Lock()
        self._closed = False

        # Determine provider model label
        self._model = self._provider.model if self._provider else (self._config.model or "none")
        self._provider_id = self._provider.id if self._provider else "none"
        # Compute a provider key hash for cache isolation across different endpoints
        self._provider_key = self._compute_provider_key()

    def _compute_provider_key(self) -> str:
        """Compute a hash of provider config for cache isolation."""
        import hashlib

        parts = [self._provider_id, self._model]
        if self._config.remote and self._config.remote.base_url:
            parts.append(self._config.remote.base_url)
        raw = ":".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        agent_id: str = "main",
        workspace_dir: str | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> MemoryIndexManager:
        """
        Create a MemoryIndexManager with auto-selected embedding provider.
        """
        config = resolve_memory_search_config(
            agent_id, workspace_dir=workspace_dir, overrides=overrides
        )
        # Re-resolve with proper workspace
        config = resolve_memory_search_config(
            agent_id, workspace_dir=workspace_dir, overrides=overrides
        )

        provider, meta = create_embedding_provider(
            provider=config.provider,
            model=config.model,
            fallback=config.fallback,
            remote=config.remote,
            local=config.local,
        )

        ws = workspace_dir or os.environ.get("OPENCLAW_WORKSPACE") or os.getcwd()
        db_path = config.store.path

        return cls(ws, db_path, config=config, provider=provider, provider_meta=meta)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        min_score: float | None = None,
        session_key: str | None = None,
    ) -> list[MemorySearchResult]:
        """
        Full hybrid search pipeline.
        Mirrors: manager.ts::search
        """
        if self._closed:
            raise RuntimeError("MemoryIndexManager is closed")

        query = query.strip()
        if not query:
            return []

        cfg_query = self._config.query
        actual_max_results = max_results if max_results is not None else cfg_query.max_results
        actual_min_score = min_score if min_score is not None else cfg_query.min_score
        hybrid_cfg = cfg_query.hybrid
        candidates = min(200, max(1, int(actual_max_results * hybrid_cfg.candidate_multiplier)))

        source_filter = self._build_source_filter()

        # FTS-only mode (no embedding provider)
        if self._provider is None:
            if not self._fts_enabled or not self._fts_available:
                return []

            keywords = extract_keywords(query)
            search_terms = keywords if keywords else [query]

            seen: dict[str, dict[str, Any]] = {}
            for term in search_terms:
                kw_results = search_keyword(
                    self._db,
                    fts_table=_FTS_TABLE,
                    provider_model=None,
                    query=term,
                    limit=candidates,
                    snippet_max_chars=_SNIPPET_MAX_CHARS,
                    source_filter=source_filter,
                )
                for r in kw_results:
                    existing = seen.get(r["id"])
                    if not existing or r["score"] > existing["score"]:
                        seen[r["id"]] = r

            merged = sorted(seen.values(), key=lambda r: r["score"], reverse=True)
            filtered = [r for r in merged if r["score"] >= actual_min_score][:actual_max_results]
            return [self._to_search_result(r) for r in filtered]

        # Hybrid mode
        keyword_results: list[dict[str, Any]] = []
        if hybrid_cfg.enabled and self._fts_available:
            keyword_results = search_keyword(
                self._db,
                fts_table=_FTS_TABLE,
                provider_model=self._model,
                query=query,
                limit=candidates,
                snippet_max_chars=_SNIPPET_MAX_CHARS,
                source_filter=source_filter,
            )

        # Embed query
        try:
            query_vec = self._provider.embed_query(query)
        except Exception:
            query_vec = []

        has_vector = any(v != 0 for v in query_vec)
        vector_results: list[dict[str, Any]] = []
        if has_vector:
            vector_results = search_vector(
                self._db,
                provider_model=self._model,
                query_vec=query_vec,
                limit=candidates,
                snippet_max_chars=_SNIPPET_MAX_CHARS,
                source_filter=source_filter,
            )

        if not hybrid_cfg.enabled:
            filtered = [r for r in vector_results if r["score"] >= actual_min_score][
                :actual_max_results
            ]
            return [self._to_search_result(r) for r in filtered]

        merged_list = merge_hybrid_results(
            vector=vector_results,
            keyword=keyword_results,
            vector_weight=hybrid_cfg.vector_weight,
            text_weight=hybrid_cfg.text_weight,
            mmr={"enabled": hybrid_cfg.mmr.enabled, "lambda_": hybrid_cfg.mmr.lambda_}
            if hybrid_cfg.mmr.enabled
            else None,
            temporal_decay={
                "enabled": hybrid_cfg.temporal_decay.enabled,
                "half_life_days": hybrid_cfg.temporal_decay.half_life_days,
            }
            if hybrid_cfg.temporal_decay.enabled
            else None,
            workspace_dir=self.workspace_dir,
        )

        filtered = [r for r in merged_list if r["score"] >= actual_min_score][:actual_max_results]
        return [self._to_search_result(r) for r in filtered]

    def _to_search_result(self, r: dict[str, Any]) -> MemorySearchResult:
        return MemorySearchResult(
            path=r["path"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            score=r["score"],
            snippet=r.get("snippet", ""),
            source=r.get("source", "memory"),
        )

    def _build_source_filter(self) -> tuple[str, list[str]] | None:
        sources = list(self._config.sources)
        if not sources:
            return None
        if len(sources) == 1:
            return (" AND source = ?", [sources[0]])
        placeholders = ", ".join("?" * len(sources))
        return (f" AND source IN ({placeholders})", sources)

    # ------------------------------------------------------------------
    # Read file
    # ------------------------------------------------------------------

    def read_file(
        self,
        rel_path: str,
        *,
        from_line: int | None = None,
        lines: int | None = None,
    ) -> dict[str, str]:
        """
        Read a memory file safely.
        Only allows files under workspace memory/ or extra_paths.
        Mirrors: manager.ts::readFile
        """
        if self._closed:
            raise RuntimeError("MemoryIndexManager is closed")

        raw_path = rel_path.strip()
        if not raw_path:
            raise ValueError("path required")

        # Resolve to absolute WITHOUT following symlinks (use abspath, not realpath)
        if os.path.isabs(raw_path):
            abs_path = os.path.abspath(raw_path)
        else:
            abs_path = os.path.abspath(os.path.join(self.workspace_dir, raw_path))

        # Symlink guard BEFORE any further checks — detect symlinks on the
        # un-resolved path so a symlink inside memory/ can't escape the workspace
        if os.path.islink(abs_path) or not os.path.isfile(abs_path):
            raise ValueError("path required")

        if not abs_path.endswith(".md"):
            raise ValueError("path required")

        # Compute relative path from workspace
        try:
            rel = os.path.relpath(abs_path, self.workspace_dir).replace("\\", "/")
        except ValueError:
            rel = ""

        in_workspace = rel and not rel.startswith("..") and not os.path.isabs(rel)
        allowed = in_workspace and is_memory_path(rel)

        # Check extra paths
        if not allowed and self._config.extra_paths:
            extra = normalize_extra_memory_paths(self.workspace_dir, self._config.extra_paths)
            for extra_path in extra:
                if os.path.islink(extra_path):
                    continue
                if os.path.isdir(extra_path):
                    if abs_path == extra_path or abs_path.startswith(extra_path + os.sep):
                        allowed = True
                        break
                elif os.path.isfile(extra_path) and extra_path.endswith(".md"):
                    if abs_path == extra_path:
                        allowed = True
                        break

        if not allowed:
            raise ValueError("path required")

        content = Path(abs_path).read_text(encoding="utf-8")

        if from_line is None and lines is None:
            return {"text": content, "path": rel}

        file_lines = content.split("\n")
        start = max(1, from_line or 1)
        count = max(1, lines if lines is not None else len(file_lines))
        sliced = file_lines[start - 1 : start - 1 + count]
        return {"text": "\n".join(sliced), "path": rel}

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(
        self,
        *,
        reason: str | None = None,
        force: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """
        Sync memory files into the index.
        Single-writer: concurrent calls block until the first finishes.
        Mirrors: manager.ts::sync
        """
        if self._closed:
            return

        with self._sync_lock:
            if self._closed:
                return
            needs_full_reindex = force or self._needs_full_reindex()
            sync_memory_files(
                self._db,
                workspace_dir=self.workspace_dir,
                extra_paths=self._config.extra_paths,
                source="memory",
                needs_full_reindex=needs_full_reindex,
                model=self._model,
                provider=self._provider,
                provider_id=self._provider_id,
                provider_key=self._provider_key,
                fts_table=_FTS_TABLE,
                fts_available=self._fts_available,
                chunk_tokens=self._config.chunking.tokens,
                chunk_overlap=self._config.chunking.overlap,
                cache_enabled=self._config.cache.enabled,
                progress_callback=progress,
            )
            self._dirty = False
            self._write_meta("lastSync", str(int(time.time() * 1000)))

    def _needs_full_reindex(self) -> bool:
        """Check whether the stored provider/model matches current config."""
        row = self._db.execute("SELECT value FROM meta WHERE key = 'providerModel'").fetchone()
        stored = row[0] if row else None
        current = f"{self._provider_id}:{self._model}"
        if stored != current:
            self._write_meta("providerModel", current)
            return True
        return False

    def _write_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> MemoryProviderStatus:
        """Return current index status. Mirrors: manager.ts::status"""
        sf_result: tuple[str, list[Any]] = ("", [])
        if sf_filter := self._build_source_filter():
            sf_result = sf_filter
        sf_sql, sf_params = sf_result

        files_row = self._db.execute(
            f"SELECT COUNT(*) FROM files WHERE 1=1{sf_sql}", sf_params
        ).fetchone()
        chunks_row = self._db.execute(
            f"SELECT COUNT(*) FROM chunks WHERE 1=1{sf_sql}", sf_params
        ).fetchone()

        files_count = files_row[0] if files_row else 0
        chunks_count = chunks_row[0] if chunks_row else 0

        # Cache entries
        cache_entries: int | None = None
        if self._config.cache.enabled:
            cr = self._db.execute(f"SELECT COUNT(*) FROM {_EMBEDDING_CACHE_TABLE}").fetchone()
            cache_entries = cr[0] if cr else 0

        search_mode = "hybrid" if self._provider else "fts-only"

        return MemoryProviderStatus(
            backend="builtin",
            provider=self._provider_id,
            model=self._model if self._provider else None,
            requested_provider=self._provider_meta.get("requested_provider", self._provider_id),
            files=files_count,
            chunks=chunks_count,
            dirty=self._dirty,
            workspace_dir=self.workspace_dir,
            db_path=self.db_path,
            extra_paths=self._config.extra_paths,
            sources=list(self._config.sources),
            fts={
                "enabled": self._fts_enabled,
                "available": self._fts_available,
                "error": self._fts_error,
            },
            cache={
                "enabled": self._config.cache.enabled,
                "entries": cache_entries,
                "max_entries": self._config.cache.max_entries,
            },
            fallback={
                "from": self._provider_meta.get("fallback_from"),
                "reason": self._provider_meta.get("fallback_reason"),
            }
            if self._provider_meta.get("fallback_from")
            else None,
            custom={"search_mode": search_mode},
        )

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def probe_embedding_availability(self) -> MemoryEmbeddingProbeResult:
        if self._provider is None:
            reason = (
                self._provider_meta.get("unavailable_reason")
                or "No embedding provider (FTS-only mode)"
            )
            return MemoryEmbeddingProbeResult(ok=False, error=reason)
        try:
            self._provider.embed_query("ping")
            return MemoryEmbeddingProbeResult(ok=True)
        except Exception as exc:
            return MemoryEmbeddingProbeResult(ok=False, error=str(exc))

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._db.close()
        except Exception:
            pass

    def __enter__(self) -> MemoryIndexManager:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
