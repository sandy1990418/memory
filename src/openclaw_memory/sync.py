"""
Memory file sync: walk files, chunk, embed, upsert into SQLite.
Mirrors: src/memory/sync-memory-files.ts, sync-index.ts, sync-stale.ts
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from .internal import (
    MemoryFileEntry,
    build_file_entry,
    chunk_markdown,
    hash_chunk_id,
    list_memory_files,
    normalize_extra_memory_paths,
)

# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------


def _upsert_file(db: sqlite3.Connection, entry: MemoryFileEntry, source: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO files (path, source, hash, mtime, size) VALUES (?, ?, ?, ?, ?)",
        (entry.path, source, entry.hash, int(entry.mtime_ms), entry.size),
    )


def _delete_chunks_for_path(
    db: sqlite3.Connection,
    path: str,
    source: str,
    *,
    fts_table: str,
    fts_available: bool,
    model: str,
) -> None:
    db.execute(
        "DELETE FROM chunks WHERE path = ? AND source = ? AND model = ?",
        (path, source, model),
    )
    if fts_available:
        try:
            db.execute(
                f"DELETE FROM {fts_table} WHERE path = ? AND source = ? AND model = ?",
                (path, source, model),
            )
        except sqlite3.OperationalError:
            pass


def _upsert_chunk(
    db: sqlite3.Connection,
    *,
    chunk_id: str,
    path: str,
    source: str,
    start_line: int,
    end_line: int,
    chunk_hash: str,
    model: str,
    text: str,
    embedding_json: str,
    updated_at: int,
    fts_table: str,
    fts_available: bool,
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO chunks "
        "(id, path, source, start_line, end_line, hash, model, text, embedding, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            path,
            source,
            start_line,
            end_line,
            chunk_hash,
            model,
            text,
            embedding_json,
            updated_at,
        ),
    )
    if fts_available:
        try:
            db.execute(
                f"INSERT OR REPLACE INTO {fts_table} "
                f"(text, id, path, source, model, start_line, end_line) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                (text, chunk_id, path, source, model, start_line, end_line),
            )
        except sqlite3.OperationalError:
            pass


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

_EMBEDDING_CACHE_TABLE = "embedding_cache"


def _get_cached_embedding(
    db: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    provider_key: str,
    text_hash: str,
) -> list[float] | None:
    row = db.execute(
        f"SELECT embedding FROM {_EMBEDDING_CACHE_TABLE} "
        f"WHERE provider = ? AND model = ? AND provider_key = ? AND hash = ?",
        (provider, model, provider_key, text_hash),
    ).fetchone()
    if row:
        from .internal import parse_embedding

        return parse_embedding(row[0])
    return None


def _put_cached_embedding(
    db: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    provider_key: str,
    text_hash: str,
    embedding: list[float],
) -> None:
    emb_json = json.dumps(embedding)
    now = int(time.time() * 1000)
    db.execute(
        f"INSERT OR REPLACE INTO {_EMBEDDING_CACHE_TABLE} "
        f"(provider, model, provider_key, hash, embedding, dims, updated_at) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?)",
        (provider, model, provider_key, text_hash, emb_json, len(embedding), now),
    )


# ---------------------------------------------------------------------------
# File indexing
# ---------------------------------------------------------------------------


def index_file_entry_if_changed(
    db: sqlite3.Connection,
    entry: MemoryFileEntry,
    source: str,
    *,
    needs_full_reindex: bool,
    model: str,
    provider: Any | None,
    provider_id: str,
    provider_key: str = "",
    fts_table: str,
    fts_available: bool,
    chunk_tokens: int = 400,
    chunk_overlap: int = 80,
    cache_enabled: bool = True,
) -> bool:
    """
    Check if file has changed and re-index it if so.
    Returns True if indexing was performed.
    Mirrors: sync-index.ts::indexFileEntryIfChanged
    """
    if not needs_full_reindex:
        row = db.execute(
            "SELECT hash FROM files WHERE path = ? AND source = ?",
            (entry.path, source),
        ).fetchone()
        if row and row[0] == entry.hash:
            return False  # unchanged

    # Read content and chunk
    from pathlib import Path

    content = Path(entry.abs_path).read_text(encoding="utf-8")
    chunks = chunk_markdown(content, (chunk_tokens, chunk_overlap))

    # Delete old chunks
    _delete_chunks_for_path(
        db,
        entry.path,
        source,
        fts_table=fts_table,
        fts_available=fts_available,
        model=model,
    )

    now = int(time.time() * 1000)

    # Embed (batch where possible, with cache)
    embeddings: list[list[float] | None] = [None] * len(chunks)
    if provider is not None:
        to_embed_texts: list[str] = []
        to_embed_indices: list[int] = []
        # 1) Try cache for each chunk
        for i, chunk in enumerate(chunks):
            cached: list[float] | None = None
            if cache_enabled:
                cached = _get_cached_embedding(
                    db,
                    provider=provider_id,
                    model=model,
                    provider_key=provider_key,
                    text_hash=chunk.hash,
                )
            if cached is not None:
                embeddings[i] = cached
            else:
                to_embed_texts.append(chunk.text)
                to_embed_indices.append(i)

        # 2) Batch-embed missing chunks
        if to_embed_texts:
            batch_size = 32
            for start in range(0, len(to_embed_texts), batch_size):
                batch_texts = to_embed_texts[start : start + batch_size]
                batch_indices = to_embed_indices[start : start + batch_size]
                try:
                    batch_embeddings = provider.embed_batch(batch_texts)
                except Exception:
                    # Fallback to per-item embedding if batch fails
                    batch_embeddings = []
                    for text in batch_texts:
                        try:
                            batch_embeddings.append(provider.embed_query(text))
                        except Exception:
                            batch_embeddings.append([])

                # Guard against provider returning unexpected length
                if len(batch_embeddings) != len(batch_texts):
                    # Fallback to per-item embedding to realign
                    batch_embeddings = []
                    for text in batch_texts:
                        try:
                            batch_embeddings.append(provider.embed_query(text))
                        except Exception:
                            batch_embeddings.append([])

                for idx, emb in zip(batch_indices, batch_embeddings):
                    embeddings[idx] = emb
                    if cache_enabled and emb:
                        chunk = chunks[idx]
                        _put_cached_embedding(
                            db,
                            provider=provider_id,
                            model=model,
                            provider_key=provider_key,
                            text_hash=chunk.hash,
                            embedding=emb,
                        )

    # 3) Upsert chunks
    for i, chunk in enumerate(chunks):
        chunk_id = hash_chunk_id(entry.path, chunk.start_line, chunk.end_line, chunk.hash)
        embedding = embeddings[i] if embeddings[i] is not None else []
        embedding_json = json.dumps(embedding)
        _upsert_chunk(
            db,
            chunk_id=chunk_id,
            path=entry.path,
            source=source,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_hash=chunk.hash,
            model=model,
            text=chunk.text,
            embedding_json=embedding_json,
            updated_at=now,
            fts_table=fts_table,
            fts_available=fts_available,
        )

    _upsert_file(db, entry, source)
    return True


# ---------------------------------------------------------------------------
# Stale path cleanup
# ---------------------------------------------------------------------------


def delete_stale_indexed_paths(
    db: sqlite3.Connection,
    *,
    source: str,
    active_paths: set[str],
    fts_table: str,
    fts_available: bool,
    model: str,
) -> None:
    """Remove chunks/files for paths that no longer exist on disk."""
    rows = db.execute("SELECT path FROM files WHERE source = ?", (source,)).fetchall()
    for (path,) in rows:
        if path not in active_paths:
            _delete_chunks_for_path(
                db,
                path,
                source,
                fts_table=fts_table,
                fts_available=fts_available,
                model=model,
            )
            db.execute(
                "DELETE FROM files WHERE path = ? AND source = ?",
                (path, source),
            )


# ---------------------------------------------------------------------------
# Top-level sync
# ---------------------------------------------------------------------------


def sync_memory_files(
    db: sqlite3.Connection,
    *,
    workspace_dir: str,
    extra_paths: list[str] | None,
    source: str = "memory",
    needs_full_reindex: bool,
    model: str,
    provider: Any | None,
    provider_id: str,
    provider_key: str = "",
    fts_table: str,
    fts_available: bool,
    chunk_tokens: int = 400,
    chunk_overlap: int = 80,
    cache_enabled: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """
    Walk all memory files, index changed ones, remove stale ones.
    Mirrors: src/memory/sync-memory-files.ts::syncMemoryFiles
    """
    resolved_extra = normalize_extra_memory_paths(workspace_dir, extra_paths)
    files = list_memory_files(workspace_dir, resolved_extra)

    file_entries = [build_file_entry(f, workspace_dir) for f in files]
    active_paths = {e.path for e in file_entries}
    total = len(file_entries)

    for i, entry in enumerate(file_entries):
        if progress_callback:
            progress_callback(i, total, f"Indexing {entry.path}")
        try:
            index_file_entry_if_changed(
                db,
                entry,
                source,
                needs_full_reindex=needs_full_reindex,
                model=model,
                provider=provider,
                provider_id=provider_id,
                provider_key=provider_key,
                fts_table=fts_table,
                fts_available=fts_available,
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
                cache_enabled=cache_enabled,
            )
        except Exception:
            pass  # don't let one bad file abort the whole sync

    db.commit()

    delete_stale_indexed_paths(
        db,
        source=source,
        active_paths=active_paths,
        fts_table=fts_table,
        fts_available=fts_available,
        model=model,
    )
    db.commit()

    if progress_callback:
        progress_callback(total, total, "Done")
