"""
Low-level SQLite search helpers for vector and keyword search.
Mirrors: src/memory/manager-search.ts
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .internal import cosine_similarity, parse_embedding, truncate_utf16_safe


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


def search_vector(
    db: sqlite3.Connection,
    *,
    provider_model: str,
    query_vec: list[float],
    limit: int,
    snippet_max_chars: int = 700,
    source_filter: tuple[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Search for the closest chunks by cosine similarity.
    Falls back to in-process cosine similarity (sqlite-vec not used in MVP).

    source_filter: optional (sql_fragment, params) e.g. (" AND source = ?", ["memory"])
    Mirrors: manager-search.ts::searchVector (fallback path)
    """
    if not query_vec or limit <= 0:
        return []

    sf_sql, sf_params = source_filter or ("", [])
    rows = db.execute(
        f"SELECT id, path, start_line, end_line, text, source, embedding "
        f"FROM chunks WHERE model = ?{sf_sql}",
        [provider_model, *sf_params],
    ).fetchall()

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        chunk_id, path, start_line, end_line, text, source, emb_json = row
        emb = parse_embedding(emb_json)
        score = cosine_similarity(query_vec, emb)
        if score != score:  # NaN guard
            continue
        scored.append((score, {
            "id": chunk_id,
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "score": score,
            "snippet": truncate_utf16_safe(text, snippet_max_chars),
            "source": source,
            "vector_score": score,
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


# ---------------------------------------------------------------------------
# Keyword (FTS5) search
# ---------------------------------------------------------------------------


def search_keyword(
    db: sqlite3.Connection,
    *,
    fts_table: str,
    provider_model: str | None,
    query: str,
    limit: int,
    snippet_max_chars: int = 700,
    source_filter: tuple[str, list[str]] | None = None,
    build_fts_query_fn: Any = None,
    bm25_rank_to_score_fn: Any = None,
) -> list[dict[str, Any]]:
    """
    Full-text search using SQLite FTS5.
    Mirrors: manager-search.ts::searchKeyword
    """
    if limit <= 0:
        return []

    # Import defaults lazily to avoid circular imports
    if build_fts_query_fn is None:
        from .hybrid import build_fts_query
        build_fts_query_fn = build_fts_query
    if bm25_rank_to_score_fn is None:
        from .hybrid import bm25_rank_to_score
        bm25_rank_to_score_fn = bm25_rank_to_score

    fts_query = build_fts_query_fn(query)
    if not fts_query:
        return []

    sf_sql, sf_params = source_filter or ("", [])

    # In FTS-only mode (no provider), search all models
    model_clause = " AND model = ?" if provider_model else ""
    model_params = [provider_model] if provider_model else []

    try:
        rows = db.execute(
            f"SELECT id, path, source, start_line, end_line, text, "
            f"bm25({fts_table}) AS rank "
            f"FROM {fts_table} "
            f"WHERE {fts_table} MATCH ?{model_clause}{sf_sql} "
            f"ORDER BY rank ASC "
            f"LIMIT ?",
            [fts_query, *model_params, *sf_params, limit],
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        chunk_id, path, source, start_line, end_line, text, rank = row
        text_score = bm25_rank_to_score_fn(rank)
        results.append({
            "id": chunk_id,
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "score": text_score,
            "text_score": text_score,
            "snippet": truncate_utf16_safe(text, snippet_max_chars),
            "source": source,
        })

    return results


# ---------------------------------------------------------------------------
# Chunk listing (for cosine fallback)
# ---------------------------------------------------------------------------


def list_chunks(
    db: sqlite3.Connection,
    *,
    provider_model: str,
    source_filter: tuple[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """List all chunks for a given model (used by vector fallback)."""
    sf_sql, sf_params = source_filter or ("", [])
    rows = db.execute(
        f"SELECT id, path, start_line, end_line, text, embedding, source "
        f"FROM chunks WHERE model = ?{sf_sql}",
        [provider_model, *sf_params],
    ).fetchall()
    return [
        {
            "id": row[0],
            "path": row[1],
            "start_line": row[2],
            "end_line": row[3],
            "text": row[4],
            "embedding": parse_embedding(row[5]),
            "source": row[6],
        }
        for row in rows
    ]
