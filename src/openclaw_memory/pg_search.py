"""
PostgreSQL-native search functions for the multi-tenant memory system.

Provides vector (pgvector cosine) and keyword (tsvector GIN) search
functions that return dicts compatible with merge_hybrid_results().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import psycopg

_VALID_TABLES = frozenset({"episodic_memories", "semantic_memories", "canonical_memories"})


def _validate_table(table: str) -> str:
    if table not in _VALID_TABLES:
        raise ValueError(f"Unknown table {table!r}. Must be one of {sorted(_VALID_TABLES)}")
    return table


def _table_text_column(table: str) -> str:
    if table == "canonical_memories":
        return "value"
    return "content"


def _table_extra_where(table: str) -> str:
    if table == "canonical_memories":
        return " AND status = 'active'"
    return ""


def _table_path_prefix(table: str) -> str:
    if table == "canonical_memories":
        return "canonical"
    return "episodic"


def pg_search_vector(
    conn: psycopg.Connection[Any],
    user_id: str,
    query_vec: list[float],
    limit: int = 20,
    table: str = "episodic_memories",
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Vector similarity search using pgvector cosine distance (``<=>``).

    Returns a list of dicts compatible with the ``vector`` parameter of
    ``merge_hybrid_results()``:
        {id, path, start_line, end_line, source, snippet, vector_score}

    All rows are scoped to ``user_id`` — no cross-user data is returned.

    Args:
        conn:          Open psycopg3 connection.
        user_id:       Tenant identifier; used as ``WHERE user_id = %s``.
        query_vec:     Query embedding as a plain Python list of floats.
        limit:         Maximum number of results to return.
        table:         Target table name; must be one of the known tables.
        source_filter: Optional ``memory_type`` filter (parameterised).
    """
    try:
        import psycopg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "psycopg (psycopg3) is required for PostgreSQL support. "
            "Install it with: pip install psycopg[binary]"
        ) from exc

    tbl = _validate_table(table)
    text_col = _table_text_column(tbl)
    extra_where = _table_extra_where(tbl)
    path_prefix = _table_path_prefix(tbl)
    source = tbl.rstrip("s").replace("_memorie", "")  # "episodic" / "semantic"

    params: list[Any] = [query_vec, user_id]
    where_extra = ""
    if source_filter is not None:
        where_extra = " AND memory_type = %s"
        params.append(source_filter)
    params.append(limit)

    sql = f"""
        SELECT
            id::text,
            {text_col},
            created_at,
            memory_type,
            1.0 - (embedding <=> %s::vector) AS cosine_similarity
        FROM {tbl}
        WHERE user_id = %s{extra_where}{where_extra}
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    # The ORDER BY clause needs its own copy of the query vector.
    # Insert it just before the LIMIT param.
    params.insert(-1, query_vec)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        row_id, content, created_at, memory_type, cosine_sim = row
        results.append(
            {
                "id": str(row_id),
                "path": f"{path_prefix}/{row_id}",
                "start_line": 0,
                "end_line": 0,
                "source": memory_type or source,
                "snippet": content or "",
                "vector_score": float(cosine_sim) if cosine_sim is not None else 0.0,
                "created_at": created_at,
            }
        )
    return results


def pg_search_keyword(
    conn: psycopg.Connection[Any],
    user_id: str,
    query: str,
    limit: int = 20,
    table: str = "episodic_memories",
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Full-text keyword search using PostgreSQL tsvector + ``ts_rank()``.

    Returns a list of dicts compatible with the ``keyword`` parameter of
    ``merge_hybrid_results()``:
        {id, path, start_line, end_line, source, snippet, text_score}

    All rows are scoped to ``user_id`` — no cross-user data is returned.

    Args:
        conn:          Open psycopg3 connection.
        user_id:       Tenant identifier; used as ``WHERE user_id = %s``.
        query:         Plain-English search query (passed to plainto_tsquery).
        limit:         Maximum number of results to return.
        table:         Target table name; must be one of the known tables.
        source_filter: Optional ``memory_type`` filter (parameterised).
    """
    try:
        import psycopg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "psycopg (psycopg3) is required for PostgreSQL support. "
            "Install it with: pip install psycopg[binary]"
        ) from exc

    tbl = _validate_table(table)
    text_col = _table_text_column(tbl)
    extra_where = _table_extra_where(tbl)
    path_prefix = _table_path_prefix(tbl)
    source = tbl.rstrip("s").replace("_memorie", "")  # "episodic" / "semantic"

    params: list[Any] = [query, user_id]
    where_extra = ""
    if source_filter is not None:
        where_extra = " AND memory_type = %s"
        params.append(source_filter)
    params.append(limit)

    if tbl == "canonical_memories":
        sql = f"""
            SELECT
                id::text,
                {text_col},
                created_at,
                memory_type,
                ts_rank(to_tsvector('english', {text_col}), plainto_tsquery('english', %s)) AS rank
            FROM {tbl}
            WHERE user_id = %s{extra_where}{where_extra}
              AND to_tsvector('english', {text_col}) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """
    else:
        sql = f"""
            SELECT
                id::text,
                {text_col},
                created_at,
                memory_type,
                ts_rank(tsv, plainto_tsquery('english', %s)) AS rank
            FROM {tbl}
            WHERE user_id = %s{extra_where}{where_extra}
              AND tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """
    # plainto_tsquery is referenced twice; insert second copy before LIMIT.
    params.insert(-1, query)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        row_id, content, created_at, memory_type, rank = row
        results.append(
            {
                "id": str(row_id),
                "path": f"{path_prefix}/{row_id}",
                "start_line": 0,
                "end_line": 0,
                "source": memory_type or source,
                "snippet": content or "",
                "text_score": float(rank) if rank is not None else 0.0,
                "created_at": created_at,
            }
        )
    return results
