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
_ALL_TABLES = ("episodic_memories", "semantic_memories", "canonical_memories")


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


def _path_from_metadata(
    metadata: Any,
    *,
    default_path: str,
) -> str:
    """Use metadata.source_path when available; otherwise keep legacy path format."""
    if isinstance(metadata, dict):
        source_path = metadata.get("source_path")
        if isinstance(source_path, str) and source_path.strip():
            return source_path.strip().replace("\\", "/")
    return default_path


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
            1.0 - (embedding <=> %s::vector) AS cosine_similarity,
            metadata
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
        row_id, content, created_at, memory_type, cosine_sim, metadata = (
            row if len(row) >= 6 else (*row, {})
        )
        default_path = f"{path_prefix}/{row_id}"
        results.append(
            {
                "id": str(row_id),
                "path": _path_from_metadata(metadata, default_path=default_path),
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
                ts_rank(to_tsvector('english', {text_col}), plainto_tsquery('english', %s)) AS rank,
                metadata
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
                ts_rank(tsv, plainto_tsquery('english', %s)) AS rank,
                metadata
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
        row_id, content, created_at, memory_type, rank, metadata = (
            row if len(row) >= 6 else (*row, {})
        )
        default_path = f"{path_prefix}/{row_id}"
        results.append(
            {
                "id": str(row_id),
                "path": _path_from_metadata(metadata, default_path=default_path),
                "start_line": 0,
                "end_line": 0,
                "source": memory_type or source,
                "snippet": content or "",
                "text_score": float(rank) if rank is not None else 0.0,
                "created_at": created_at,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Tier 1: Compact search (index-only)
# ---------------------------------------------------------------------------


def pg_search_compact(
    conn: psycopg.Connection[Any],
    user_id: str,
    query_vec: list[float],
    *,
    limit: int = 20,
    tables: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Tier 1: Return lightweight index entries across memory tables.

    Returns only id, title (first 80 chars), memory_type, score, created_at,
    and source table — enough to decide which memories to drill into.
    """
    target_tables = tables or list(_ALL_TABLES)
    for t in target_tables:
        _validate_table(t)

    unions: list[str] = []
    params: list[Any] = []

    for tbl in target_tables:
        text_col = _table_text_column(tbl)
        extra_where = _table_extra_where(tbl)
        source_label = tbl.replace("_memories", "")  # episodic/semantic/canonical

        unions.append(f"""
            SELECT id::text, LEFT({text_col}, 80) AS title, memory_type,
                   1.0 - (embedding <=> %s::vector) AS score,
                   created_at, '{source_label}' AS source
            FROM {tbl}
            WHERE user_id = %s{extra_where}
              AND embedding IS NOT NULL
        """)
        params.extend([query_vec, user_id])

    sql = " UNION ALL ".join(unions) + " ORDER BY score DESC LIMIT %s"
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "id": str(row[0]),
            "title": row[1] or "",
            "memory_type": row[2] or "",
            "score": float(row[3]) if row[3] is not None else 0.0,
            "created_at": row[4],
            "source": row[5],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Tier 2: Timeline context
# ---------------------------------------------------------------------------


def pg_get_timeline(
    conn: psycopg.Connection[Any],
    user_id: str,
    memory_id: str,
    *,
    depth_before: int = 3,
    depth_after: int = 3,
) -> list[dict[str, Any]]:
    """
    Tier 2: Get memories adjacent in time to a given memory_id.

    Searches across all three tables to find the anchor memory, then
    returns ``depth_before`` memories before and ``depth_after`` after,
    ordered by created_at.
    """
    # Step 1: Find the anchor memory and its created_at
    anchor_row = None
    anchor_source = None
    for tbl in _ALL_TABLES:
        text_col = _table_text_column(tbl)
        sql = f"SELECT id::text, {text_col}, memory_type, created_at FROM {tbl} WHERE id::text = %s"
        with conn.cursor() as cur:
            cur.execute(sql, (memory_id,))
            row = cur.fetchone()
        if row is not None:
            anchor_row = row
            anchor_source = tbl.replace("_memories", "")
            break

    if anchor_row is None:
        return []

    anchor_id, anchor_content, anchor_type, anchor_ts = anchor_row

    # Step 2: Get neighbors before and after
    neighbors: list[dict[str, Any]] = []

    for tbl in _ALL_TABLES:
        text_col = _table_text_column(tbl)
        extra_where = _table_extra_where(tbl)
        source_label = tbl.replace("_memories", "")

        # Before
        sql_before = f"""
            SELECT id::text, LEFT({text_col}, 500) AS content, memory_type,
                   created_at, '{source_label}' AS source
            FROM {tbl}
            WHERE user_id = %s{extra_where}
              AND created_at < %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql_before, (user_id, anchor_ts, depth_before))
            for row in cur.fetchall():
                neighbors.append({
                    "id": str(row[0]),
                    "content": row[1] or "",
                    "memory_type": row[2] or "",
                    "created_at": row[3],
                    "source": row[4],
                })

        # After
        sql_after = f"""
            SELECT id::text, LEFT({text_col}, 500) AS content, memory_type,
                   created_at, '{source_label}' AS source
            FROM {tbl}
            WHERE user_id = %s{extra_where}
              AND created_at > %s
            ORDER BY created_at ASC
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql_after, (user_id, anchor_ts, depth_after))
            for row in cur.fetchall():
                neighbors.append({
                    "id": str(row[0]),
                    "content": row[1] or "",
                    "memory_type": row[2] or "",
                    "created_at": row[3],
                    "source": row[4],
                })

    # Sort all neighbors by created_at, dedup, trim
    neighbors.sort(key=lambda x: x["created_at"] if x["created_at"] else "")
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for n in neighbors:
        if n["id"] not in seen:
            seen.add(n["id"])
            unique.append(n)

    # Split into before/after relative to anchor
    before = [n for n in unique if n["created_at"] and n["created_at"] < anchor_ts][-depth_before:]
    after = [n for n in unique if n["created_at"] and n["created_at"] > anchor_ts][:depth_after]

    # Build result: anchor + neighbor summaries
    anchor_result = {
        "id": str(anchor_id),
        "content": (anchor_content or "")[:500],
        "memory_type": anchor_type or "",
        "score": 1.0,
        "created_at": anchor_ts,
        "source": anchor_source,
        "neighbors": [
            {"id": n["id"], "title": (n["content"] or "")[:80], "created_at": n["created_at"]}
            for n in before + after
        ],
    }

    return [anchor_result]


# ---------------------------------------------------------------------------
# Tier 3: Full content by IDs
# ---------------------------------------------------------------------------


def pg_get_memories_by_ids(
    conn: psycopg.Connection[Any],
    memory_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Tier 3: Fetch full content for specific memory IDs across all tables.
    """
    if not memory_ids:
        return []

    results: list[dict[str, Any]] = []

    for tbl in _ALL_TABLES:
        text_col = _table_text_column(tbl)
        extra_where = _table_extra_where(tbl)
        path_prefix = _table_path_prefix(tbl)
        source_label = tbl.replace("_memories", "")

        placeholders = ",".join(["%s"] * len(memory_ids))
        sql = f"""
            SELECT id::text, {text_col}, created_at, memory_type, metadata
            FROM {tbl}
            WHERE id::text IN ({placeholders}){extra_where}
        """
        with conn.cursor() as cur:
            cur.execute(sql, memory_ids)
            for row in cur.fetchall():
                row_id, content, created_at, memory_type, metadata = (
                    row if len(row) >= 5 else (*row, {})
                )
                default_path = f"{path_prefix}/{row_id}"
                results.append({
                    "id": str(row_id),
                    "path": _path_from_metadata(metadata, default_path=default_path),
                    "start_line": 0,
                    "end_line": 0,
                    "source": memory_type or source_label,
                    "snippet": content or "",
                    "score": 1.0,
                    "created_at": created_at,
                })

    return results
