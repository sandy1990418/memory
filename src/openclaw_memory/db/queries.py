"""
PostgreSQL query functions for the multi-layer memory system.

Provides vector (pgvector cosine), keyword (tsvector), compact search,
timeline context, and full-content retrieval — all scoped by user_id.
"""

from __future__ import annotations

from typing import Any

import psycopg

# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

_VALID_TABLES = frozenset({
    "episodic_memories", "semantic_memories", "canonical_memories",
})
_ALL_TABLES = ("episodic_memories", "semantic_memories", "canonical_memories")


def _validate_table(table: str) -> str:
    if table not in _VALID_TABLES:
        raise ValueError(f"Unknown table {table!r}")
    return table


def _text_col(table: str) -> str:
    return "value" if table == "canonical_memories" else "content"


def _extra_where(table: str) -> str:
    return " AND status = 'active'" if table == "canonical_memories" else ""


def _source_label(table: str) -> str:
    return table.replace("_memories", "")


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


def search_vector(
    conn: psycopg.Connection[Any],
    user_id: str,
    query_vec: list[float],
    *,
    limit: int = 20,
    table: str = "episodic_memories",
) -> list[dict[str, Any]]:
    """pgvector cosine similarity search. Returns dicts with vector_score."""
    tbl = _validate_table(table)
    col = _text_col(tbl)
    extra = _extra_where(tbl)
    source = _source_label(tbl)

    sql = f"""
        SELECT id::text, {col}, created_at, memory_type,
               1.0 - (embedding <=> %s::vector) AS cosine_sim, metadata
        FROM {tbl}
        WHERE user_id = %s{extra} AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (query_vec, user_id, query_vec, limit))
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]),
            "snippet": r[1] or "",
            "created_at": r[2],
            "memory_type": r[3] or "",
            "vector_score": float(r[4]) if r[4] is not None else 0.0,
            "source": r[3] or source,
            "metadata": r[5] if isinstance(r[5], dict) else {},
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def search_keyword(
    conn: psycopg.Connection[Any],
    user_id: str,
    query: str,
    *,
    limit: int = 20,
    table: str = "episodic_memories",
) -> list[dict[str, Any]]:
    """Full-text search using PostgreSQL tsvector. Returns dicts with text_score."""
    tbl = _validate_table(table)
    col = _text_col(tbl)
    extra = _extra_where(tbl)
    source = _source_label(tbl)

    if tbl == "canonical_memories":
        tsv_expr = f"to_tsvector('english', {col})"
    else:
        tsv_expr = "tsv"

    sql = f"""
        SELECT id::text, {col}, created_at, memory_type,
               ts_rank({tsv_expr}, plainto_tsquery('english', %s)) AS rank,
               metadata
        FROM {tbl}
        WHERE user_id = %s{extra}
          AND {tsv_expr} @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (query, user_id, query, limit))
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]),
            "snippet": r[1] or "",
            "created_at": r[2],
            "memory_type": r[3] or "",
            "text_score": float(r[4]) if r[4] is not None else 0.0,
            "source": r[3] or source,
            "metadata": r[5] if isinstance(r[5], dict) else {},
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Compact search (Tier 1 — progressive disclosure)
# ---------------------------------------------------------------------------


def search_compact(
    conn: psycopg.Connection[Any],
    user_id: str,
    query_vec: list[float],
    *,
    limit: int = 20,
    tables: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Tier 1: Lightweight index entries across memory tables (~50 tokens each)."""
    target_tables = tables or list(_ALL_TABLES)
    for t in target_tables:
        _validate_table(t)

    unions: list[str] = []
    params: list[Any] = []

    for tbl in target_tables:
        col = _text_col(tbl)
        extra = _extra_where(tbl)
        src = _source_label(tbl)
        unions.append(f"""
            SELECT id::text, LEFT({col}, 80) AS title, memory_type,
                   1.0 - (embedding <=> %s::vector) AS score,
                   created_at, '{src}' AS source
            FROM {tbl}
            WHERE user_id = %s{extra} AND embedding IS NOT NULL
        """)
        params.extend([query_vec, user_id])

    sql = " UNION ALL ".join(unions) + " ORDER BY score DESC LIMIT %s"
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]),
            "title": r[1] or "",
            "memory_type": r[2] or "",
            "score": float(r[3]) if r[3] is not None else 0.0,
            "created_at": r[4],
            "source": r[5],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Timeline (Tier 2)
# ---------------------------------------------------------------------------


def get_timeline(
    conn: psycopg.Connection[Any],
    user_id: str,
    memory_id: str,
    *,
    depth_before: int = 3,
    depth_after: int = 3,
) -> dict[str, Any] | None:
    """Tier 2: Get temporal context around a specific memory."""
    # Find the anchor memory
    anchor_row = None
    anchor_source = None
    for tbl in _ALL_TABLES:
        col = _text_col(tbl)
        sql = f"SELECT id::text, {col}, memory_type, created_at FROM {tbl} WHERE id::text = %s"
        with conn.cursor() as cur:
            cur.execute(sql, (memory_id,))
            row = cur.fetchone()
        if row is not None:
            anchor_row = row
            anchor_source = _source_label(tbl)
            break

    if anchor_row is None:
        return None

    anchor_id, anchor_content, anchor_type, anchor_ts = anchor_row

    # Get neighbors
    neighbors: list[dict[str, Any]] = []
    for tbl in _ALL_TABLES:
        col = _text_col(tbl)
        extra = _extra_where(tbl)
        src = _source_label(tbl)

        for direction, op, order, depth in [
            ("before", "<", "DESC", depth_before),
            ("after", ">", "ASC", depth_after),
        ]:
            sql = f"""
                SELECT id::text, LEFT({col}, 500), memory_type, created_at, '{src}'
                FROM {tbl}
                WHERE user_id = %s{extra} AND created_at {op} %s
                ORDER BY created_at {order}
                LIMIT %s
            """
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, anchor_ts, depth))
                for r in cur.fetchall():
                    neighbors.append({
                        "id": str(r[0]),
                        "content": r[1] or "",
                        "memory_type": r[2] or "",
                        "created_at": r[3],
                        "source": r[4],
                    })

    # Deduplicate and sort
    seen: set[str] = set()
    unique = []
    for n in sorted(neighbors, key=lambda x: x["created_at"] or ""):
        if n["id"] not in seen:
            seen.add(n["id"])
            unique.append(n)

    return {
        "id": str(anchor_id),
        "content": (anchor_content or "")[:500],
        "memory_type": anchor_type or "",
        "score": 1.0,
        "created_at": anchor_ts,
        "source": anchor_source,
        "neighbors": [
            {"id": n["id"], "title": (n["content"] or "")[:80], "created_at": n["created_at"]}
            for n in unique
        ],
    }


# ---------------------------------------------------------------------------
# Full content by IDs (Tier 3)
# ---------------------------------------------------------------------------


def get_memories_by_ids(
    conn: psycopg.Connection[Any],
    memory_ids: list[str],
) -> list[dict[str, Any]]:
    """Tier 3: Fetch full content for specific memory IDs across all tables."""
    if not memory_ids:
        return []

    results: list[dict[str, Any]] = []
    for tbl in _ALL_TABLES:
        col = _text_col(tbl)
        extra = _extra_where(tbl)
        src = _source_label(tbl)
        placeholders = ",".join(["%s"] * len(memory_ids))

        sql = f"""
            SELECT id::text, {col}, created_at, memory_type, metadata
            FROM {tbl}
            WHERE id::text IN ({placeholders}){extra}
        """
        with conn.cursor() as cur:
            cur.execute(sql, memory_ids)
            for r in cur.fetchall():
                results.append({
                    "id": str(r[0]),
                    "content": r[1] or "",
                    "created_at": r[2],
                    "memory_type": r[3] or "",
                    "source": src,
                    "metadata": r[4] if isinstance(r[4], dict) else {},
                })

    return results


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


def insert_memory(
    conn: psycopg.Connection[Any],
    *,
    user_id: str,
    content: str,
    embedding_literal: str,
    memory_type: str,
    metadata_json: str,
    table: str = "episodic_memories",
    session_id: str | None = None,
) -> str:
    """Insert a memory row and return its UUID."""
    tbl = _validate_table(table)
    if tbl == "episodic_memories" and session_id:
        sql = f"""
            INSERT INTO {tbl} (user_id, session_id, content, embedding, memory_type, metadata)
            VALUES (%s, %s, %s, %s::vector, %s, %s)
            RETURNING id::text
        """
        params = (user_id, session_id, content, embedding_literal, memory_type, metadata_json)
    else:
        sql = f"""
            INSERT INTO {tbl} (user_id, content, embedding, memory_type, metadata)
            VALUES (%s, %s, %s::vector, %s, %s)
            RETURNING id::text
        """
        params = (user_id, content, embedding_literal, memory_type, metadata_json)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return str(row[0]) if row else ""


def delete_memory(
    conn: psycopg.Connection[Any],
    memory_id: str,
    user_id: str,
) -> bool:
    """Soft-delete a memory (canonical) or hard-delete (episodic/semantic)."""
    # Try canonical first (soft-delete)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE canonical_memories SET status = 'deleted', updated_at = now() "
            "WHERE id::text = %s AND user_id = %s AND status = 'active'",
            (memory_id, user_id),
        )
        if cur.rowcount and cur.rowcount > 0:
            return True

    # Hard-delete from episodic/semantic
    for tbl in ("episodic_memories", "semantic_memories"):
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {tbl} WHERE id::text = %s AND user_id = %s",
                (memory_id, user_id),
            )
            if cur.rowcount and cur.rowcount > 0:
                return True

    return False


def get_user_memories(
    conn: psycopg.Connection[Any],
    user_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List all memories for a user across all tables."""
    results: list[dict[str, Any]] = []
    for tbl in _ALL_TABLES:
        col = _text_col(tbl)
        extra = _extra_where(tbl)
        src = _source_label(tbl)
        sql = f"""
            SELECT id::text, {col}, created_at, memory_type, metadata
            FROM {tbl}
            WHERE user_id = %s{extra}
            ORDER BY created_at DESC
        """
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            for r in cur.fetchall():
                results.append({
                    "id": str(r[0]),
                    "content": r[1] or "",
                    "created_at": r[2],
                    "memory_type": r[3] or "",
                    "source": src,
                    "metadata": r[4] if isinstance(r[4], dict) else {},
                })

    # Sort all by created_at desc, then paginate
    results.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return results[offset: offset + limit]


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def check_token_budget(
    conn: psycopg.Connection[Any],
    user_id: str,
) -> tuple[int, int]:
    """
    Return (remaining_tokens, daily_limit) for a user.

    Auto-resets ``tokens_used_today`` when the calendar day changes.
    Creates a profile row if the user doesn't have one yet.
    Uses SELECT ... FOR UPDATE to prevent race conditions with
    concurrent record_token_usage calls.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_profiles (user_id)
            VALUES (%s)
            ON CONFLICT (user_id) DO NOTHING
            """,
            (user_id,),
        )
        cur.execute(
            """
            UPDATE user_profiles
            SET tokens_used_today = 0,
                budget_reset_at = now()
            WHERE user_id = %s
              AND budget_reset_at::date < now()::date
            """,
            (user_id,),
        )
        cur.execute(
            "SELECT token_budget_daily, tokens_used_today "
            "FROM user_profiles WHERE user_id = %s FOR UPDATE",
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        return (100000, 100000)
    budget, used = int(row[0]), int(row[1])
    return (max(0, budget - used), budget)


def record_token_usage(
    conn: psycopg.Connection[Any],
    user_id: str,
    tokens: int,
) -> None:
    """Atomically increment tokens_used_today and return new remaining budget."""
    if tokens <= 0:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_profiles (user_id, tokens_used_today)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE
            SET tokens_used_today = user_profiles.tokens_used_today + EXCLUDED.tokens_used_today,
                updated_at = now()
            """,
            (user_id, tokens),
        )


def get_token_usage(
    conn: psycopg.Connection[Any],
    user_id: str,
) -> dict[str, Any]:
    """Return token usage stats for a user."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT token_budget_daily, tokens_used_today, budget_reset_at, updated_at
            FROM user_profiles WHERE user_id = %s
            """,
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        return {"user_id": user_id, "budget": 100000, "used": 0, "remaining": 100000}
    budget, used = int(row[0]), int(row[1])
    return {
        "user_id": user_id,
        "budget": budget,
        "used": used,
        "remaining": max(0, budget - used),
        "reset_at": row[2].isoformat() if row[2] else None,
        "updated_at": row[3].isoformat() if row[3] else None,
    }


def set_token_budget(
    conn: psycopg.Connection[Any],
    user_id: str,
    daily_limit: int,
) -> None:
    """Set the daily token budget for a user."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_profiles (user_id, token_budget_daily)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE
            SET token_budget_daily = EXCLUDED.token_budget_daily,
                updated_at = now()
            """,
            (user_id, daily_limit),
        )


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


def verify_api_key(
    conn: psycopg.Connection[Any],
    key_hash: str,
) -> str | None:
    """Return user_id if key_hash is valid and not revoked, else None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM api_keys WHERE key_hash = %s AND revoked_at IS NULL",
            (key_hash,),
        )
        row = cur.fetchone()
    return str(row[0]) if row else None


def create_api_key(
    conn: psycopg.Connection[Any],
    key_hash: str,
    user_id: str,
    label: str = "",
) -> None:
    """Insert a new API key."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (key_hash, user_id, label) VALUES (%s, %s, %s)",
            (key_hash, user_id, label),
        )
