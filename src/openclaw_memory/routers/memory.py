"""
Memory CRUD endpoints (mem0-style).

  POST   /memory/add          — add memories from conversation
  GET    /memory/{user_id}    — list all memories for user
  DELETE /memory/{user_id}/{id} — delete a memory
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
import psycopg

from ..core.service import MemoryService
from ..dependencies import get_db, get_memory_service
from ..models.memory import (
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryDeleteResponse,
    MemoryItem,
    MemoryListResponse,
)

router = APIRouter()


@router.post("/add", response_model=MemoryAddResponse)
def add_memory(
    req: MemoryAddRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> MemoryAddResponse:
    """Add memories from a conversation or directly."""
    conn.autocommit = False
    try:
        if req.conversation:
            result = service.ingest_conversation(
                conn, req.user_id, req.conversation,
                session_id=req.session_id,
            )
            conn.commit()
            return MemoryAddResponse(
                stored=result.get("stored", 0),
                extracted=result.get("extracted", 0),
            )
        elif req.content:
            memory_id = service.add_memory(
                conn, req.user_id, req.content,
                memory_type=req.memory_type or "fact",
                metadata=req.metadata,
            )
            conn.commit()
            return MemoryAddResponse(stored=1, extracted=0, memory_id=memory_id)
        else:
            raise HTTPException(400, "Either 'conversation' or 'content' is required")
    except Exception:
        conn.rollback()
        raise


@router.get("/{user_id}", response_model=MemoryListResponse)
def list_memories(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> MemoryListResponse:
    """List all memories for a user."""
    rows = service.get_user_memories(conn, user_id, limit=limit, offset=offset)
    items = [
        MemoryItem(
            id=r["id"],
            content=r["content"],
            memory_type=r.get("memory_type", ""),
            source=r.get("source", ""),
            created_at=str(r.get("created_at", "")),
            metadata=r.get("metadata", {}),
        )
        for r in rows
    ]
    return MemoryListResponse(user_id=user_id, memories=items, total=len(items))


@router.delete("/{user_id}/{memory_id}", response_model=MemoryDeleteResponse)
def delete_memory(
    user_id: str,
    memory_id: str,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> MemoryDeleteResponse:
    """Delete a specific memory."""
    deleted = service.delete_memory(conn, memory_id, user_id)
    if not deleted:
        raise HTTPException(404, "Memory not found")
    conn.commit()
    return MemoryDeleteResponse(deleted=True, memory_id=memory_id)
