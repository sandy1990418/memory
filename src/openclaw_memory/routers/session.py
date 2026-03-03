"""
Session lifecycle endpoints.

  POST /session/start    — start a new session
  POST /session/message  — record a message to working memory
  POST /session/end      — end session: extract, store, clear
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
import psycopg

from ..core.service import MemoryService
from ..dependencies import get_db, get_memory_service
from ..models.session import (
    MessageRequest,
    MessageResponse,
    SessionEndRequest,
    SessionEndResponse,
    SessionStartRequest,
    SessionStartResponse,
)

router = APIRouter()


@router.post("/start", response_model=SessionStartResponse)
def start_session(
    req: SessionStartRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> SessionStartResponse:
    """Start a new conversation session."""
    service.start_session(conn, req.user_id, req.session_id)
    conn.commit()
    return SessionStartResponse(
        user_id=req.user_id,
        session_id=req.session_id,
        status="started",
    )


@router.post("/message", response_model=MessageResponse)
def record_message(
    req: MessageRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> MessageResponse:
    """Record a message to working memory."""
    service.record_message(conn, req.user_id, req.session_id, req.role, req.content)
    conn.commit()
    return MessageResponse(status="recorded")


@router.post("/end", response_model=SessionEndResponse)
def end_session(
    req: SessionEndRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> SessionEndResponse:
    """End a session: extract memories, store, and clear working memory."""
    conn.autocommit = False
    try:
        result = service.end_session(conn, req.user_id, req.session_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return SessionEndResponse(
        user_id=req.user_id,
        session_id=req.session_id,
        extracted=result.get("extracted", 0),
        stored=result.get("stored", 0),
    )
