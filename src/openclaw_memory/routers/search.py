"""
Search endpoints with claude-mem-style progressive disclosure.

  POST /search          — compact index (Tier 1)
  POST /search/detail   — full content for selected IDs (Tier 3)
  GET  /search/timeline/{id} — temporal context (Tier 2)
  POST /search/answer   — RAG: search + generate answer
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
import psycopg

from ..core.service import MemoryService
from ..dependencies import get_db, get_memory_service
from ..models.search import (
    AnswerRequest,
    AnswerResponse,
    DetailRequest,
    DetailResponse,
    SearchRequest,
    SearchResponse,
    TimelineResponse,
)

router = APIRouter()


@router.post("", response_model=SearchResponse)
def search(
    req: SearchRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> SearchResponse:
    """Tier 1: Compact search returning lightweight index entries."""
    results = service.search_compact(conn, req.user_id, req.query, limit=req.limit or 20)
    return SearchResponse(
        results=[
            {
                "id": r.id,
                "title": r.title,
                "memory_type": r.memory_type,
                "score": r.score,
                "created_at": r.created_at,
                "source": r.source,
            }
            for r in results
        ],
    )


@router.post("/detail", response_model=DetailResponse)
def search_detail(
    req: DetailRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> DetailResponse:
    """Tier 3: Full content for selected memory IDs."""
    results = service.search_detail(conn, req.memory_ids)
    return DetailResponse(
        results=[
            {
                "id": r.id,
                "content": r.content,
                "memory_type": r.memory_type,
                "created_at": r.created_at,
                "source": r.source,
                "metadata": r.metadata,
            }
            for r in results
        ],
    )


@router.get("/timeline/{memory_id}", response_model=TimelineResponse)
def search_timeline(
    memory_id: str,
    user_id: str,
    depth_before: int = 3,
    depth_after: int = 3,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> TimelineResponse:
    """Tier 2: Temporal context around a specific memory."""
    result = service.search_timeline(
        conn, user_id, memory_id,
        depth_before=depth_before, depth_after=depth_after,
    )
    if result is None:
        raise HTTPException(404, "Memory not found")
    return TimelineResponse(
        id=result.id,
        content=result.content,
        memory_type=result.memory_type,
        score=result.score,
        created_at=result.created_at,
        source=result.source,
        neighbors=result.neighbors,
    )


@router.post("/answer", response_model=AnswerResponse)
def search_answer(
    req: AnswerRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> AnswerResponse:
    """RAG: Search memories and generate an LLM answer."""
    payload = service.answer(
        conn, req.user_id, req.query,
        top_k=req.top_k or 6,
        session_id=req.session_id,
    )
    return AnswerResponse(
        answer=payload.answer,
        confidence=payload.confidence,
        abstain=payload.abstain,
        abstain_reason=payload.abstain_reason,
        evidence=[
            {"memory_id": e.memory_id, "quote": e.quote, "reason": e.reason}
            for e in payload.evidence
        ],
    )
