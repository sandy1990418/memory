"""Pydantic models for search endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# -- Tier 1: Compact search --


class SearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = Field(default=20, ge=1, le=100)


class SearchResultItem(BaseModel):
    id: str
    title: str
    memory_type: str
    score: float
    created_at: str
    source: str


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]


# -- Tier 3: Detail --


class DetailRequest(BaseModel):
    memory_ids: list[str] = Field(..., min_length=1, max_length=50)


class DetailResultItem(BaseModel):
    id: str
    content: str
    memory_type: str
    created_at: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetailResponse(BaseModel):
    results: list[dict[str, Any]]


# -- Tier 2: Timeline --


class TimelineNeighbor(BaseModel):
    id: str
    title: str
    created_at: str | None = None


class TimelineResponse(BaseModel):
    id: str
    content: str
    memory_type: str
    score: float
    created_at: str
    source: str
    neighbors: list[dict[str, Any]] = Field(default_factory=list)


# -- Answer (RAG) --


class AnswerRequest(BaseModel):
    user_id: str
    query: str
    top_k: int | None = Field(default=6, ge=1, le=20)
    session_id: str | None = None


class AnswerEvidenceItem(BaseModel):
    memory_id: str
    quote: str
    reason: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    abstain: bool
    abstain_reason: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
