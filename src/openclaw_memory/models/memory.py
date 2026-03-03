"""Pydantic models for memory CRUD endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryAddRequest(BaseModel):
    user_id: str
    conversation: list[dict[str, Any]] | None = Field(
        default=None,
        description="Conversation to extract memories from",
    )
    content: str | None = Field(
        default=None,
        description="Direct memory content (alternative to conversation)",
    )
    memory_type: str | None = None
    metadata: dict[str, Any] | None = None
    session_id: str | None = None


class MemoryAddResponse(BaseModel):
    stored: int = 0
    extracted: int = 0
    memory_id: str | None = None


class MemoryItem(BaseModel):
    id: str
    content: str
    memory_type: str
    created_at: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryListResponse(BaseModel):
    user_id: str
    memories: list[MemoryItem]
    total: int


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    memory_type: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryDeleteResponse(BaseModel):
    deleted: bool
    memory_id: str
