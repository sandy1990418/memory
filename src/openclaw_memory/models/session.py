"""Pydantic models for session lifecycle endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Unique session identifier")


class SessionStartResponse(BaseModel):
    user_id: str
    session_id: str
    status: str = "started"


class MessageRequest(BaseModel):
    user_id: str
    session_id: str
    role: str = Field(..., description="Message role: user / assistant / system")
    content: str = Field(..., description="Message text content")


class MessageResponse(BaseModel):
    status: str = "recorded"


class SessionEndRequest(BaseModel):
    user_id: str
    session_id: str


class SessionEndResponse(BaseModel):
    user_id: str
    session_id: str
    extracted: int = 0
    stored: int = 0
