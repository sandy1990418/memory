"""
Admin endpoints for token usage monitoring and budget management.

  GET  /admin/usage/{user_id}   — token usage stats
  POST /admin/budget/{user_id}  — set daily token budget
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
import psycopg

from ..db import queries
from ..dependencies import get_db

router = APIRouter()


class BudgetRequest(BaseModel):
    daily_limit: int = Field(..., gt=0, description="Daily token budget")


class BudgetResponse(BaseModel):
    user_id: str
    daily_limit: int


class UsageResponse(BaseModel):
    user_id: str
    budget: int
    used: int
    remaining: int
    reset_at: str | None = None
    updated_at: str | None = None


@router.get("/usage/{user_id}", response_model=UsageResponse)
def get_usage(
    user_id: str,
    conn: psycopg.Connection[Any] = Depends(get_db),
) -> UsageResponse:
    """Return token usage stats for a user."""
    data = queries.get_token_usage(conn, user_id)
    return UsageResponse(**data)


@router.post("/budget/{user_id}", response_model=BudgetResponse)
def set_budget(
    user_id: str,
    req: BudgetRequest,
    conn: psycopg.Connection[Any] = Depends(get_db),
) -> BudgetResponse:
    """Set or update daily token budget for a user."""
    queries.set_token_budget(conn, user_id, req.daily_limit)
    conn.commit()
    return BudgetResponse(user_id=user_id, daily_limit=req.daily_limit)
