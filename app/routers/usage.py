"""LLM usage read endpoints — paired with app/core/usage_tracker.py."""
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.dependencies import require_user
from app.core import usage_tracker

router = APIRouter()


@router.get("/v1/usage/today")
def usage_today(auth: Dict[str, Any] = Depends(require_user)):
    """Today's tokens + estimated cost for the calling user."""
    total = usage_tracker.daily_total(auth["user_id"])
    return {
        "user_id": auth["user_id"],
        "tokens": total["tokens"],
        "cost_usd": round(total["cost_usd"], 4),
    }


@router.get("/v1/usage")
def usage_history(auth: Dict[str, Any] = Depends(require_user), days: int = 7):
    """Per-day, per-agent, per-provider breakdown for the last `days` days."""
    rows = usage_tracker.history(auth["user_id"], days=days)
    return {
        "user_id": auth["user_id"],
        "days": days,
        "rows": rows,
        "totals": {
            "tokens": sum(r["tokens"] for r in rows),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        },
    }
