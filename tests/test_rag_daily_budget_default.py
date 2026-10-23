"""The code default for the daily RAG token budget matches the live task def.

The env var ``RAG_DAILY_TOKEN_BUDGET`` still wins when it is set. With it
unset, the default is 2_000_000, not the old 500_000 code default.
"""
from __future__ import annotations


def test_daily_token_budget_default_is_two_million_and_env_overrides(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RAG_DAILY_TOKEN_BUDGET", raising=False)
    from app.core.rag import budget

    unset = budget.snapshot(day="2026-09-28")
    assert unset["budget"] == 2_000_000
    assert unset["remaining"] == 2_000_000
    assert unset["degraded"] is False

    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "12345")
    overridden = budget.snapshot(day="2026-09-28")
    assert overridden["budget"] == 12345
    assert overridden["remaining"] == 12345
