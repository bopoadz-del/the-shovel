"""Tests for the soft daily cost cap in Agent._call_llm.

The cap is read from USAGE_DAILY_CAP_USD. When today's spend for the
calling user is already at or above the cap, _call_llm must short-circuit
with a structured error and MUST NOT issue the upstream HTTP request.

Properties verified:
- short-circuits when over cap (returns status=error, never hits httpx)
- doesn't short-circuit when under cap
- ignores cap when user_id is None (internal call path)
- ignores cap when env var unset, zero, or unparseable
- tracker exceptions don't sink real calls (failure mode is "allow")
"""

import asyncio
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from app.agents.runtime import Agent
from tests._fixture_values import ENV_SECRET_02


def _make_agent():
    return Agent(
        name="test-agent",
        description="cap test",
        system_prompt="x",
        allowed_blocks=[],
    )


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def fresh_usage_db(monkeypatch):
    """Point usage_tracker at a throwaway DATA_DIR so each test starts
    with an empty runs table."""
    tmpdir = tempfile.mkdtemp(prefix="usage-test-")
    monkeypatch.setenv("DATA_DIR", tmpdir)
    # Pricing override env should be unset so we fall back to the default
    # config file (rates don't matter — we seed rows manually).
    monkeypatch.delenv("LLM_PRICING_FILE", raising=False)
    yield tmpdir


def _seed_usage(user_id: str, cost_usd: float):
    """Insert a runs row attributed to today so daily_total() returns it."""
    from datetime import datetime, timezone

    from app.core import usage_tracker
    from app.core.db import SessionLocal, get_database_url
    from app.core.models import UsageRun

    usage_tracker.init_db()
    now = datetime.now(timezone.utc).isoformat()
    if get_database_url().startswith("sqlite"):
        with sqlite3.connect(usage_tracker._db_path()) as conn:
            conn.execute(
                "INSERT INTO runs (id, user_id, agent_name, provider, model, "
                "prompt_tokens, completion_tokens, total_tokens, "
                "estimated_cost_usd, created_at) VALUES "
                "(?, ?, 'x', 'groq', 'm', 0, 0, 0, ?, ?)",
                ("seed-" + user_id, user_id, cost_usd, now),
            )
            conn.commit()
        return
    with SessionLocal() as session:
        session.add(
            UsageRun(
                id="seed-" + user_id,
                user_id=user_id,
                agent_name="x",
                provider="groq",
                model="m",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=cost_usd,
                created_at=now,
            )
        )
        session.commit()


# ── short-circuit when over cap ──────────────────────────────────────────────

def test_call_llm_short_circuits_when_over_cap(fresh_usage_db, monkeypatch):
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "1.00")
    monkeypatch.setenv("GROQ_API_KEY", ENV_SECRET_02)  # force groq path
    _seed_usage("user-A", 1.50)

    agent = _make_agent()
    # If we hit httpx the test fails — short-circuit must beat the call.
    with patch("app.agents.runtime.httpx.AsyncClient") as mock_client:
        resp = _run(agent._call_llm([], "key", user_id="user-A"))
        mock_client.assert_not_called()

    assert resp["status"] == "error"
    assert "Daily LLM cost cap" in resp["error"]
    assert "$1.50" in resp["error"] or "1.5000" in resp["error"]


# ── pass through when under cap ──────────────────────────────────────────────

def test_call_llm_passes_through_when_under_cap(fresh_usage_db, monkeypatch):
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "10.00")
    monkeypatch.setenv("GROQ_API_KEY", ENV_SECRET_02)
    _seed_usage("user-B", 0.10)

    agent = _make_agent()
    # Stub the HTTP layer so we don't actually call Groq but DO observe
    # that the code reached the call site.
    fake = MagicMock()
    fake_post_resp = MagicMock(status_code=200)
    fake_post_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.post = AsyncMock(return_value=fake_post_resp)
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=False)
    with patch("app.agents.runtime.httpx.AsyncClient", return_value=fake):
        resp = _run(agent._call_llm([], "key", user_id="user-B"))

    assert resp["status"] == "success"
    fake.post.assert_awaited_once()


# ── no user_id => never blocked ──────────────────────────────────────────────

def test_call_llm_ignores_cap_when_no_user_id(fresh_usage_db, monkeypatch):
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "0.01")
    monkeypatch.setenv("GROQ_API_KEY", ENV_SECRET_02)
    _seed_usage("", 99.0)  # anonymous bucket already way over

    agent = _make_agent()
    fake = MagicMock()
    fake_post_resp = MagicMock(status_code=200)
    fake_post_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.post = AsyncMock(return_value=fake_post_resp)
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=False)
    with patch("app.agents.runtime.httpx.AsyncClient", return_value=fake):
        resp = _run(agent._call_llm([], "key", user_id=None))

    assert resp["status"] == "success"
    fake.post.assert_awaited_once()


# ── env var unset / zero / unparseable => never blocked ──────────────────────

@pytest.mark.parametrize("cap_env", [None, "", "0", "-5", "not-a-number"])
def test_call_llm_ignores_invalid_cap(fresh_usage_db, monkeypatch, cap_env):
    if cap_env is None:
        monkeypatch.delenv("USAGE_DAILY_CAP_USD", raising=False)
    else:
        monkeypatch.setenv("USAGE_DAILY_CAP_USD", cap_env)
    monkeypatch.setenv("GROQ_API_KEY", ENV_SECRET_02)
    _seed_usage("user-C", 100.0)

    agent = _make_agent()
    fake = MagicMock()
    fake_post_resp = MagicMock(status_code=200)
    fake_post_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.post = AsyncMock(return_value=fake_post_resp)
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=False)
    with patch("app.agents.runtime.httpx.AsyncClient", return_value=fake):
        resp = _run(agent._call_llm([], "key", user_id="user-C"))

    assert resp["status"] == "success"


# ── tracker raises => fail open ──────────────────────────────────────────────

def test_call_llm_fails_open_when_tracker_raises(fresh_usage_db, monkeypatch):
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "1.00")
    monkeypatch.setenv("GROQ_API_KEY", ENV_SECRET_02)

    agent = _make_agent()
    fake = MagicMock()
    fake_post_resp = MagicMock(status_code=200)
    fake_post_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.post = AsyncMock(return_value=fake_post_resp)
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=False)

    with patch("app.core.usage_tracker.is_over_cap", side_effect=RuntimeError("db gone")), \
         patch("app.agents.runtime.httpx.AsyncClient", return_value=fake):
        resp = _run(agent._call_llm([], "key", user_id="user-D"))

    assert resp["status"] == "success"
