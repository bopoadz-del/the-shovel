"""Model-call meter — exposure of the existing usage_tracker on /health, the
public Prometheus /metrics, and the admin /v1/metrics.

The meter counts nothing new: ``usage_tracker`` already records every LLM
round-trip (provider, model, tokens, estimated cost). These tests pin that the
aggregate is SURFACED, and — the load-bearing property — that the PUBLIC
surfaces stay COUNT-ONLY and NAME-FREE (ORDERS_0715Z): the call count only, with
no tokens, cost, provider or model on /health or /metrics. The tokens/cost
figures and the per-provider,model breakdown appear only behind the admin
/v1/metrics gate.

Handlers are called directly (no TestClient / app lifespan) so the suite stays
light: the meter is pure DB aggregation plus guarded probes.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import delete

from app.core import usage_tracker
from app.core.db import SessionLocal
from app.core.models import UsageRun

# Public surfaces must never echo a provider or model name.
_FORBIDDEN_NAMES = ("deepseek", "openrouter", "kimi", "groq", "flash", "chat")


@pytest.fixture(autouse=True)
def _empty_runs():
    """Each test starts with an empty runs table (the suite shares one DB)."""
    usage_tracker.init_db()
    with SessionLocal() as s:
        s.execute(delete(UsageRun))
        s.commit()
    yield


def _seed(provider: str, model: str, pt: int, ct: int, user: str = "u1") -> None:
    usage_tracker.record(
        user_id=user,
        agent_name="a1",
        provider=provider,
        model=model,
        usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    )


# ── aggregate helpers ──────────────────────────────────────────────────────

def test_totals_aggregate_over_all_rows():
    _seed("deepseek", "deepseek-flash", 100, 50)
    _seed("deepseek", "deepseek-flash", 200, 20)
    _seed("openrouter", "some-free-model", 10, 5)
    t = usage_tracker.totals()
    assert t["calls"] == 3
    assert t["tokens"] == 150 + 220 + 15
    expected = round(
        usage_tracker._estimate_cost("deepseek", "deepseek-flash", 100, 50)
        + usage_tracker._estimate_cost("deepseek", "deepseek-flash", 200, 20)
        + usage_tracker._estimate_cost("openrouter", "some-free-model", 10, 5),
        6,
    )
    assert t["cost_usd"] == pytest.approx(expected)


def test_totals_by_provider_model_groups_and_orders():
    _seed("deepseek", "deepseek-flash", 100, 50)
    _seed("deepseek", "deepseek-flash", 200, 20)
    _seed("openrouter", "some-free-model", 10, 5)
    rows = usage_tracker.totals_by_provider_model()
    by = {(r["provider"], r["model"]): r for r in rows}
    assert by[("deepseek", "deepseek-flash")]["calls"] == 2
    assert by[("deepseek", "deepseek-flash")]["tokens"] == 370
    assert by[("openrouter", "some-free-model")]["calls"] == 1
    # highest call count first
    assert [r["calls"] for r in rows] == sorted((r["calls"] for r in rows), reverse=True)


def test_empty_meter_is_zero_not_error():
    assert usage_tracker.totals() == {"calls": 0, "tokens": 0, "cost_usd": 0.0}
    assert usage_tracker.totals_by_provider_model() == []


# ── public /health: name-free aggregate ─────────────────────────────────────

def test_health_meter_is_count_only_and_name_free():
    _seed("deepseek", "deepseek-flash", 100, 50)
    from app.routers.health import health

    body = health()
    assert "llm_usage" in body
    meter = body["llm_usage"]
    assert meter["calls"] == 1
    # COUNT-ONLY on the public /health (ORDERS_0715Z): no tokens, no cost_usd
    # -- those are admin-only on /v1/metrics.
    assert "tokens" not in meter
    assert "cost_usd" not in meter
    assert set(meter) <= {"calls", "error"}
    blob = str(meter).lower()
    for name in _FORBIDDEN_NAMES:
        assert name not in blob, name


# ── public /metrics: three global counters, no provider label ───────────────

def test_public_metrics_is_count_only_and_name_free():
    _seed("deepseek", "deepseek-flash", 100, 50)
    from app.routers.monitoring import prometheus_metrics

    try:
        resp = prometheus_metrics()
    except HTTPException as exc:
        if exc.status_code == 503:
            pytest.skip("prometheus-client not installed")
        raise
    text = resp.body.decode()
    assert "the_fork_llm_calls" in text
    # COUNT-ONLY on the public /metrics (ORDERS_0715Z): no tokens gauge, no
    # cost gauge -- those are admin-only on /v1/metrics.
    assert "the_fork_llm_tokens" not in text
    assert "the_fork_llm_cost_usd" not in text
    low = text.lower()
    for name in _FORBIDDEN_NAMES:
        assert name not in low, name


# ── admin /v1/metrics: provider+model breakdown, gated ──────────────────────

def test_admin_metrics_has_provider_model_breakdown():
    _seed("deepseek", "deepseek-flash", 100, 50)
    _seed("openrouter", "some-free-model", 10, 5)
    from app.routers.monitoring import get_block_metrics

    body = get_block_metrics(auth={"role": "admin"})
    assert "llm_usage" in body
    # Admin surface KEEPS tokens + cost (ORDERS_0715Z) alongside the breakdown.
    assert body["llm_usage"]["totals"]["calls"] == 2
    assert body["llm_usage"]["totals"]["tokens"] == 165
    assert "cost_usd" in body["llm_usage"]["totals"]
    rows = body["llm_usage"]["by_provider_model"]
    assert all("tokens" in r and "cost_usd" in r for r in rows)
    pairs = {(r["provider"], r["model"]) for r in rows}
    assert ("deepseek", "deepseek-flash") in pairs
    assert ("openrouter", "some-free-model") in pairs


def test_admin_metrics_forbidden_for_non_admin():
    from app.routers.monitoring import get_block_metrics

    with pytest.raises(HTTPException) as ei:
        get_block_metrics(auth={"role": "user"})
    assert ei.value.status_code == 403
