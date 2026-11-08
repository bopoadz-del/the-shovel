"""Account-wide daily cost guard and cited pricing. No provider calls.

Each decision is exercised with two inputs. These tests do not claim a live
deploy is priced correctly: unmapped runtime model ids stay unmapped.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete

from app.core import usage_tracker
from app.core.db import SessionLocal
from app.core.models import UsageRun

_OFFPEAK = datetime(2026, 11, 8, 2, 0, tzinfo=timezone.utc)  # Sunday
_PEAK = datetime(2026, 11, 9, 2, 0, tzinfo=timezone.utc)  # Monday 02:00 UTC
_MILLION = 1_000_000


@pytest.fixture()
def usage_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("USAGE_DAILY_CAP_USD", raising=False)
    usage_tracker._ALERTED_DAYS.clear()
    usage_tracker._UNKNOWN_LOGGED.clear()
    usage_tracker.init_db()
    with SessionLocal() as session:
        session.execute(delete(UsageRun))
        session.commit()
    yield


def _insert(user_id: str, cost: float, day: str) -> None:
    with SessionLocal() as session:
        session.add(
            UsageRun(
                id=str(uuid.uuid4()),
                user_id=user_id,
                agent_name="a",
                provider="deepseek",
                model="deepseek-flash",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=cost,
                created_at=f"{day}T12:00:00+00:00",
            )
        )
        session.commit()


@pytest.mark.parametrize(
    "provider,model,kind,cost",
    [
        ("openrouter", "openrouter/free", "free", 0.0),
        ("deepseek", "deepseek-chat", "fallback", 1.32),
    ],
)
def test_explicit_free_is_not_the_same_as_unknown(provider, model, kind, cost):
    quoted = usage_tracker.price_call(
        provider, model, _MILLION, 0, when=_OFFPEAK
    )
    assert quoted["kind"] == kind
    assert quoted["cost_usd"] == pytest.approx(cost)


@pytest.mark.parametrize(
    "when,usage,completion,expected",
    [
        (
            _OFFPEAK,
            {"prompt_cache_hit_tokens": _MILLION, "prompt_cache_miss_tokens": 0},
            0,
            0.003,
        ),
        (
            _OFFPEAK,
            {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": _MILLION},
            0,
            0.15,
        ),
        (_OFFPEAK, None, _MILLION, 0.6),
        (
            _PEAK,
            {"prompt_cache_miss_tokens": _MILLION, "prompt_cache_hit_tokens": 0},
            0,
            0.3,
        ),
    ],
)
def test_flash_cache_input_and_output_follow_the_cited_window(
    when, usage, completion, expected
):
    prompt = 0 if usage is None else _MILLION
    quoted = usage_tracker.price_call(
        "deepseek",
        "deepseek-flash",
        prompt,
        completion,
        usage=usage,
        when=when,
    )
    assert quoted["kind"] == "priced"
    assert quoted["cost_usd"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "model",
    ["deepseek-v4-flash", "deepseek-v4-flash-vision-exp"],
)
def test_documented_legacy_flash_names_use_the_flash_price(model):
    legacy = usage_tracker.price_call(
        "deepseek", model, _MILLION, 0, when=_OFFPEAK
    )
    current = usage_tracker.price_call(
        "deepseek", "deepseek-flash", _MILLION, 0, when=_OFFPEAK
    )
    assert legacy["cost_usd"] == pytest.approx(current["cost_usd"])
    assert legacy["kind"] == "priced"


@pytest.mark.parametrize(
    "provider,model",
    [("deepseek", "deepseek-chat"), ("kimi", "kimi-k2.6")],
)
def test_unknown_model_is_logged_once_per_pair(caplog, provider, model):
    usage_tracker._UNKNOWN_LOGGED.clear()
    with caplog.at_level("WARNING"):
        usage_tracker.price_call(provider, model, 10, 1, when=_OFFPEAK)
        usage_tracker.price_call(provider, model, 10, 1, when=_OFFPEAK)
    messages = [
        rec.getMessage()
        for rec in caplog.records
        if "unpriced model" in rec.getMessage()
    ]
    assert messages == [
        "unpriced model; using configured fallback rate "
        f"provider={provider} model={model}"
    ]


@pytest.mark.parametrize(
    "user_spend,other_spend,cap,over",
    [
        (6.0, 6.0, 10.0, True),
        (6.0, 1.0, 20.0, False),
    ],
)
def test_cap_is_account_wide_and_still_sees_one_user(
    usage_db, user_spend, other_spend, cap, over
):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert("user-a", user_spend, today)
    _insert("user-b", other_spend, today)
    assert usage_tracker.daily_total("user-a")["cost_usd"] == pytest.approx(user_spend)
    assert usage_tracker.is_over_cap("user-a", cap) is over


@pytest.mark.parametrize(
    "day,cost,blocked",
    [
        ("2020-01-01", 50.0, False),
        (datetime.now(timezone.utc).strftime("%Y-%m-%d"), 50.0, True),
    ],
)
def test_cap_rolls_with_the_utc_day(usage_db, day, cost, blocked):
    _insert("user-a", cost, day)
    assert usage_tracker.is_over_cap("user-a", 10.0) is blocked


@pytest.mark.parametrize(
    "cost,cap,state",
    [
        (7.9, 10.0, "ok"),
        (8.0, 10.0, "alert"),
        (10.0, 10.0, "cap"),
        (8.0, 0.0, "disabled"),
    ],
)
def test_threshold_states(cost, cap, state):
    assert usage_tracker.threshold_state(cost, cap) == state


@pytest.mark.parametrize("day", ["2026-11-08", "2026-11-09"])
def test_eighty_percent_line_is_logged_once_per_utc_day(caplog, day):
    usage_tracker._ALERTED_DAYS.clear()
    with caplog.at_level("WARNING"):
        first = usage_tracker.note_daily_threshold(8.0, 10.0, day=day)
        second = usage_tracker.note_daily_threshold(9.0, 10.0, day=day)
    assert first == "alert" and second == "alert"
    lines = [rec.getMessage() for rec in caplog.records if "80" in rec.getMessage()]
    assert lines == [f"USAGE_DAILY_CAP_USD 80% line reached for utc day {day}"]


@pytest.mark.parametrize("workers", [2, 4])
def test_concurrent_records_keep_every_call(usage_db, workers):
    import threading

    barrier = threading.Barrier(workers)

    def _work() -> None:
        barrier.wait(timeout=5)
        usage_tracker.record(
            user_id="user-a",
            agent_name="a",
            provider="openrouter",
            model="openrouter/free",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    threads = [threading.Thread(target=_work) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert usage_tracker.totals()["calls"] == workers
    assert usage_tracker.totals()["cost_usd"] == pytest.approx(0.0)


def test_missing_fallback_refuses_even_with_no_spend(monkeypatch):
    monkeypatch.setattr(usage_tracker, "_pricing_document", lambda: {"providers": {}})
    assert usage_tracker.is_over_cap("user-a", 1.0) is True
    monkeypatch.setattr(
        usage_tracker,
        "_pricing_document",
        lambda: {"providers": {}, "fallback": {"prompt": 1.0, "completion": 2.0}},
    )
    assert usage_tracker.pricing_fail_closed() is True
