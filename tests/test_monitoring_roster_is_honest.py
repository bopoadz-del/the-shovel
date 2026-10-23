"""The provider monitor may not report on providers the platform cannot call,
and may not diagnose an outage it has no data for.

Audit 2026-08-12. Found by reading a week of production logs while answering
"what is Groq still doing here?" -- the only Groq mention in seven days was a
boot banner:

    [monitoring] Monitoring Block initialized
       Tracking: ['deepseek', 'groq', 'openai', 'anthropic', 'local_ollama']

DeepSeek, OpenAI and Anthropic were removed from the platform on 2026-07-25.
`MonitoringBlock` still carried them, so `/v1/leaderboard` -- an admin endpoint
that goes on a client's desk -- listed three providers the deployment has no
key for and cannot reach.

The worse half was `/v1/recommend`. Nothing internal calls `record_call`; the
only writer is `POST /v1/metrics/record`, which nothing calls either. So every
provider has zero observations, no entry ever reaches the "use" threshold, and
the endpoint fell through to:

    {"recommended": "local_ollama", "confidence": 100.0,
     "reason": "All cloud providers degraded - using edge fallback",
     "emergency_mode": true}

A platform-wide provider outage, reported at full confidence, as the steady
state of a healthy system. That is the confident-zero pattern this audit keeps
finding: absence of data rendered as a diagnosis. An alarm that fires on every
call is how a real degradation gets ignored.

A prior audit (§8.6) already fixed the per-provider half of this -- zero-call
providers report `"unknown"` / `null` / `"unproven"` rather than the seeded
100%. What it did not fix was the aggregate verdicts built on top of them.
"""
from __future__ import annotations

import pytest

from app.infra.monitoring import MonitoringBlock

# Providers `_llm_config` can actually return: DeepSeek (primary) and
# OpenRouter (fallback). Kimi / Groq / Ollama / OpenAI were removed.
LIVE_LADDER = {"openrouter", "deepseek"}


@pytest.fixture
def monitor():
    return MonitoringBlock(None, {})


def test_the_roster_is_exactly_the_ladder_the_platform_can_call():
    """Asserted by EQUALITY in both directions.

    A subset check would let a removed provider linger; a superset check would
    let a new one go unmonitored. This drifted for 18 days in one direction
    precisely because nothing checked the other.
    """
    monitor = MonitoringBlock(None, {})
    assert set(monitor.providers) == LIVE_LADDER, (
        "the monitored roster no longer matches the live provider ladder.\n"
        f"  monitored but uncallable: {sorted(set(monitor.providers) - LIVE_LADDER)}\n"
        f"  callable but unmonitored: {sorted(LIVE_LADDER - set(monitor.providers))}"
    )


def test_the_roster_matches_what_llm_config_can_actually_return():
    """Derived from the code rather than from LIVE_LADDER, so the constant
    above cannot become its own stale twin.

    `_llm_config` resolves an unset/unrecognised LLM_PROVIDER to deepseek, so
    driving it with each name is a complete enumeration of what the platform
    can be pointed at.
    """
    import os
    from unittest import mock

    from app.agents.runtime import _llm_config

    resolvable = set()
    for name in ("kimi", "groq", "ollama", "openai", "openrouter", "deepseek", "", "nonsense"):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": name}, clear=False):
            resolvable.add(_llm_config()["provider"])

    assert resolvable == LIVE_LADDER, (
        f"_llm_config can return {sorted(resolvable)}, but this file pins "
        f"{sorted(LIVE_LADDER)} -- update both, and the monitor's roster"
    )


@pytest.mark.parametrize(
    "dead",
    ["openai", "anthropic", "local_ollama", "kimi", "groq", "ollama"],
)
def test_a_removed_provider_is_not_monitored(dead):
    """Named explicitly, because these lingered on the roster after prior
    purges. DeepSeek and OpenRouter are the only live providers; OpenAI /
    Anthropic / Kimi / Groq / Ollama (and the old ``local_ollama`` alias)
    stay off."""
    assert dead not in MonitoringBlock(None, {}).providers


@pytest.mark.asyncio
async def test_no_recorded_calls_is_reported_as_no_basis_not_as_an_outage(monitor):
    """The finding. This is the state of every normal deployment."""
    result = await monitor.execute({"action": "recommend"})

    assert result["recommended"] is None, result
    assert result.get("emergency_mode") is False, (
        "an emergency was declared with zero observations behind it: " + str(result)
    )
    assert result.get("observed") is False, result
    assert "no basis" in result["reason"].lower(), result["reason"]
    assert "degraded" not in result["reason"].lower(), (
        f"unobserved providers are still being described as degraded: {result['reason']}"
    )


@pytest.mark.asyncio
async def test_a_genuinely_degraded_provider_still_raises_the_alarm(monitor):
    """The other direction, and the reason the fix above is not just deleting
    the alarm. Once there IS evidence, it must fire."""
    for _ in range(10):
        await monitor.execute({
            "action": "record_call", "provider": "deepseek",
            "latency_ms": 9000, "success": False, "error_type": "timeout",
        })

    result = await monitor.execute({"action": "recommend"})

    assert result["recommended"] is None, result
    assert result.get("emergency_mode") is True, (
        "ten failed calls did not raise the alarm: " + str(result)
    )
    assert result.get("observed") is True, result
    assert "deepseek" in result["reason"], result["reason"]


@pytest.mark.asyncio
async def test_a_healthy_provider_is_recommended(monitor):
    """The full third state -- otherwise the two tests above are satisfied by
    a function that never recommends anything."""
    for _ in range(10):
        await monitor.execute({
            "action": "record_call", "provider": "deepseek",
            "latency_ms": 120, "success": True,
        })

    result = await monitor.execute({"action": "recommend"})

    assert result["recommended"] == "deepseek", result
    assert result["confidence"] is not None and result["confidence"] >= 70, result


@pytest.mark.asyncio
async def test_top_provider_names_an_observed_winner_or_nothing(monitor):
    """`leaderboard[0]` is an arbitrary roster entry when nothing is observed.
    Publishing it as `top_provider` presents roster order as a measurement."""
    empty = await monitor.execute({"action": "leaderboard"})
    assert empty["top_provider"] is None, (
        f"an unproven provider was published as top: {empty['top_provider']}"
    )

    for _ in range(5):
        await monitor.execute({
            "action": "record_call", "provider": "openrouter",
            "latency_ms": 100, "success": True,
        })
    monitor.leaderboard_cache = None          # the endpoint caches for 60s

    observed = await monitor.execute({"action": "leaderboard"})
    assert observed["top_provider"] == "openrouter", observed["top_provider"]


@pytest.mark.asyncio
async def test_the_leaderboard_does_not_claim_auto_routing(monitor):
    """`_call_llm` picks its provider from `_llm_config` and its fallback from
    `_llm_fallback_config`. Neither reads this leaderboard, so `True` was a
    capability claim with nothing behind it."""
    result = await monitor.execute({"action": "leaderboard"})
    assert result["auto_route_enabled"] is False, result
    assert monitor.health()["auto_route_enabled"] is False


@pytest.mark.asyncio
async def test_the_health_report_says_unknown_rather_than_degraded_on_no_data(monitor):
    """Three states, not two. An alarm that fires on every call trains the
    reader to ignore it, which is worse than no alarm."""
    report = await monitor.execute({"action": "health_report"})

    assert report["overall_status"] == "unknown", (
        f"no observations were reported as {report['overall_status']!r}"
    )
    assert report["failover_readiness"] == "not_probed", (
        "failover readiness is asserted but never probed: "
        + str(report["failover_readiness"])
    )


@pytest.mark.asyncio
async def test_an_unknown_provider_is_refused_not_silently_recorded(monitor):
    """Metrics attributed to a provider the platform cannot call would make
    the leaderboard measure something that never ran."""
    result = await monitor.execute({
        "action": "record_call", "provider": "openai",
        "latency_ms": 100, "success": True,
    })
    assert "error" in result, result
    assert not monitor.latency_history["openai"], (
        "the call was rejected but still recorded"
    )
