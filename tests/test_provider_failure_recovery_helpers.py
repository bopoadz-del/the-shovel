"""BUILD 1 helpers: the missing-key check, the result recovery after any
provider failure, and the one unreached-model line. Every input is invented
here; two inputs per decision."""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    MODEL_UNREACHED_NOTE,
    _call_model_unless_missing,
    _missing_model_error,
    _recover_after_provider_failure,
    _resolve_model_key,
    _with_model_unreached_note,
)


@pytest.mark.parametrize("env_key", ["ALPHA_TEST_KEY", "BETA_TEST_KEY"])
def test_a_missing_key_is_named_and_a_present_key_is_not(monkeypatch, env_key):
    monkeypatch.delenv(env_key, raising=False)
    assert _missing_model_error({"env_key": env_key}, None) == f"No {env_key} configured."
    assert _missing_model_error({"env_key": env_key}, "passed-in") is None
    monkeypatch.setenv(env_key, "from-env")
    assert _missing_model_error({"env_key": env_key}, None) is None


@pytest.mark.parametrize("cfg", [{"env_key": ""}, {}])
def test_a_provider_that_declares_no_key_is_never_missing(cfg):
    assert _missing_model_error(cfg, None) is None


@pytest.mark.parametrize("env_key,env_val", [("GAMMA_TEST_KEY", "g-env"), ("DELTA_TEST_KEY", "d-env")])
def test_the_key_resolves_passed_then_env_then_missing(monkeypatch, env_key, env_val):
    cfg = {"env_key": env_key}
    assert _resolve_model_key(cfg, "passed") == (None, "passed")
    monkeypatch.setenv(env_key, env_val)
    assert _resolve_model_key(cfg, None) == (None, env_val)
    monkeypatch.delenv(env_key)
    assert _resolve_model_key(cfg, None) == (f"No {env_key} configured.", "")


class _Recorder:
    def __init__(self):
        self.calls = []

    async def _call_llm(self, messages, api_key, **kwargs):
        self.calls.append((messages, api_key, kwargs))
        return {"status": "success", "echo": api_key}


@pytest.mark.parametrize("missing", ["No EPS_TEST_KEY configured.", "No ZETA_TEST_KEY configured."])
def test_a_missing_model_is_the_calls_error_and_no_call_is_made(missing):
    import asyncio
    agent = _Recorder()
    resp = asyncio.run(_call_model_unless_missing(agent, missing, [{"role": "user"}], ""))
    assert resp == {"status": "error", "error": missing}
    assert agent.calls == []


@pytest.mark.parametrize("key,kw", [("k1", {"with_tools": True}), ("k2", {"deadline": 9.5})])
def test_a_present_model_is_called_with_the_turns_arguments(key, kw):
    import asyncio
    agent = _Recorder()
    msgs = [{"role": "user", "content": key}]
    resp = asyncio.run(_call_model_unless_missing(agent, None, msgs, key, **kw))
    assert resp == {"status": "success", "echo": key}
    assert agent.calls == [(msgs, key, kw)]


@pytest.mark.parametrize("body", ["Volume: 36.00 m3", "Moment: 45 kN m\n"])
def test_the_note_follows_the_result_once(body):
    once = _with_model_unreached_note(body)
    assert once.startswith(body.rstrip())
    assert once.endswith(MODEL_UNREACHED_NOTE)
    assert _with_model_unreached_note(once) == once


def _calc_thread(calculation: str, note: str) -> list[dict]:
    inner = {"status": "success", "calculation": calculation,
             "result": {"value": 1.0, "note": note}}
    call = {"id": "c1", "type": "function",
            "function": {"name": "construction_calc", "arguments": "{}"}}
    return [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "name": "construction_calc", "tool_call_id": "c1",
         "content": json.dumps(inner)},
        {"role": "user", "content": "PLATFORM PRE-DISPATCH: construction_calc\n"
         "Authoritative result: " + json.dumps(inner) + "\nAnswer from that result."},
    ]


@pytest.mark.parametrize("calculation, note", [
    ("Trench volume", "Bank = 4.0*1.0*2.0 = 8.00 m3"),
    ("Slab volume", "Net = 5.0*4.0*0.3 = 6.00 m3"),
])
def test_a_recorded_calculation_is_recovered_formatted(calculation, note):
    out = _recover_after_provider_failure(_calc_thread(calculation, note))
    assert note in out
    # Never the raw record or the instruction addressed to the model.
    assert '"status"' not in out and "Answer from that result" not in out


@pytest.mark.parametrize("thread", [
    [{"role": "user", "content": "explain a programme"}],
    [{"role": "user", "content": "q"},
     {"role": "tool", "name": "search_project_documents", "tool_call_id": "s1",
      "content": json.dumps({"results": [], "note": "2 rows withheld"})}],
])
def test_nothing_to_recover_returns_empty(thread):
    assert _recover_after_provider_failure(thread) == ""
