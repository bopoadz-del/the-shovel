"""A template deliverable declines an ask it cannot take as written -- on
EVERY path that could hand it the turn, not only the runtime predispatch.

``cash_flow_forecast`` draws a template curve from ONE money total and a
horizon. An ask that writes out its own period-by-period figures can be
neither read by it (it stops on "contract_value required", printed to the
user as "I need the contract value") nor honoured (it would print its own
curve). The runtime predispatch already declined such an ask; three other
doors did not:

* ``understand_intent`` forced the template on the cash-flow wording alone,
  and ``/v1/chat/stream`` ran it as a predefined workflow;
* ``/v1/agents/{name}/chat/stream`` ran the orchestrator's routed action as
  a predefined workflow without asking;
* the agent loop still offered the tool to the model.

Here every door is driven through its real entry point. Every input is
invented for this file; two inputs per decision, and each decision has the
other side (a single-total ask still reaches the template).
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient

import app.agents.runtime as rt
import app.routers.agents as ag
import app.routers.chat as chat_mod
from app.core import action_router
from app.core.dynamic_reasoning import understand_intent
from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import PW_06, STUB_KEY_11

# Cash-flow / S-curve wording AND the user's own period figures.
SERIES_ASKS = (
    "Works valued per quarter: 2.4, 3.1, 3.6, 1.9 M; cost runs at 88 % of "
    "value. Advance 12 %, retention 7 %, paid 45 days after each quarter. "
    "Give me the cash flow and the funding peak.",
    "Crew hours budgeted by fortnight 640, 720, 810, 455 and booked 600, "
    "735, 790 to date. Plot both cumulative S-curves and say how far behind.",
)
# Cash-flow / S-curve wording and ONE total + a horizon: the template's own
# contract, which must still reach it.
TOTAL_ASKS = (
    "Prepare a 16-month cash flow forecast for a SAR 48,000,000 hospital.",
    "Draw the S-curve for a USD 7.25M school extension over 11 months.",
)


# -- the shared rule ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    (SERIES_ASKS[0], True),
    (SERIES_ASKS[1], True),
    (TOTAL_ASKS[0], False),
    (TOTAL_ASKS[1], False),
])
def test_the_shared_reader_finds_a_written_series(text, expected):
    assert action_router.message_supplies_period_series(text) is expected


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_the_template_declines_a_series_ask(text):
    assert action_router.template_declines_ask("cash_flow_forecast", text) is True
    assert "cash_flow_forecast" in action_router.template_actions_declining(text)


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_the_template_takes_a_single_total_ask(text):
    assert action_router.template_declines_ask("cash_flow_forecast", text) is False
    assert action_router.template_actions_declining(text) == set()


@pytest.mark.parametrize("action", [None, "", "generate_wbs", "look_ahead"])
def test_an_action_with_no_template_limit_never_declines(action):
    for text in SERIES_ASKS:
        assert action_router.template_declines_ask(action, text) is False


@pytest.mark.parametrize("text", SERIES_ASKS + TOTAL_ASKS)
def test_the_runtime_predispatch_reads_the_same_rule(text):
    """One rule, not two copies: the predispatch and the front doors agree."""
    assert rt._message_wants_cash_flow_forecast(text) is (
        not action_router.template_declines_ask("cash_flow_forecast", text)
    )


# -- door 1: understand_intent (the /v1/chat/stream classifier) --------------

def _no_model(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("the intent model must not be asked")
    monkeypatch.setattr("app.core.dynamic_reasoning.complete_json", boom)


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_understand_intent_does_not_force_the_template_on_a_series(text, monkeypatch):
    _no_model(monkeypatch)
    out = asyncio.run(understand_intent(text))
    assert out["action"] is None, out


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_understand_intent_still_forces_the_template_on_a_total(text, monkeypatch):
    _no_model(monkeypatch)
    out = asyncio.run(understand_intent(text))
    assert out["action"] == "cash_flow_forecast", out


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_a_classifier_that_still_picks_the_template_is_declined(text):
    """Whichever classifier names the template, the router asks the rule."""
    assert chat_mod._predefined_declines("cash_flow_forecast", text) is True


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_the_router_lets_the_template_take_a_total(text):
    assert chat_mod._predefined_declines("cash_flow_forecast", text) is False


# -- the agent loop: what the model is offered -------------------------------

def _record_offers(monkeypatch, offers: list):
    async def fake_call_llm(self, messages, api_key=None, project_id=None,
                            with_tools=True, user_id=None, exclude_tools=None):
        offers.append(set(exclude_tools or ()))
        return {"status": "success",
                "choice": {"message": {"role": "assistant", "content": "noted"}}}
    monkeypatch.setattr(rt.Agent, "_call_llm", fake_call_llm)


def _agent():
    return rt.Agent(name="series-probe", description="", system_prompt="answer",
                    allowed_blocks=["construction"])


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_the_model_is_not_offered_the_template_on_a_series(text, monkeypatch):
    offers: list = []
    _record_offers(monkeypatch, offers)
    asyncio.run(_agent().chat(text, api_key=STUB_KEY_11))
    assert offers, "the model was never called"
    assert all("cash_flow_forecast" in ex for ex in offers), offers


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_the_model_keeps_the_template_on_a_total(text, monkeypatch):
    offers: list = []
    _record_offers(monkeypatch, offers)
    asyncio.run(_agent().chat(text, api_key=STUB_KEY_11))
    assert all("cash_flow_forecast" not in ex for ex in offers), offers


# -- door 1 end to end: /v1/chat/stream --------------------------------------

def _template_is_a_workflow(monkeypatch):
    """Profile-independent: the template is a predefined workflow, as it is
    under the construction kit, so only the decline rule decides the door."""
    monkeypatch.setattr(chat_mod, "_predefined_workflows",
                        lambda: {"cash_flow_forecast"})


def _predefined_recorder(monkeypatch, seen: list):
    async def fake_predefined(**kwargs):
        seen.append(kwargs["action"])
        yield 'data: {"type": "start", "mode": "predefined"}\n\n'
        yield 'data: {"type": "end", "mode": "predefined"}\n\n'
    monkeypatch.setattr(chat_mod, "_stream_from_predefined", fake_predefined)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _chat_stream(client, monkeypatch, text):
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")
    _template_is_a_workflow(monkeypatch)
    _no_model(monkeypatch)

    async def fake_classify(prompt):
        return (None, 0.0)
    monkeypatch.setattr(chat_mod, "_classify_intent", fake_classify)
    _record_offers(monkeypatch, [])
    seen: list = []
    _predefined_recorder(monkeypatch, seen)
    r = client.post("/v1/chat/stream", json={"prompt": text},
                    headers={"Authorization": "Bearer " + DEV_KEY})
    assert r.status_code == 200
    return seen


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_chat_stream_does_not_run_the_template_on_a_series(client, monkeypatch, text):
    assert _chat_stream(client, monkeypatch, text) == []


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_chat_stream_runs_the_template_on_a_total(client, monkeypatch, text):
    assert _chat_stream(client, monkeypatch, text) == ["cash_flow_forecast"]


# -- door 2 end to end: /v1/agents/{name}/chat/stream ------------------------

def _token(c: TestClient) -> str:
    email = f"series-{uuid.uuid4().hex[:8]}@x.com"
    c.post("/v1/users/register", json={"email": email, "password": PW_06})
    return c.post("/v1/users/login",
                  json={"email": email, "password": PW_06}).json()["token"]


def _agents_stream(client, monkeypatch, text):
    """The orchestrator routes the template action, as it did live."""
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")
    _template_is_a_workflow(monkeypatch)

    async def routed(message, agent):
        return agent, {"requested": agent.name, "final": agent.name,
                       "action": "cash_flow_forecast", "confidence": 0.2}
    monkeypatch.setattr(ag, "select_agent_for_message", routed)
    offers: list = []
    _record_offers(monkeypatch, offers)
    seen: list = []
    _predefined_recorder(monkeypatch, seen)
    tok = _token(client)
    r = client.post("/v1/agents/project-assistant/chat/stream",
                    headers={"Authorization": f"Bearer {tok}"},
                    json={"message": text, "project_id": None})
    assert r.status_code == 200
    return seen, offers, r.text


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_agents_stream_leaves_a_series_to_the_agent(client, monkeypatch, text):
    seen, _offers, body = _agents_stream(client, monkeypatch, text)
    assert seen == []
    assert '"mode": "predefined"' not in body


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_agents_stream_runs_the_template_on_a_total(client, monkeypatch, text):
    seen, _offers, _body = _agents_stream(client, monkeypatch, text)
    assert seen == ["cash_flow_forecast"]


def test_the_events_are_json(client, monkeypatch):
    """Sanity: the agent path still streams well-formed SSE on a declined ask."""
    _seen, _offers, body = _agents_stream(client, monkeypatch, SERIES_ASKS[0])
    for line in body.splitlines():
        if line.startswith("data: "):
            json.loads(line[6:])
