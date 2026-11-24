"""X4: a predefined workflow that built nothing must not own the turn.

A chat turn routed to the schedule workflow cancels the agent's own call. When
the work-breakdown engine then refuses (no scope it can structure), the plan
used to report that step as a success, ``cost_load`` failed on the empty
session, and the user received one line ("I could not build the schedule ...")
while every figure the request asked for went uncomputed.

Two parts are tested here, each with invented inputs:
  * the build step reports the engine's refusal as a FAILED step, in the
    engine's own words;
  * ``run_workflow`` says whether its deliverable was built, and the stream
    hands an unbuilt turn to the caller's agent stream (and only then).
"""
from __future__ import annotations

import asyncio
import json

import pytest

import app.routers.chat as chat_mod
from app.containers.construction import ConstructionContainer
from app.core import predefined_reasoning as pr
from app.core.plan_executor import PlanExecutor
from app.schemas.execution_plan import ExecutionPlan, PlanStep
from app.schemas.project_session import ProjectSession
from tests import _wbs_fakes


def _refusing_engine(monkeypatch, words):
    async def fake_generate_wbs(self, _ctx, _params):  # noqa: ANN001 - test double
        return {"status": "error", "action": "generate_wbs", "error": words}
    monkeypatch.setattr(ConstructionContainer, "generate_wbs", fake_generate_wbs)


def _session():
    return ProjectSession.new("wf-test", user_id="system")


# ── the build step does not hide the engine's refusal ─────────────────────────

@pytest.mark.parametrize("words", [
    "no schedulable scope: give the activity list",
    "brief too short to structure into activities",
])
def test_an_engine_refusal_is_a_failed_build_step_in_its_own_words(monkeypatch, words):
    _refusing_engine(monkeypatch, words)
    plan = ExecutionPlan(understanding="t", steps=[PlanStep(type="build_wbs", args={"brief": "x"})])
    s = _session()
    run = asyncio.run(PlanExecutor().run(plan, s))
    step = run.step_results[0]
    assert step.status == "error"
    assert step.error == words
    assert "activities" not in s.data


@pytest.mark.parametrize("brief", [
    "Excavate footings, pour blinding, fix rebar, cast pad foundations.",
    "Strip formwork, lay screed, tile wet areas, paint ceilings.",
])
def test_a_built_breakdown_is_still_a_successful_step(monkeypatch, brief):
    _wbs_fakes.install(monkeypatch)
    plan = ExecutionPlan(understanding="t", steps=[PlanStep(type="build_wbs", args={"brief": brief})])
    s = _session()
    run = asyncio.run(PlanExecutor().run(plan, s))
    assert run.step_results[0].status == "success"
    assert len(s.data["activities"]) > 0


# ── run_workflow says whether it built its deliverable ────────────────────────

def _ctx(message):
    return {"message": message, "project_id": None, "project_name": "P",
            "document_ids": [], "params": {}, "conversation_id": None}


@pytest.mark.parametrize("words,message", [
    ("no schedulable scope: give the activity list",
     "Wall 12 m long, crew lays 9 m2 per hour: how long, then a schedule?"),
    ("brief too short to structure into activities",
     "Trench 30 x 1 x 1.5 m at 20 m3 per day, then backfill: programme?"),
])
def test_a_workflow_that_built_nothing_says_so_with_the_failing_steps(monkeypatch, words, message):
    _refusing_engine(monkeypatch, words)
    out = asyncio.run(pr.run_workflow("generate_wbs", _ctx(message), _session()))
    assert out["handled"] is True
    assert out["built"] is False
    assert out["unbuilt_reasons"][0] == f"build_wbs: {words}"


@pytest.mark.parametrize("message", [
    "Excavate footings, pour blinding, fix rebar, cast pad foundations: schedule it.",
    "Strip formwork, lay screed, tile wet areas, paint ceilings: schedule it.",
])
def test_a_workflow_that_built_its_schedule_says_built(monkeypatch, message):
    _wbs_fakes.install(monkeypatch)
    out = asyncio.run(pr.run_workflow("generate_wbs", _ctx(message), _session()))
    assert out["built"] is True
    assert out["unbuilt_reasons"] == []
    assert "working days" in out["answer"]


# ── the stream hands an unbuilt turn to the caller's agent ────────────────────

def _collect(gen):
    async def run():
        evs = []
        async for chunk in gen:
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    evs.append(json.loads(line[6:]))
        return evs
    return asyncio.run(run())


def _fallback(text):
    calls = []

    async def agent_stream():
        calls.append(1)
        yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'complete': True})}\n\n"
    return agent_stream, calls


def _stream(message, fallback):
    return chat_mod._stream_from_predefined(
        action="generate_wbs", user_message=message, project_id=None,
        user_id="u1", session_id="sess", document_ids=[], emit_start=False,
        fallback=fallback)


@pytest.mark.parametrize("message,agent_text", [
    ("Wall 12 m long, crew lays 9 m2 per hour: how long, then a schedule?",
     "Area 36 m2, 4 hours."),
    ("Trench 30 x 1 x 1.5 m at 20 m3 per day, then backfill: programme?",
     "Volume 45 m3, 2.25 days."),
])
def test_an_unbuilt_workflow_hands_the_turn_to_the_agent(monkeypatch, message, agent_text):
    _refusing_engine(monkeypatch, "no schedulable scope")
    fb, calls = _fallback(agent_text)
    evs = _collect(_stream(message, fb))
    text = "".join(e.get("content", "") for e in evs if e["type"] == "token")
    assert calls == [1]
    assert text == agent_text
    assert not any(e.get("mode") == "predefined" for e in evs)


@pytest.mark.parametrize("message", [
    "Excavate footings, pour blinding, fix rebar, cast pad foundations: schedule it.",
    "Strip formwork, lay screed, tile wet areas, paint ceilings: schedule it.",
])
def test_a_built_workflow_keeps_the_turn(monkeypatch, message):
    _wbs_fakes.install(monkeypatch)
    fb, calls = _fallback("agent text")
    evs = _collect(_stream(message, fb))
    end = next(e for e in evs if e["type"] == "end")
    assert calls == []
    assert end["workflow"] == "generate_wbs"
    assert end["mode"] == "predefined"


@pytest.mark.parametrize("message", [
    "Wall 12 m long, crew lays 9 m2 per hour: how long, then a schedule?",
    "Trench 30 x 1 x 1.5 m at 20 m3 per day, then backfill: programme?",
])
def test_without_a_fallback_the_stream_keeps_its_own_answer(monkeypatch, message):
    _refusing_engine(monkeypatch, "no schedulable scope")
    evs = _collect(_stream(message, None))
    end = next(e for e in evs if e["type"] == "end")
    assert end["workflow"] == "generate_wbs"
    text = "".join(e.get("content", "") for e in evs if e["type"] == "token")
    assert text.strip() == "I could not build the schedule for this request."
