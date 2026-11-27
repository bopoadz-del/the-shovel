"""Unit tests that name each piece of the catalogue path directly: the
ceilings, the sheet tool schemas, the trace store, the build stamp, the
owner download, a non-object given or plan step, and the unmarked door
into the agent's own chat stream. Every value is invented for the test."""
from __future__ import annotations

import asyncio
import datetime as dt
import json

import pytest
from fastapi import HTTPException

from app.agents import driver
from app.agents.core import loop as core_loop
from app.agents.driver import loop as driver_loop
from app.agents.driver import sheet as ws
from app.agents.driver import tools, trace_store
from app.routers.admin import admin_trace_records
from tests.lane5_core_support import install_driver_stream


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DRIVER_MODE", raising=False)
    trace_store.forget_held()
    yield
    trace_store.forget_held()


# --- the ceilings -------------------------------------------------------------

@pytest.mark.parametrize("raw,rounds", [("", 12), ("3", 3), ("0", 1), ("many", 12), ("20", 20)])
def test_sheet_max_rounds(monkeypatch, raw, rounds):
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", raw)
    assert driver.sheet_max_rounds() == rounds


@pytest.mark.parametrize("raw,seconds", [("", 150.0), ("45", 45.0), ("0", 1.0), ("soon", 150.0)])
def test_sheet_max_seconds(monkeypatch, raw, seconds):
    monkeypatch.setenv("DRIVER_SHEET_MAX_SECONDS", raw)
    assert driver.sheet_max_seconds() == seconds


@pytest.mark.parametrize("ceiling", [-1.0, -0.001])
def test_the_seconds_ceiling_ends_the_rounds(monkeypatch, ceiling):
    """Past the seconds ceiling no tool round starts: only the closing call.
    A ceiling below zero is already past when the first round would start."""
    from app.agents.driver import llm

    install_driver_stream(monkeypatch)
    monkeypatch.setattr(driver, "sheet_max_seconds", lambda: ceiling)
    seen = []

    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append([t["function"]["name"] for t in offered])
        return {"status": "success", "message": {"content": "Out of time."}}

    monkeypatch.setattr(llm, "call", fake_call)
    from app.agents.runtime import Agent

    agent = Agent(name="ceiling-probe", description="probe", system_prompt="", allowed_blocks=[])

    async def go():
        return [e async for e in driver_loop.stream(agent, user_message="q", conversation_id="ws-c-1",
                                                    sheet_mode=True)]

    asyncio.run(go())
    assert seen == [[]]
    day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    assert trace_store.records(day, "ws-c-1")[0]["ended"] == "ceiling"


# --- the sheet tools ------------------------------------------------------------

@pytest.mark.parametrize("index,name,required", [(0, "sheet_update", None), (1, "run_step", ["step"])])
def test_sheet_schemas(index, name, required):
    schemas = tools.sheet_schemas()
    assert [s["function"]["name"] for s in schemas] == [tools.SHEET_UPDATE, tools.RUN_STEP]
    fn = schemas[index]["function"]
    assert fn["name"] == name
    assert fn["parameters"].get("required") == required
    if name == "sheet_update":
        assert set(fn["parameters"]["properties"]) == {"givens", "plan", "missing"}


@pytest.mark.parametrize("given", ["g1", 5, None, ["g1", 2, "m"]])
def test_a_given_that_is_not_an_object_is_refused(given):
    s = ws.new_sheet("t", "q")
    out, problems = ws.apply_update(s, {"givens": [given]})
    assert problems == ["a given must be an object"] and out is s
    assert ws._given_problems(given) == ["a given must be an object"]


@pytest.mark.parametrize("step", ["s1", 7, None])
def test_a_plan_step_that_is_not_an_object_is_refused(step):
    s = ws.new_sheet("t", "q")
    out, problems = ws.apply_update(s, {"plan": [step]})
    assert problems == ["a plan step must be an object"] and out is s
    assert ws._plan_problems(step, set()) == ["a plan step must be an object"]


# --- the trace store -----------------------------------------------------------

def _now(day):
    return dt.datetime.fromisoformat(day + "T10:00:00+00:00")


@pytest.mark.parametrize("day,conv", [("2026-11-20", "ws-a-1"), ("2026-12-01", "ws-b-2")])
def test_save_appends_one_line_and_returns_its_id(tmp_path, day, conv):
    rid = trace_store.save({"schema": "trace/1", "sheet": {"givens": [{"id": "g1"}]}}, conv, now=_now(day))
    assert rid.startswith(day + "/") and len(rid) == len(day) + 1 + 32
    lines = (tmp_path / "trace_records" / f"{day}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["record_id"] == rid and row["conversation"] == conv
    second = trace_store.save({"schema": "trace/1"}, conv, now=_now(day))
    assert second != rid and len(trace_store.records(day, conv)) == 2


@pytest.mark.parametrize("conv", ["ws-x-1", "ws-y-2"])
def test_save_that_cannot_write_returns_none(tmp_path, monkeypatch, conv):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the directory should be", encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(blocker))
    assert trace_store.save({"schema": "trace/1", "sheet": {"givens": []}}, conv) is None
    assert trace_store.last_sheet(conv) is None  # nothing held for a record not kept


@pytest.mark.parametrize("value", [12, 7.5])
def test_last_sheet_is_held_then_read_back(value):
    day = "2026-11-26"
    sheet = {"givens": [{"id": "g1", "value": value}]}
    trace_store.save({"schema": "trace/1", "sheet": sheet}, "ws-h-1", now=_now(day))
    assert trace_store.last_sheet("ws-h-1", now=_now(day)) == sheet
    trace_store.forget_held()
    assert trace_store.last_sheet("ws-h-1", now=_now(day)) == sheet          # from the file
    trace_store.forget_held()
    assert trace_store.last_sheet("ws-h-1", now=_now("2026-11-27")) == sheet  # yesterday's file
    trace_store.forget_held()
    assert trace_store.last_sheet("ws-h-1", now=_now("2026-11-29")) is None   # older: not carried
    assert trace_store.last_sheet("ws-other", now=_now(day)) is None
    assert trace_store.last_sheet(None) is None


# --- the build stamp -----------------------------------------------------------

@pytest.mark.parametrize("sha", ["0123abcd", "feedbeef99"])
def test_build_sha_comes_from_the_deploy(monkeypatch, sha):
    monkeypatch.setenv("GIT_SHA", sha)
    assert driver_loop._build_sha() == sha
    monkeypatch.delenv("GIT_SHA")
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    assert driver_loop._build_sha() == ""


# --- the owner download ---------------------------------------------------------

@pytest.mark.parametrize("conv,count", [(None, 2), ("ws-d-1", 1)])
def test_admin_trace_records(conv, count):
    day = "2026-11-25"
    trace_store.save({"schema": "trace/1", "question": "one"}, "ws-d-1", now=_now(day))
    trace_store.save({"schema": "trace/1", "question": "two"}, "ws-d-2", now=_now(day))
    response = admin_trace_records(day=day, conversation_id=conv, auth={"role": "admin"})
    body = json.loads(response.body)
    assert body["count"] == count and len(body["records"]) == count
    assert response.headers["content-disposition"] == f'attachment; filename="trace_records_{day}.json"'


@pytest.mark.parametrize("auth,day,status", [({"role": "user"}, "2026-11-25", 403),
                                             ({"role": "admin"}, "26/11/2026", 422)])
def test_admin_trace_records_refuses(auth, day, status):
    with pytest.raises(HTTPException) as err:
        admin_trace_records(day=day, conversation_id=None, auth=auth)
    assert err.value.status_code == status


# --- the unmarked door: the agent's own chat stream ---------------------------------

@pytest.mark.parametrize("profile,on_sheet", [({"answer_path": "catalogue"}, True), ({"answer_path": "x"}, False)])
def test_chat_stream_reads_the_setting_when_no_door_marked_the_turn(monkeypatch, profile, on_sheet):
    from app.agents.driver import llm
    from app.agents.runtime import Agent
    from app.core import projects

    install_driver_stream(monkeypatch)
    monkeypatch.setattr(projects, "get_project_accessible", lambda pid, uid=None: {"id": pid, "profile": profile})
    seen = []

    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append([t["function"]["name"] for t in offered])
        return {"status": "success", "message": {"content": "Answered."}}

    monkeypatch.setattr(llm, "call", fake_call)
    agent = Agent(name="door-probe", description="probe", system_prompt="", allowed_blocks=[])

    async def old_path(*a, **k):
        yield {"type": "token", "content": "old path"}
        yield {"type": "end", "complete": True}

    monkeypatch.setattr(type(agent), "_chat_stream_impl", old_path)
    assert agent.chat_stream.__func__ is core_loop.chat_stream

    async def go():
        return [e async for e in agent.chat_stream(user_message="q", project_id="p-door", user_id="u",
                                                   conversation_id="ws-p-door-1")]

    events = asyncio.run(go())
    driven = any(e.get("type") == "start" and e.get("mode") == "driver" for e in events)
    assert driven is on_sheet
    assert bool(seen and "sheet_update" in seen[0]) is on_sheet


@pytest.mark.parametrize("conv", ["ws-m-1", "ws-m-2"])
def test_save_holds_the_sheet_in_memory(tmp_path, conv):
    day = "2026-11-26"
    sheet = {"givens": [{"id": "g1", "value": len(conv)}]}
    trace_store.save({"schema": "trace/1", "sheet": sheet}, conv, now=_now(day))
    (tmp_path / "trace_records" / f"{day}.jsonl").unlink()
    assert trace_store.last_sheet(conv, now=_now(day)) == sheet  # held, not re-read
    trace_store.save({"schema": "trace/1"}, conv, now=_now(day))  # a record without a sheet
    assert trace_store.last_sheet(conv, now=_now(day)) == sheet


@pytest.mark.parametrize("missing", [None, ""])
def test_a_turn_without_a_conversation_takes_no_ones_sheet(missing):
    day = "2026-11-26"
    trace_store.save({"schema": "trace/1", "sheet": {"givens": [{"id": "g1"}]}}, "ws-n-1", now=_now(day))
    trace_store.forget_held()
    assert trace_store.last_sheet(missing, now=_now(day)) is None
