"""The catalogue path (DESIGN V2, PR-1): a project whose profile sets
``answer_path: "catalogue"`` takes the driver with a working sheet; the sheet
is handed back every round; the sheet tools are never withdrawn; a
calculator's question does not end the turn while a plan step is open; one
trace record (trace/1) is written per turn and the owner downloads it.
Every project, figure and answer below is invented for the test; the model
is scripted."""
from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json

import pytest

from app.agents import driver
from app.agents.driver import llm, loop, trace_store
from tests.lane5_core_support import install_driver_stream


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DRIVER_MODE", raising=False)
    install_driver_stream(monkeypatch)
    trace_store.forget_held()
    yield
    trace_store.forget_held()


def _agent():
    from app.agents.runtime import Agent

    return Agent(name="sheet-probe", description="probe", system_prompt="", allowed_blocks=[])


def _call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _scripted(monkeypatch, replies):
    seen = []

    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append({"messages": [dict(m) for m in messages],
                     "tools": [t["function"]["name"] for t in offered]})
        return {"status": "success", "message": replies.pop(0)}

    monkeypatch.setattr(llm, "call", fake_call)
    return seen


def _run(**kw):
    async def go():
        return [e async for e in loop.stream(_agent(), **kw)]
    return asyncio.run(go())


def _text(events):
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


def _today():
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _sheet_in(messages):
    note = [m["content"] for m in messages if m["role"] == "system" and m["content"].startswith("Working sheet")]
    assert len(note) == 1
    return json.loads(note[0].split("\n", 1)[1])


# --- the project setting ----------------------------------------------------

@pytest.mark.parametrize("profile,expected", [
    ({"answer_path": "catalogue"}, True),
    ({"answer_path": " Catalogue "}, True),
    ({"answer_path": "default"}, False),
    ({"client": "x"}, False),
    (None, False),
])
def test_the_setting_is_read_from_the_projects_own_profile(monkeypatch, profile, expected):
    from app.core import projects

    monkeypatch.setattr(projects, "get_project_accessible",
                        lambda pid, uid=None: {"id": pid, "profile": profile})
    assert driver.catalogue_path("p-1", "u-1") is expected
    assert driver.catalogue_path(None, "u-1") is False


def test_an_unreadable_profile_keeps_the_default_path(monkeypatch):
    from app.core import projects

    def boom(pid, uid=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(projects, "get_project_accessible", boom)
    assert driver.catalogue_path("p-1") is False
    monkeypatch.setattr(projects, "get_project_accessible", lambda pid, uid=None: None)
    assert driver.catalogue_path("p-1") is False


@pytest.mark.parametrize("mark,expected", [(True, True), (False, False)])
def test_the_turns_mark_switches_the_driver_on_without_driver_mode(mark, expected):
    def in_turn():
        assert driver.on_sheet() is None and driver.enabled() is False
        driver.mark_project(mark)
        return driver.on_sheet(), driver.enabled()

    assert contextvars.copy_context().run(in_turn) == (mark, expected)


# --- the sheet turn ---------------------------------------------------------

@pytest.mark.parametrize("total,area,shown", [(2059200, 3000, "686.4"), (450000, 1200, "375")])
def test_a_sheet_turn_plans_runs_in_code_and_records_the_trace(monkeypatch, total, area, shown):
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "what": "total", "value": total, "unit": "currency", "source": "user"},
                       {"id": "g2", "what": "area", "value": area, "unit": "m2", "source": "user"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic",
                      "inputs": {"operator": "divide", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The rate is {shown} currency/m2."},
    ])
    events = _run(user_message="what is the rate", conversation_id="ws-p1-1", project_id=None,
                  sheet_mode=True)
    assert shown in _text(events)
    # The sheet is handed back every round, as it stands after the last call.
    assert [len(_sheet_in(s["messages"])["givens"]) for s in seen] == [0, 2, 2]
    assert [len(_sheet_in(s["messages"])["steps"]) for s in seen] == [0, 0, 1]
    # The sheet tools are offered on every round, never withdrawn.
    assert all({"sheet_update", "run_step"} <= set(s["tools"]) for s in seen)
    end = [e for e in events if e.get("type") == "end"][0]
    assert end["trace_record"].startswith(_today() + "/")
    rows = trace_store.records(_today(), "ws-p1-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "trace/1" and row["ended"] == "answer" and row["question_set"] == "unset"
    assert row["sheet"]["done"] is True
    assert row["sheet"]["steps"][0]["result"]["value"] == pytest.approx(total / area)
    assert row["figures_from_tools"] is True and row["rounds_used"] == 3
    assert row["record_id"] == end["trace_record"]
    assert row["rounds"][1]["calls"][0]["tool"] == "run_step" and row["rounds"][1]["calls"][0]["step"] == "s1"
    assert "step" not in row["rounds"][0]["calls"][0]  # only a run_step call names a step


@pytest.mark.parametrize("reply", ["A plain answer.", "Another reply here."])
def test_without_the_setting_no_sheet_and_no_record(monkeypatch, reply):
    seen = _scripted(monkeypatch, [{"content": reply}])
    events = _run(user_message="hello", conversation_id="ws-p2-1")
    assert reply in _text(events)
    assert "sheet_update" not in seen[0]["tools"]
    assert not any(m["content"].startswith("Working sheet") for m in seen[0]["messages"] if m["role"] == "system")
    assert "trace_record" not in [e for e in events if e.get("type") == "end"][0]
    assert trace_store.records(_today()) == []


def _asking_calculator(monkeypatch, agent_cls, question="What is the unit rate?"):
    async def fake_tool(self, call, *a, **k):
        return {"ok": False, "needs_input": True,
                "result": {"calculation": "unit_cost_total", "status": "needs_input",
                           "missing": ["unit_rate"], "question": question}}

    monkeypatch.setattr(agent_cls, "_run_tool_call", fake_tool)


@pytest.mark.parametrize("quantity", [40, 12])
def test_a_calculators_question_does_not_end_the_turn_while_a_step_is_open(monkeypatch, quantity):
    from app.agents.runtime import Agent

    _asking_calculator(monkeypatch, Agent)
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": quantity, "unit": "nr"}],
            "plan": [{"id": "s1", "entry": "formula:unit_cost_total", "inputs": {"quantity": "g1"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("construction_calc", {
            "calculation": "unit_cost_total", "params": {"quantity": quantity}}, "c2")]},
        {"content": "", "tool_calls": [_call("sheet_update", {
            "missing": [{"for": "s1", "input": "unit_rate", "label": "unit rate"}]}, "c3")]},
        {"content": f"I need the unit rate for the {quantity} items to finish."},
    ])
    events = _run(user_message="cost of the items", conversation_id="ws-p3-1", sheet_mode=True)
    assert len(seen) == 4  # the model had the round after the calculator asked
    assert "unit rate" in _text(events)
    row = trace_store.records(_today(), "ws-p3-1")[0]
    assert row["sheet"]["givens"][0]["value"] == quantity
    assert row["sheet"]["missing"] == [{"for": "s1", "input": "unit_rate", "label": "unit rate", "asked": False}]
    assert row["ended"] == "answer"


@pytest.mark.parametrize("question", ["What is the unit rate?", "What is the crew size?"])
def test_off_the_sheet_the_calculators_question_still_ends_the_turn(monkeypatch, question):
    from app.agents.runtime import Agent

    _asking_calculator(monkeypatch, Agent, question)
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("construction_calc", {
            "calculation": "unit_cost_total", "params": {"quantity": 40}}, "c2")]},
        {"content": "never reached"},
    ])
    events = _run(user_message="cost of the items", conversation_id="ws-p4-1")
    assert len(seen) == 1
    assert question in _text(events)


@pytest.mark.parametrize("question", ["What is the unit rate?", "What is the crew size?"])
def test_on_the_sheet_with_no_open_step_the_question_ends_the_turn(monkeypatch, question):
    from app.agents.runtime import Agent

    _asking_calculator(monkeypatch, Agent, question)
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("construction_calc", {
            "calculation": "unit_cost_total", "params": {"quantity": 40}}, "c2")]},
        {"content": "never reached"},
    ])
    events = _run(user_message="cost of the items", conversation_id="ws-p5-1", sheet_mode=True)
    assert len(seen) == 1
    assert question in _text(events)
    assert trace_store.records(_today(), "ws-p5-1")[0]["ended"] == "asked_input"


@pytest.mark.parametrize("rounds", [2, 3])
def test_the_round_ceiling_ends_a_turn_whose_step_stays_open(monkeypatch, rounds):
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", str(rounds))
    replies = [{"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, f"c{i}")]} for i in range(rounds)]
    seen = _scripted(monkeypatch, replies + [{"content": "Stopped before the step could run."}])
    events = _run(user_message="q", conversation_id="ws-p6-1", sheet_mode=True)
    assert len(seen) == rounds + 1
    assert seen[-1]["tools"] == []  # only the ceiling's closing call has no tools
    row = trace_store.records(_today(), "ws-p6-1")[0]
    assert row["ended"] == "ceiling" and row["sheet"]["steps"][0]["status"] == "failed"
    assert [e for e in events if e.get("type") == "end"][0]["trace_record"]


@pytest.mark.parametrize("value,needed", [(12, "width"), (7.5, "depth")])
def test_the_next_turn_starts_from_the_last_sheet(monkeypatch, value, needed):
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": value, "unit": "m"}],
            "missing": [{"for": "s1", "input": needed, "label": needed}]}, "c1")]},
        {"content": "What is the width?"},
    ])
    _run(user_message="first", conversation_id="ws-p7-1", sheet_mode=True)
    trace_store.forget_held()  # read back from the file, as after a restart
    seen = _scripted(monkeypatch, [{"content": "Thanks."}])
    _run(user_message="second", conversation_id="ws-p7-1", sheet_mode=True)
    sheet = _sheet_in(seen[0]["messages"])
    assert sheet["givens"][0]["value"] == value and sheet["givens"][0]["source"] == "user"
    assert sheet["missing"][0]["input"] == needed
    seen = _scripted(monkeypatch, [{"content": "New."}])
    _run(user_message="other", conversation_id="ws-p8-1", sheet_mode=True)
    assert _sheet_in(seen[0]["messages"])["givens"] == []


@pytest.mark.parametrize("named,recorded", [("teaching", "teaching"), ("frozen", "frozen"),
                                             ("Held_Back", "held_back"), ("other", "unset")])
def test_a_project_question_set_is_recorded(monkeypatch, named, recorded):
    from app.core import projects

    monkeypatch.setattr(projects, "get_project_accessible",
                        lambda pid, uid=None: {"id": pid, "profile": {"answer_path": "catalogue",
                                                                      "question_set": named}})
    _scripted(monkeypatch, [{"content": "Done."}])
    _run(user_message="q", conversation_id="ws-p9-1", project_id="p9", sheet_mode=True)
    assert trace_store.records(_today(), "ws-p9-1")[0]["question_set"] == recorded


# --- the public route and the owner download --------------------------------

@pytest.mark.parametrize("ask", ["hello", "what is the rate per m2"])
def test_the_public_route_takes_the_sheet_for_a_catalogue_project(monkeypatch, ask):
    from fastapi.testclient import TestClient

    from app.core import jwt_auth
    from app.core import projects as store
    from app.core import users as users_store
    from app.main import app

    def headers(uid, role):
        users_store.ensure_user_exists(uid, role=role, email=f"{uid}@local")
        return {"Authorization": f"Bearer {jwt_auth.create_token(uid)}"}

    admin, user = headers("cp-admin", "admin"), headers("cp-user", "user")
    on = store.create_project("sheet project", user_id="cp-user")["id"]
    off = store.create_project("plain project", user_id="cp-user")["id"]
    seen = _scripted(monkeypatch, [{"content": "Sheet answer."}, {"content": "Chat door answer."}])
    with TestClient(app) as client:
        r = client.put(f"/v1/admin/projects/{on}/profile", json={"answer_path": "catalogue"}, headers=admin)
        assert r.status_code == 200
        conv = f"ws-{on}-1700000000000"
        text = client.post("/v1/agents/project-assistant/chat/stream",
                           json={"message": ask, "project_id": on, "conversation_id": conv},
                           headers=user).text
        assert '"mode": "driver"' in text and '"trace_record"' in text
        assert "sheet_update" in seen[0]["tools"]
        # The other door: the chat stream marks the turn from its body's project.
        conv2 = f"ws-{on}-1700000000001"
        text = client.post("/v1/chat/stream", json={"message": "hello", "project_id": on,
                                                    "conversation_id": conv2}, headers=user).text
        assert '"mode": "driver"' in text and "sheet_update" in seen[1]["tools"]

        async def no_driver(*a, **k):
            raise AssertionError("the driver ran for a project without the setting")
            yield  # pragma: no cover

        monkeypatch.setattr(loop, "stream", no_driver)
        text = client.post("/v1/agents/project-assistant/chat/stream",
                           json={"message": "hello", "project_id": off,
                                 "conversation_id": f"ws-{off}-1700000000000"}, headers=user).text
        assert '"mode": "driver"' not in text

        assert client.get(f"/v1/admin/trace-records?day={_today()}", headers=user).status_code == 403
        assert client.get("/v1/admin/trace-records?day=not-a-day", headers=admin).status_code == 422
        r = client.get(f"/v1/admin/trace-records?day={_today()}&conversation_id={conv}", headers=admin)
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        body = r.json()
        assert body["count"] == 1 and body["records"][0]["conversation"] == conv
        assert body["records"][0]["final_answer"].startswith("Sheet answer")
        assert body["records"][0]["question"] == ask


@pytest.mark.parametrize("first,second", [("first ask", "second ask"), ("rate please", "area please")])
def test_the_chat_door_marks_the_turn_before_the_shortcut_can_take_it(monkeypatch, first, second):
    """The predefined shortcut runs ahead of the agent on the chat stream.
    For a catalogue project the turn is marked first, so the shortcut never
    reads the ask; for a plain project it still does."""
    from fastapi.testclient import TestClient

    from app.core import jwt_auth
    from app.core import projects as store
    from app.core import users as users_store
    from app.main import app
    from app.routers import chat as chat_router

    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")
    users_store.ensure_user_exists("cd-admin", role="admin", email="cd-admin@local")
    users_store.ensure_user_exists("cd-user", role="user", email="cd-user@local")
    admin = {"Authorization": f"Bearer {jwt_auth.create_token('cd-admin')}"}
    user = {"Authorization": f"Bearer {jwt_auth.create_token('cd-user')}"}
    on = store.create_project("door project", user_id="cd-user")["id"]
    off = store.create_project("door plain", user_id="cd-user")["id"]
    read = []

    async def understand(prompt, has_documents=False):
        read.append(prompt)
        return {"action": None}

    monkeypatch.setattr(chat_router, "understand_intent", understand)
    monkeypatch.setattr(chat_router, "_message_vetoes_predefined", lambda prompt: False)
    _scripted(monkeypatch, [{"content": "One."}, {"content": "Two."}])
    with TestClient(app) as client:
        assert client.put(f"/v1/admin/projects/{on}/profile", json={"answer_path": "catalogue"},
                          headers=admin).status_code == 200
        client.post("/v1/chat/stream", json={"message": first, "project_id": on,
                                             "conversation_id": f"ws-{on}-1700000000002"}, headers=user)
        assert read == []
        client.post("/v1/chat/stream", json={"message": second, "project_id": off,
                                             "conversation_id": f"ws-{off}-1700000000002"}, headers=user)
        assert read == [second]
