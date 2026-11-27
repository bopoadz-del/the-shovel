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
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
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
    # The model ended the turn with an input still missing: it asked for it.
    assert row["sheet"]["missing"] == [{"for": "s1", "input": "unit_rate", "label": "unit rate", "asked": True}]
    assert row["ended"] == "asked_input"


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
def test_on_the_sheet_with_no_plan_the_question_goes_back_to_the_model(monkeypatch, question):
    """No plan written yet: the calculator's question is the tool result of
    that round, and the model has the next round (code used to end it)."""
    from app.agents.runtime import Agent

    _asking_calculator(monkeypatch, Agent, question)
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("construction_calc", {
            "calculation": "unit_cost_total", "params": {"quantity": 40}}, "c2")]},
        {"content": "Please give the rate per item; the quantity is 40."},
    ])
    events = _run(user_message="cost of the items", conversation_id="ws-p5-1", sheet_mode=True)
    assert len(seen) == 2
    told = [m["content"] for m in seen[1]["messages"] if m["role"] == "tool"]
    assert told and "did not run; needs input (unit_rate)" in told[-1] and question in told[-1]
    assert "Ask the user exactly this" not in told[-1]
    text = _text(events)
    assert "rate per item" in text and question not in text
    assert trace_store.records(_today(), "ws-p5-1")[0]["ended"] == "answer"


def _refusing_workflow(monkeypatch, missing):
    from app.agents.driver import routes

    async def refusing(args, *a, **k):
        return {"ok": False, "missing": list(missing)}

    async def nothing_stated(*a, **k):
        return {}

    monkeypatch.setattr(routes, "run", refusing)
    monkeypatch.setattr(loop, "_inputs_stated_in_ask", nothing_stated)
    monkeypatch.setattr(routes, "input_request", lambda w, r: "CODE REQUEST " + ",".join(r["missing"]))
    return routes.RUN_WORKFLOW


@pytest.mark.parametrize("missing", [["value"], ["start", "duration"]])
def test_on_the_sheet_a_refusing_workflow_does_not_end_the_turn_with_its_request(monkeypatch, missing):
    """Out of rounds with a workflow still refusing: on the sheet the closing
    call answers; code does not write the workflow's request."""
    run_workflow = _refusing_workflow(monkeypatch, missing)
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", "1")
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call(run_workflow, {"workflow": "probe_flow"}, "c1")]},
        {"content": "The workflow still needs " + " and ".join(missing) + "."},
    ])
    events = _run(user_message="make the document", conversation_id="ws-p5w-1", sheet_mode=True)
    assert len(seen) == 2 and seen[-1]["tools"] == []
    text = _text(events)
    assert "CODE REQUEST" not in text and "workflow still needs" in text
    assert trace_store.records(_today(), "ws-p5w-1")[0]["ended"] == "ceiling"


@pytest.mark.parametrize("missing", [["value"], ["start", "duration"]])
def test_off_the_sheet_a_refusing_workflow_still_ends_with_its_request(monkeypatch, missing):
    run_workflow = _refusing_workflow(monkeypatch, missing)
    monkeypatch.setenv("DRIVER_MAX_STEPS", "1")
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call(run_workflow, {"workflow": "probe_flow"}, "c1")]},
        {"content": "never reached"},
    ])
    events = _run(user_message="make the document", conversation_id="ws-p5x-1")
    assert len(seen) == 1
    assert "CODE REQUEST " + ",".join(missing) in _text(events)


# --- the per-request selection ----------------------------------------------

@pytest.mark.parametrize("mode,headers,driver_on,sheet,question_set", [
    ("request", {"x-fork-driver": "sheet", "x-fork-question-set": "frozen"}, True, True, "frozen"),
    ("request", {"x-fork-driver": " Sheet ", "x-fork-question-set": "Held-Back"}, True, True, "held_back"),
    ("request", {"x-fork-driver": "on", "x-fork-question-set": "frozen"}, True, None, ""),
    ("request", {"x-fork-question-set": "teaching"}, False, None, ""),
    ("request", {"x-fork-driver": "sheets"}, False, None, ""),
    ("", {"x-fork-driver": "sheet", "x-fork-question-set": "teaching"}, False, None, ""),
    ("on", {"x-fork-driver": "sheet"}, True, None, ""),
])
def test_a_request_header_selects_the_sheet_for_one_turn(monkeypatch, mode, headers, driver_on, sheet,
                                                          question_set):
    def check():
        if mode:
            monkeypatch.setenv("DRIVER_MODE", mode)
        driver.mark_request(headers)
        assert driver.enabled() is driver_on
        assert driver.on_sheet() is sheet
        assert driver.requested_question_set() == question_set
        # A project without the setting does not take a requested sheet away.
        assert driver.mark_project(False) is bool(sheet)
        assert driver.on_sheet() is bool(sheet)
        assert driver.mark_project(True) is True
    contextvars.copy_context().run(check)


@pytest.mark.parametrize("header,profile_set,recorded", [
    ("teaching", "frozen", "teaching"), ("held-back", None, "held_back"),
    ("", "frozen", "frozen"), ("nonsense", None, "unset")])
def test_the_requests_question_set_is_recorded(monkeypatch, header, profile_set, recorded):
    from app.core import projects

    monkeypatch.setenv("DRIVER_MODE", "request")
    profile = {"question_set": profile_set} if profile_set else {}
    monkeypatch.setattr(projects, "get_project_accessible", lambda pid, uid=None: {"id": pid, "profile": profile})
    _scripted(monkeypatch, [{"content": "Done."}])

    def go():
        driver.mark_request({"x-fork-driver": "sheet", "x-fork-question-set": header})
        _run(user_message="q", conversation_id="ws-p10-1", project_id="p10", sheet_mode=True)
    contextvars.copy_context().run(go)
    assert trace_store.records(_today(), "ws-p10-1")[0]["question_set"] == recorded


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


def _owned_conversation(conv, user, title="carry project"):
    """A stored conversation of ``user`` on a new project of theirs."""
    from app.core import agent_memory
    from app.core import projects as store
    from app.core import users as users_store

    users_store.ensure_user_exists(user, role="user", email=f"{user}@local")
    pid = store.create_project(title, user_id=user)["id"]
    agent_memory.get_or_create_conversation(conv, "sheet-probe", pid, owner_id=user)
    return pid


@pytest.mark.parametrize("value,needed", [(12, "width"), (7.5, "depth")])
def test_the_next_turn_starts_from_the_last_sheet(monkeypatch, value, needed):
    conv = f"carry-c-{needed}"
    pid = _owned_conversation(conv, "carry-u")
    who = {"conversation_id": conv, "user_id": "carry-u", "project_id": pid, "sheet_mode": True}
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": value, "unit": "m"}],
            "missing": [{"for": "s1", "input": needed, "label": needed}]}, "c1")]},
        {"content": "What is the width?"},
    ])
    _run(user_message="first", **who)
    trace_store.forget_held()  # read back from the file, as after a restart
    seen = _scripted(monkeypatch, [{"content": "Thanks."}])
    _run(user_message="second", **who)
    sheet = _sheet_in(seen[0]["messages"])
    assert sheet["givens"][0]["value"] == value and sheet["givens"][0]["source"] == "user"
    assert sheet["missing"][0]["input"] == needed and sheet["missing"][0]["asked"] is True
    other = f"carry-o-{needed}"
    _owned_conversation(other, "carry-u", "carry other")
    seen = _scripted(monkeypatch, [{"content": "New."}])
    _run(user_message="other", **dict(who, conversation_id=other))
    assert _sheet_in(seen[0]["messages"])["givens"] == []


def _one_given(monkeypatch, value):
    return _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "what": "private sum", "value": value, "unit": "currency"}]}, "c1")]},
        {"content": "Noted."},
    ])


@pytest.mark.parametrize("value,held", [(987654, True), (123457, False)])
def test_no_sheet_is_carried_for_an_id_with_no_stored_conversation(monkeypatch, value, held):
    from app.core import agent_memory
    from app.core import projects as store
    from app.core import users as users_store

    users_store.ensure_user_exists("nr-u", role="user", email="nr-u@local")
    pid = store.create_project("no row project", user_id="nr-u")["id"]
    conv = f"nr-c-{value}"
    # A record for this very user, project and id -- but no stored conversation.
    trace_store.save({"schema": "trace/1", "sheet": {"givens": [
        {"id": "g1", "value": value, "unit": "currency", "source": "user"}], "missing": []}},
        conv, user_id="nr-u", project_id=pid)
    assert agent_memory.get_conversation(conv) is None
    assert trace_store.last_sheet(conv, "nr-u", pid)["givens"][0]["value"] == value
    if not held:
        trace_store.forget_held()
    seen = _scripted(monkeypatch, [{"content": "Nothing held."}])
    _run(user_message="what do you hold", conversation_id=conv, user_id="nr-u", project_id=pid, sheet_mode=True)
    assert _sheet_in(seen[0]["messages"])["givens"] == []
    # The sheet turn stored its conversation, owned by its user, so the next
    # turn of the same user and project carries this turn's sheet.
    row = agent_memory.get_conversation(conv)
    assert row is not None and row.get("owner_id") == "nr-u"
    _one_given(monkeypatch, value + 1)
    _run(user_message="hold this", conversation_id=conv, user_id="nr-u", project_id=pid, sheet_mode=True)
    if not held:
        trace_store.forget_held()
    seen = _scripted(monkeypatch, [{"content": "Held."}])
    _run(user_message="again", conversation_id=conv, user_id="nr-u", project_id=pid, sheet_mode=True)
    assert _sheet_in(seen[0]["messages"])["givens"][0]["value"] == value + 1


@pytest.mark.parametrize("value,held", [(987654, True), (123457, False)])
def test_another_users_turn_on_the_same_id_gets_no_sheet(monkeypatch, value, held):
    from app.core import users as users_store

    conv = f"xu-c-{value}"
    pid = _owned_conversation(conv, "xu-owner")
    _one_given(monkeypatch, value)
    _run(user_message="hold this", conversation_id=conv, user_id="xu-owner", project_id=pid, sheet_mode=True)
    if not held:
        trace_store.forget_held()
    users_store.ensure_user_exists("xu-other", role="user", email="xu-other@local")
    # Same id and the same project id: the other user fails the
    # conversation's access check, so nothing is carried.
    seen = _scripted(monkeypatch, [{"content": "Nothing held."}])
    _run(user_message="what do you hold", conversation_id=conv, user_id="xu-other", project_id=pid,
         sheet_mode=True)
    assert _sheet_in(seen[0]["messages"])["givens"] == []
    # The owner still gets their own sheet back.
    if not held:
        trace_store.forget_held()
    seen = _scripted(monkeypatch, [{"content": "Held."}])
    _run(user_message="again", conversation_id=conv, user_id="xu-owner", project_id=pid, sheet_mode=True)
    assert _sheet_in(seen[0]["messages"])["givens"][0]["value"] == value


@pytest.mark.parametrize("value,held", [(987654, True), (123457, False)])
def test_a_sheet_keyed_to_the_user_is_refused_when_the_access_check_fails(monkeypatch, value, held):
    """The key matches -- this user, project and id wrote the sheet -- but
    the stored conversation now belongs to a project the user cannot open:
    the access check refuses the carry."""
    from app.core import users as users_store

    conv = f"ac-c-{value}"
    pid = _owned_conversation(conv, "ac-owner")
    users_store.ensure_user_exists("ac-was-shared", role="user", email="ac-was-shared@local")
    trace_store.save({"schema": "trace/1", "sheet": {"givens": [
        {"id": "g1", "value": value, "unit": "currency", "source": "user"}], "missing": []}},
        conv, user_id="ac-was-shared", project_id=pid)
    assert trace_store.last_sheet(conv, "ac-was-shared", pid)["givens"][0]["value"] == value
    if not held:
        trace_store.forget_held()
    seen = _scripted(monkeypatch, [{"content": "Nothing held."}])
    _run(user_message="what do you hold", conversation_id=conv, user_id="ac-was-shared", project_id=pid,
         sheet_mode=True)
    assert _sheet_in(seen[0]["messages"])["givens"] == []


@pytest.mark.parametrize("empty", [None, ""])
def test_carried_sheet_without_an_id_reads_nothing(monkeypatch, empty):
    from app.core import agent_memory

    def no_read(cid):
        raise AssertionError("a turn with no conversation id read the conversation store")

    monkeypatch.setattr(agent_memory, "get_conversation", no_read)
    assert trace_store.carried_sheet(empty, "cs-u", "cs-p") is None


@pytest.mark.parametrize("value", [41.5, 7])
def test_carried_sheet_needs_a_stored_conversation_and_the_access_check(value):
    from app.core import users as users_store

    conv = f"cs-c-{value}"
    users_store.ensure_user_exists("cs-other", role="user", email="cs-other@local")
    sheet = {"givens": [{"id": "g1", "value": value, "unit": "m", "source": "user"}], "missing": []}
    trace_store.save({"schema": "trace/1", "sheet": sheet}, conv, user_id="cs-u", project_id="cs-p")
    # No id, or no stored conversation: nothing, though a record matches.
    assert trace_store.carried_sheet(None, "cs-u", "cs-p") is None
    assert trace_store.carried_sheet("", "cs-u", "cs-p") is None
    assert trace_store.carried_sheet(conv, "cs-u", "cs-p") is None
    pid = _owned_conversation(conv, "cs-u")
    trace_store.save({"schema": "trace/1", "sheet": sheet}, conv, user_id="cs-u", project_id=pid)
    assert trace_store.carried_sheet(conv, "cs-u", pid) == sheet
    trace_store.save({"schema": "trace/1", "sheet": sheet}, conv, user_id="cs-other", project_id=pid)
    assert trace_store.carried_sheet(conv, "cs-other", pid) is None  # refused by the access check


def _user(uid, role="user"):
    from app.core import jwt_auth
    from app.core import users as users_store

    users_store.ensure_user_exists(uid, role=role, email=f"{uid}@local")
    return {"Authorization": f"Bearer {jwt_auth.create_token(uid)}"}


def _sheets_seen(seen):
    return [_sheet_in(s["messages"]) for s in seen
            if any(m["role"] == "system" and m["content"].startswith("Working sheet") for m in s["messages"])]


@pytest.mark.parametrize("value,held", [(987654, True), (123457, False)])
def test_public_route_two_users_one_id_never_share_a_sheet(monkeypatch, value, held):
    from fastapi.testclient import TestClient

    from app.core import projects as store
    from app.main import app

    admin, ua, ub = _user("pu-admin", "admin"), _user("pu-a"), _user("pu-b")
    pa = store.create_project("A project", user_id="pu-a")["id"]
    pb = store.create_project("B project", user_id="pu-b")["id"]
    conv = f"pu-conv-{value}"
    seen = _one_given(monkeypatch, value)
    with TestClient(app) as client:
        for p in (pa, pb):
            assert client.put(f"/v1/admin/projects/{p}/profile", json={"answer_path": "catalogue"},
                              headers=admin).status_code == 200
        client.post("/v1/agents/project-assistant/chat/stream",
                    json={"message": "hold this", "project_id": pa, "conversation_id": conv}, headers=ua)
        assert len(seen) == 2
        if not held:
            trace_store.forget_held()
        after = _scripted(monkeypatch, [{"content": "B answer."}])
        r = client.post("/v1/agents/project-assistant/chat/stream",
                        json={"message": "what do you hold", "project_id": pb, "conversation_id": conv},
                        headers=ub)
    # B is refused at the door or runs with an empty sheet: never A's figure.
    assert r.status_code in (200, 404)
    assert all(sh["givens"] == [] for sh in _sheets_seen(after))
    assert str(value) not in r.text


@pytest.mark.parametrize("value,held", [(987654, True), (123457, False)])
def test_public_route_one_user_two_projects_one_id_never_share_a_sheet(monkeypatch, value, held):
    from fastapi.testclient import TestClient

    from app.core import projects as store
    from app.main import app

    admin, user = _user("pp-admin", "admin"), _user("pp-u")
    p1 = store.create_project("first project", user_id="pp-u")["id"]
    p2 = store.create_project("second project", user_id="pp-u")["id"]
    conv = f"pp-conv-{value}"
    seen = _one_given(monkeypatch, value)
    with TestClient(app) as client:
        for p in (p1, p2):
            assert client.put(f"/v1/admin/projects/{p}/profile", json={"answer_path": "catalogue"},
                              headers=admin).status_code == 200
        client.post("/v1/agents/project-assistant/chat/stream",
                    json={"message": "hold this", "project_id": p1, "conversation_id": conv}, headers=user)
        assert len(seen) == 2
        if not held:
            trace_store.forget_held()
        other = _scripted(monkeypatch, [{"content": "Second project answer."}])
        r = client.post("/v1/agents/project-assistant/chat/stream",
                        json={"message": "what do you hold", "project_id": p2, "conversation_id": conv},
                        headers=user)
        assert r.status_code == 200 and len(other) == 1
        assert _sheets_seen(other)[0]["givens"] == []
        # Back on the first project, the same user gets the first sheet.
        if not held:
            trace_store.forget_held()
        same = _scripted(monkeypatch, [{"content": "First project answer."}])
        client.post("/v1/agents/project-assistant/chat/stream",
                    json={"message": "and now", "project_id": p1, "conversation_id": conv}, headers=user)
        assert _sheets_seen(same)[0]["givens"][0]["value"] == value


@pytest.mark.parametrize("needed,label", [("width", "trench width"), ("unit_rate", "unit rate")])
def test_public_route_an_input_the_model_asked_for_is_marked_asked_next_turn(monkeypatch, needed, label):
    """Two turns through the door: the model records a missing input and ends
    its turn asking; the next turn's sheet carries it marked asked."""
    from fastapi.testclient import TestClient

    from app.core import projects as store
    from app.main import app

    admin, user = _user("ak-admin", "admin"), _user("ak-u")
    pid = store.create_project("asking project", user_id="ak-u")["id"]
    conv = f"ws-{pid}-1700000000020"
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "missing": [{"for": "s1", "input": needed, "label": label}]}, "c1")]},
        {"content": f"What is the {label}?"},
    ])
    with TestClient(app) as client:
        assert client.put(f"/v1/admin/projects/{pid}/profile", json={"answer_path": "catalogue"},
                          headers=admin).status_code == 200
        client.post("/v1/agents/project-assistant/chat/stream",
                    json={"message": "work it out", "project_id": pid, "conversation_id": conv}, headers=user)
        assert len(seen) == 2
        nxt = _scripted(monkeypatch, [{"content": "Thanks."}])
        client.post("/v1/agents/project-assistant/chat/stream",
                    json={"message": "go on", "project_id": pid, "conversation_id": conv}, headers=user)
    assert _sheets_seen(nxt)[0]["missing"] == [{"for": "s1", "input": needed, "label": label, "asked": True}]
    first, second = trace_store.records(_today(), conv)
    assert first["ended"] == "asked_input" and first["sheet"]["missing"][0]["asked"] is True
    # The second turn's missing input was already asked: it is an answer.
    assert second["ended"] == "answer"


@pytest.mark.parametrize("needed", ["width", "depth"])
def test_a_turn_with_nothing_missing_or_cut_by_the_ceiling_marks_nothing_asked(monkeypatch, needed):
    _scripted(monkeypatch, [{"content": "A plain answer."}])
    _run(user_message="q", conversation_id=f"na-1-{needed}", sheet_mode=True)
    assert trace_store.records(_today(), f"na-1-{needed}")[0]["ended"] == "answer"
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", "1")
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {"missing": [{"for": "s1", "input": needed}]}, "c1")]},
        {"content": "Stopped."},
    ])
    _run(user_message="q", conversation_id=f"na-2-{needed}", sheet_mode=True)
    row = trace_store.records(_today(), f"na-2-{needed}")[0]
    assert row["ended"] == "ceiling" and row["sheet"]["missing"][0]["asked"] is False


@pytest.mark.parametrize("pause,seconds,rounds,tool_rounds", [
    (0.6, "1", "40", 2),   # 0.6 s spent, then 1.2 s: the third round never starts
    (0.4, "1", "40", 3),   # 0.4, 0.8, 1.2 s
    (3.0, "10", "40", 4),  # 3, 6, 9, 12 s: the fifth round never starts
    (2.5, "10", "40", 5),  # exactly 10.0 s spent is not past the ceiling: a fifth round
    (0.1, "10", "5", 5),   # the seconds are not spent: the rounds run out first
])
def test_the_seconds_ceiling_ends_a_turn_before_its_rounds_run_out(monkeypatch, pause, seconds, rounds,
                                                                  tool_rounds):
    import time as real_time
    import types

    monkeypatch.setenv("DRIVER_SHEET_MAX_SECONDS", seconds)
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", rounds)
    clock = {"now": 0.0}
    # The loop's own clock: each model round spends ``pause`` seconds.
    monkeypatch.setattr(loop, "time", types.SimpleNamespace(monotonic=lambda: clock["now"], time=real_time.time))
    calls = []

    async def slow(agent, messages, offered, model="", timeout=120.0):
        calls.append(len(offered))
        clock["now"] += pause
        return {"status": "success", "message": {"content": "", "tool_calls": [
            _call("sheet_update", {"givens": [{"id": "g1", "value": len(calls), "unit": "m"}]}, f"c{len(calls)}")]}}

    monkeypatch.setattr(llm, "call", slow)
    conv = f"sc-{pause}-{rounds}"
    _run(user_message="q", conversation_id=conv, sheet_mode=True)
    row = trace_store.records(_today(), conv)[0]
    assert row["ended"] == "ceiling"
    # The tool rounds, then the closing call with no tools.
    assert len([n for n in calls if n]) == tool_rounds and calls[-1] == 0


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


@pytest.mark.parametrize("quantity,qset", [(40, "frozen"), (12, "teaching")])
def test_the_public_route_takes_the_sheet_on_request_and_the_question_goes_back(monkeypatch, quantity, qset):
    """Through the door, DRIVER_MODE=request, a project WITHOUT the setting:
    the turn with the sheet header takes the sheet; its calculator lacks an
    input and the model still reaches a second round; it writes a trace
    record under the header's question set. The plain turn writes none."""
    from fastapi.testclient import TestClient

    from app.agents.runtime import Agent
    from app.core import jwt_auth
    from app.core import projects as store
    from app.core import users as users_store
    from app.main import app

    monkeypatch.setenv("DRIVER_MODE", "request")
    _asking_calculator(monkeypatch, Agent)
    users_store.ensure_user_exists("rq-user", role="user", email="rq-user@local")
    user = {"Authorization": f"Bearer {jwt_auth.create_token('rq-user')}"}
    pid = store.create_project("request project", user_id="rq-user")["id"]
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("construction_calc", {
            "calculation": "unit_cost_total", "params": {"quantity": quantity}}, "c1")]},
        {"content": f"The {quantity} items need a unit rate before the cost can be worked."},
    ])
    conv, plain = f"ws-{pid}-1700000000010", f"ws-{pid}-1700000000011"
    with TestClient(app) as client:
        text = client.post("/v1/agents/project-assistant/chat/stream",
                           json={"message": "cost of the items", "project_id": pid, "conversation_id": conv},
                           headers={**user, "X-Fork-Driver": "sheet", "X-Fork-Question-Set": qset}).text
        assert '"mode": "driver"' in text and '"trace_record"' in text
        assert len(seen) == 2 and "sheet_update" in seen[0]["tools"]
        assert "need a unit rate" in text and "What is the unit rate?" not in text

        async def no_driver(*a, **k):
            raise AssertionError("the driver ran for a plain turn")
            yield  # pragma: no cover

        monkeypatch.setattr(loop, "stream", no_driver)
        text = client.post("/v1/agents/project-assistant/chat/stream",
                           json={"message": "cost of the items", "project_id": pid, "conversation_id": plain},
                           headers=user).text
        assert '"mode": "driver"' not in text
    rows = trace_store.records(_today(), conv)
    assert len(rows) == 1 and rows[0]["question_set"] == qset and rows[0]["ended"] == "answer"
    assert trace_store.records(_today(), plain) == []


# --- a result code wrote on the sheet reaches the answer ------------------------

@pytest.mark.parametrize("a,ua,op,b,ub,reply,figure", [
    (55, "m", "add", 2, "m", "The total is 57 m.", "57 m"),
    (550, "m", "add", 2.75, "m", "The total is 552.75 m.", "552.75 m"),
    (12, "m", "multiply", 4, "m", "The area is 48 m*m.", "48 m*m"),
    (100, "m3", "divide", 4, "-", "The volume is 25 m3.", "25 m3"),
    (12.5, "m2", "multiply", 3, "-", "The area is 37.5 m2.", "37.5 m2"),
])
def test_a_figure_code_wrote_on_the_sheet_is_shown_and_listed_in_the_trace(monkeypatch, a, ua, op, b, ub,
                                                                             reply, figure):
    conv = f"fig-{op}-{a}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": a, "unit": ua}, {"id": "g2", "value": b, "unit": ub}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": op, "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": reply},
    ])
    # The user's own message carries both givens: the result rests on them.
    events = _run(user_message=f"work it out from {a} {ua} and {b}", conversation_id=conv, sheet_mode=True)
    shown = _text(events)
    assert figure in shown and "incomplete" not in shown
    row = trace_store.records(_today(), conv)[0]
    credited = [f for f in row["figures"] if f.get("step") == "s1"]
    assert credited and credited[0]["source"] == "calculator"
    assert credited[0]["formula"] == "formula:arithmetic_step"
    assert credited[0]["unit"] == row["sheet"]["steps"][0]["result"]["unit"]
    assert row["figures_from_tools"] is True and row["ended"] == "answer"


@pytest.mark.parametrize("a,b,wrong", [(55, 2, "58"), (12, 4, "49")])
def test_a_figure_in_a_unit_that_no_step_produced_is_still_removed(monkeypatch, a, b, wrong):
    """The gate stays on: only the value code wrote is credited."""
    conv = f"nofig-{a}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The total is {wrong} m."},
    ])
    shown = _text(_run(user_message="work it out", conversation_id=conv, sheet_mode=True))
    assert f"{wrong} m" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert not [f for f in row["figures"] if wrong in str(f.get("figure")) and f.get("source")]


# --- an input counts as asked only when its step has no result ------------------

def _found_then_ran(a, b):
    """Round 1 records b as missing; round 2 finds it and plans s1 without
    rewriting missing; round 3 runs s1; round 4 answers with the total."""
    return [
        {"content": "", "tool_calls": [_call("sheet_update", {"missing": [{"for": "s1", "input": "b"}]}, "c1")]},
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c2")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c3")]},
        {"content": f"The total is {a + b} m."},
    ]


@pytest.mark.parametrize("a,b", [(55, 77), (12, 30)])
def test_a_turn_whose_missing_step_ran_ends_as_an_answer_and_carries_no_asked_input(monkeypatch, a, b):
    conv = f"ran-c-{a}"
    pid = _owned_conversation(conv, "ran-u")
    who = {"conversation_id": conv, "user_id": "ran-u", "project_id": pid, "sheet_mode": True}
    _scripted(monkeypatch, _found_then_ran(a, b))
    shown = _text(_run(user_message=f"add {a} m and {b} m", **who))
    assert f"{a + b} m" in shown
    row = trace_store.records(_today(), conv)[0]
    assert row["ended"] == "answer" and row["sheet"]["done"] is True and row["sheet"]["missing"] == []
    trace_store.forget_held()  # read back from the file, as after a restart
    seen = _scripted(monkeypatch, [{"content": "Thanks."}])
    _run(user_message="second", **who)
    assert _sheet_in(seen[0]["messages"])["missing"] == []


@pytest.mark.parametrize("a,b", [(55, 77), (12, 30)])
def test_an_input_for_a_step_without_a_result_is_still_asked_when_another_step_ran(monkeypatch, a, b):
    conv = f"ran-o-{a}"
    replies = _found_then_ran(a, b)
    replies[0] = {"content": "", "tool_calls": [_call("sheet_update", {
        "missing": [{"for": "s1", "input": "b"}, {"for": "s2", "input": "rate"}]}, "c1")]}
    replies[-1] = {"content": f"The total is {a + b} m. What is the rate?"}
    _scripted(monkeypatch, replies)
    _run(user_message="first", conversation_id=conv, sheet_mode=True)
    row = trace_store.records(_today(), conv)[0]
    assert row["ended"] == "asked_input"
    assert row["sheet"]["missing"] == [{"for": "s2", "input": "rate", "label": "rate", "asked": True}]


# --- a result is credited only when every number under it was given to us -------

@pytest.mark.parametrize("value,unit", [(4321, "m3"), (987.5, "m2")])
def test_a_number_the_model_wrote_in_the_plan_is_not_made_a_code_figure_by_a_step(monkeypatch, value, unit):
    """The model types a number as a plan input and adds zero: the step ran,
    but its result rests on a number nobody gave, so it gets no credit."""
    conv = f"own-plan-{value}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": value, "b": 0, "a_unit": unit, "b_unit": unit}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The quantity is {value} {unit}."},
    ])
    shown = _text(_run(user_message="how much is there?", conversation_id=conv, sheet_mode=True))
    assert f"{value} {unit}" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert row["sheet"]["steps"][0]["status"] == "done"  # the step did run
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("value,unit", [(4321, "m3"), (987.5, "m2")])
def test_a_given_nobody_gave_is_not_made_a_code_figure_by_a_step(monkeypatch, value, unit):
    """The same through a given the model wrote with source "user": the
    user's message does not carry it, so the result gets no credit."""
    conv = f"own-given-{value}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": value, "unit": unit, "source": "user"},
                       {"id": "g2", "value": 2, "unit": "-", "source": "user"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The quantity is {value * 2:g} {unit}."},
    ])
    shown = _text(_run(user_message="double it, 2 times", conversation_id=conv, sheet_mode=True))
    assert f"{value * 2:g} {unit}" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("a,b", [(55, 77), (12, 30)])
def test_a_chain_resting_on_one_number_nobody_gave_is_not_credited_at_any_step(monkeypatch, a, b):
    """s2 reads s1; s1 reads a given the user wrote and a number the model
    wrote. Neither result is a figure from code."""
    conv = f"own-chain-{a}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": a, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": b, "b_unit": "m"}},
                     {"id": "s2", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "s1", "b": "g1"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s2"}, "c3")]},
        {"content": f"The total is {a + b + a} m."},
    ])
    shown = _text(_run(user_message=f"the run is {a} m", conversation_id=conv, sheet_mode=True))
    assert f"{a + b + a} m" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert not [f for f in row["figures"] if f.get("step") in ("s1", "s2")]


@pytest.mark.parametrize("a,b", [(55, 77), (12, 30)])
def test_a_missing_input_that_names_no_step_is_refused_and_the_turn_ends_as_an_answer(monkeypatch, a, b):
    conv = f"nofor-{a}"
    pid = _owned_conversation(conv, "nofor-u")
    who = {"conversation_id": conv, "user_id": "nofor-u", "project_id": pid, "sheet_mode": True}
    replies = _found_then_ran(a, b)
    replies[0] = {"content": "", "tool_calls": [_call("sheet_update", {
        "missing": [{"input": "b", "label": "second"}]}, "c1")]}
    seen = _scripted(monkeypatch, replies)
    shown = _text(_run(user_message=f"add {a} m and {b} m", **who))
    assert f"{a + b} m" in shown
    refused = [m["content"] for m in seen[1]["messages"] if m["role"] == "tool"][-1]
    assert "must say which plan step it is for" in refused
    row = trace_store.records(_today(), conv)[0]
    assert row["ended"] == "answer" and row["sheet"]["done"] is True and row["sheet"]["missing"] == []
