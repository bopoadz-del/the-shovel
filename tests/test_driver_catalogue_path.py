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
    # The end event carries the final sheet and the ending, the same as the
    # record holds, for the answer's "how this was worked out" panel.
    assert end["sheet"] == row["sheet"] and end["ended"] == row["ended"] == "answer"


def _strict_json(frame):
    """Parse as the browser's JSON.parse does: NaN / Infinity are not JSON."""
    def refuse(constant):
        raise ValueError(f"not JSON: {constant}")
    return json.loads(frame, parse_constant=refuse)


@pytest.mark.parametrize("bad,good,total", [
    (float("nan"), 1.05, 42.0),
    ({"value": float("inf"), "unit": "-"}, {"value": 1.1, "unit": "-"}, 44.0),
])
def test_a_non_finite_plan_literal_is_refused_to_the_model_and_the_end_frame_stays_json(monkeypatch, bad, good,
                                                                                         total):
    def update(b):  # a refused update leaves the sheet unchanged, so the retry is whole
        return {"givens": [{"id": "g1", "what": "spoil", "value": 40, "unit": "m3", "source": "user"}],
                "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                          "inputs": {"operator": "multiply", "a": "g1", "b": b}}]}
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", update(bad), "c1")]},
        {"content": "", "tool_calls": [_call("sheet_update", update(good), "c2")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c3")]},
        {"content": "Done."},
    ])
    events = _run(user_message="spoil 40 m3 times a factor", conversation_id=f"ws-nf-{total:g}",
                  project_id=None, sheet_mode=True)
    # The model reads the refusal as its tool result, and can correct it.
    refused = [m["content"] for m in seen[1]["messages"] if m["role"] == "tool"]
    assert any("not finite" in c for c in refused), refused
    assert _sheet_in(seen[1]["messages"])["plan"] == []  # the bad step never reached the sheet
    end = [e for e in events if e.get("type") == "end"][0]
    # Serialized as the route serializes a frame (app/routers/agents.py), the
    # end frame parses as strict JSON and carries the corrected plan.
    frame = _strict_json(json.dumps(end, default=str))
    assert frame["sheet"]["plan"][0]["inputs"]["b"] == good
    assert frame["sheet"]["steps"][0]["result"]["value"] == pytest.approx(total)


@pytest.mark.parametrize("b,total", [(1.05, 42.0), ({"value": 2.5, "unit": "-"}, 100.0)])
def test_a_finite_plan_literal_is_not_refused_and_the_end_frame_is_json(monkeypatch, b, total):
    seen = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "what": "spoil", "value": 40, "unit": "m3", "source": "user"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "g1", "b": b}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": "Done."},
    ])
    events = _run(user_message="spoil 40 m3 times a factor", conversation_id=f"ws-fin-{total:g}",
                  project_id=None, sheet_mode=True)
    assert not any("not finite" in m["content"] for m in seen[1]["messages"] if m["role"] == "tool")
    end = _strict_json(json.dumps([e for e in events if e.get("type") == "end"][0], default=str))
    assert end["sheet"]["steps"][0]["result"]["value"] == pytest.approx(total)


@pytest.mark.parametrize("reply", ["A plain answer.", "Another reply here."])
def test_without_the_setting_no_sheet_and_no_record(monkeypatch, reply):
    seen = _scripted(monkeypatch, [{"content": reply}])
    events = _run(user_message="hello", conversation_id="ws-p2-1")
    assert reply in _text(events)
    assert "sheet_update" not in seen[0]["tools"]
    assert not any(m["content"].startswith("Working sheet") for m in seen[0]["messages"] if m["role"] == "system")
    end = [e for e in events if e.get("type") == "end"][0]
    assert "trace_record" not in end and "sheet" not in end and "ended" not in end
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
    end = [e for e in events if e.get("type") == "end"][0]
    assert end["trace_record"]
    # The panel reads the ending from the end event: the ceiling, not "answer".
    assert end["ended"] == row["ended"] == "ceiling"


@pytest.mark.parametrize("rounds", [2, 3])
def test_every_round_tells_the_model_which_round_of_its_budget_it_is(monkeypatch, rounds):
    """The budget is stated in the sheet note each round, never enforced by it:
    the turn still runs to the same ceiling and the sheet still parses."""
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", str(rounds))
    replies = [{"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, f"c{i}")]} for i in range(rounds)]
    seen = _scripted(monkeypatch, replies + [{"content": "Stopped."}])
    _run(user_message="q", conversation_id=f"ws-p7-{rounds}", sheet_mode=True)
    notes = [next(m["content"] for m in s["messages"]
                  if m["role"] == "system" and m["content"].startswith("Working sheet"))
             for s in seen[:rounds]]
    for i, note in enumerate(notes, start=1):
        rules = note.split("\n", 1)[0]
        assert f"round {i} of {rounds} " in rules
        assert f"no tool round starts after round {rounds} or once the seconds" in rules
        assert all(f"round {j} of" not in rules for j in range(1, rounds + 2) if j != i)
    assert all(isinstance(_sheet_in(s["messages"]), dict) for s in seen[:rounds])
    assert len(seen) == rounds + 1  # the ceiling did not move


def test_the_default_path_gets_no_round_line(monkeypatch):
    """Negative: a turn off the sheet path carries no sheet note and so no budget line."""
    seen = _scripted(monkeypatch, [{"content": "Done."}])
    _run(user_message="q", conversation_id="ws-p8-1")
    assert not any("for this turn; tool calls end after round" in m["content"]
                   for m in seen[0]["messages"] if isinstance(m.get("content"), str))


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


@pytest.mark.parametrize("a,b", [(40, 67), (12.5, 30)])
def test_a_given_the_user_wrote_in_an_earlier_turn_still_backs_a_result_in_the_next(monkeypatch, a, b):
    """Turn 1 carries both numbers; turn 2's message carries neither. The
    givens were checked when they were written, so turn 2's result is shown."""
    conv = f"carry-given-{a}"
    pid = _owned_conversation(conv, "cg-u")
    who = {"conversation_id": conv, "user_id": "cg-u", "project_id": pid, "sheet_mode": True}
    plan = {"givens": [{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}
    _scripted(monkeypatch, [{"content": "", "tool_calls": [_call("sheet_update", plan, "c1")]},
                            {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
                            {"content": f"The total is {a + b} m."}])
    assert f"{a + b} m" in _text(_run(user_message=f"add {a} m and {b} m", **who))
    first = trace_store.records(_today(), conv)[0]
    assert [g.get("checked") for g in first["sheet"]["givens"]] == ["user", "user"]
    trace_store.forget_held()  # read back from the file, as after a restart
    again = {"plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                       "inputs": {"operator": "multiply", "a": "g1", "b": "g2"}}]}
    _scripted(monkeypatch, [{"content": "", "tool_calls": [_call("sheet_update", again, "c1")]},
                            {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
                            {"content": f"The area is {a * b:g} m*m."}])
    shown = _text(_run(user_message="now multiply them instead", **who))
    assert f"{a * b:g} m*m" in shown and "incomplete" not in shown
    # A reply that quotes the earlier turn's own figure keeps it, as the user's.
    _scripted(monkeypatch, [{"content": "", "tool_calls": [_call("sheet_update", again, "c1")]},
                            {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
                            {"content": f"The length you gave is {a} m, so the area is {a * b:g} m*m."}])
    shown = _text(_run(user_message="and once more", **who))
    assert f"{a} m" in shown and f"{a * b:g} m*m" in shown
    quoted = [f for f in trace_store.records(_today(), conv)[-1]["figures"] if f.get("figure") == f"{a} m"]
    assert quoted and quoted[0]["source"] == "user_input"


@pytest.mark.parametrize("length,width,depth,volume", [(43, 10, 2, 860), (12.5, 8, 3, 300)])
def test_numbers_written_in_the_plan_with_their_unit_are_not_made_a_code_figure(monkeypatch, length, width,
                                                                               depth, volume):
    """The model writes the quantities as text with a unit ("43 m"): the
    formula reads them, and so does the check."""
    conv = f"own-text-{length}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                      "inputs": {"length_m": f"{length} m", "width_m": f"{width} m",
                                 "depth_m": f"{depth} m"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The excavation is {volume} m3."},
    ])
    shown = _text(_run(user_message="how much is there?", conversation_id=conv, sheet_mode=True))
    assert f"{volume} m3" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("value,unit", [(4321, "m3"), (987.5, "m2")])
def test_a_number_written_in_the_plan_as_a_value_object_is_not_made_a_code_figure(monkeypatch, value, unit):
    conv = f"own-object-{value}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": {"value": value, "unit": unit},
                                 "b": {"value": 0, "unit": unit}}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The quantity is {value} {unit}."},
    ])
    shown = _text(_run(user_message="how much is there?", conversation_id=conv, sheet_mode=True))
    assert f"{value} {unit}" not in shown
    assert not [f for f in trace_store.records(_today(), conv)[0]["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("level,rise,new", [(-43, 10, "-33"), (-7, 2.5, "-4.5")])
def test_a_negative_number_the_user_wrote_is_a_given_and_its_result_is_shown(monkeypatch, level, rise, new):
    conv = f"neg-{rise}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": level, "unit": "m"}, {"id": "g2", "value": rise, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The new level is {new} m."},
    ])
    shown = _text(_run(user_message=f"The formation level is {level} m. Raise it by {rise} m: what is the new level?",
                       conversation_id=conv, sheet_mode=True))
    assert f"{new} m" in shown and "incomplete" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert [g.get("checked") for g in row["sheet"]["givens"]] == ["user", "user"]


@pytest.mark.parametrize("depth,volume", [(".5 m", 30), ("+.75 m", 45)])
def test_a_depth_nobody_gave_written_without_its_zero_is_not_made_a_code_figure(monkeypatch, depth, volume):
    """The user gives the length and the width only. The model writes the
    depth in the plan as ".5 m": the formula reads it, and so does the
    check, so the volume is the model's own number passed through a formula."""
    conv = f"own-dot-{volume}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                      "inputs": {"length_m": "40 m", "width_m": "1.5 m", "depth_m": depth}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The excavation is {volume} m3."},
    ])
    shown = _text(_run(user_message="A trench 40 m long and 1.5 m wide. How much is excavated?",
                       conversation_id=conv, sheet_mode=True))
    assert f"{volume} m3" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert [st["result"]["value"] for st in row["sheet"]["steps"]] == [float(volume)]    # the step did run
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("level,rise,wrong", [(-43, 10, "53"), (-7, 2.5, "9.5")])
def test_a_given_with_its_sign_turned_is_not_the_users_and_its_result_is_not_shown(monkeypatch, level, rise, wrong):
    """The user writes a level below datum; the model writes the given
    without the minus. Nobody gave that number, so the sum is not credited."""
    conv = f"flip-{rise}"
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": -level, "unit": "m"}, {"id": "g2", "value": rise, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The new level is {wrong} m."},
    ])
    shown = _text(_run(user_message=f"The formation level is {level} m. It rises by {rise} m. What is the new level?",
                       conversation_id=conv, sheet_mode=True))
    assert f"{wrong} m" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert [g.get("checked") for g in row["sheet"]["givens"]] == [None, "user"]
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("a,b", [(55, 2), (120, 35)])
def test_a_carried_given_keeps_its_credit_when_no_history_is_sent(monkeypatch, a, b):
    """Turn 2 arrives with no history at all: the given written in turn 1 was
    checked then, so the result resting on it is still shown."""
    conv = f"nohist-{a}"
    pid = _owned_conversation(conv, "nh-u")
    who = {"conversation_id": conv, "user_id": "nh-u", "project_id": pid, "sheet_mode": True}
    _scripted(monkeypatch, [{"content": "", "tool_calls": [_call("sheet_update", {
        "givens": [{"id": "g1", "value": a, "unit": "m"}]}, "c1")]}, {"content": "Noted."}])
    _run(user_message=f"The wall is {a} m long.", **who)
    trace_store.forget_held()
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g2", "value": b, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The total length is {a + b} m."}])
    shown = _text(_run(user_message=f"Add the {b} m return. What is the total?", history=[], **who))
    assert f"{a + b} m" in shown and "incomplete" not in shown


# --- a number the user gave, with its unit changed by the model, is not the user's ---

_TRENCH_ASK = "A trench is 12 m long, 3 m wide and 2 m deep. What volume is excavated?"


def _volume_turn(monkeypatch, conv, update, said):
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", update, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": said},
    ])
    shown = _text(_run(user_message=_TRENCH_ASK, conversation_id=conv, sheet_mode=True))
    return shown, trace_store.records(_today(), conv)[0]


@pytest.mark.parametrize("length,ran", [("12 mm", 0.072), ("12 cm", 0.72), ({"value": 12, "unit": "mm"}, 0.072),
                                         ("12 ft", 21.946)])
def test_a_plan_literal_with_the_users_number_and_another_unit_is_not_credited(monkeypatch, length, ran):
    """The user wrote 12 m. The model plans the length as "12 mm": the step
    runs on 0.012 m. That quantity is the model's, so its result is neither
    shown nor credited to the calculator."""
    shown, row = _volume_turn(monkeypatch, f"unit-lit-{ran}", {
        "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                  "inputs": {"length_m": length, "width_m": 3, "depth_m": 2}}]},
        f"The excavation is {ran:g} m3.")
    assert [st["result"]["value"] for st in row["sheet"]["steps"]] == [pytest.approx(ran)]   # the step did run
    assert f"{ran:g} m3" not in shown
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("value,unit,ran", [(12, "mm", 0.072), (12, "cm", 0.72), (12000, "m", 72000.0)])
def test_a_given_with_the_users_number_and_another_unit_is_not_the_users(monkeypatch, value, unit, ran):
    shown, row = _volume_turn(monkeypatch, f"unit-giv-{ran}", {
        "givens": [{"id": "g1", "value": value, "unit": unit}, {"id": "g2", "value": 3, "unit": "m"},
                   {"id": "g3", "value": 2, "unit": "m"}],
        "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                  "inputs": {"length_m": "g1", "width_m": "g2", "depth_m": "g3"}}]},
        f"The excavation is {ran:g} m3.")
    assert [g.get("checked") for g in row["sheet"]["givens"]] == [None, "user", "user"]
    assert f"{ran:g} m3" not in shown
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("value,unit", [(12, "m"), (12000, "mm"), (0.012, "km")])
def test_the_users_quantity_in_any_unit_of_its_kind_is_the_users_and_is_credited(monkeypatch, value, unit):
    shown, row = _volume_turn(monkeypatch, f"unit-ok-{unit}", {
        "givens": [{"id": "g1", "value": value, "unit": unit}, {"id": "g2", "value": 3, "unit": "m"},
                   {"id": "g3", "value": 2, "unit": "m"}],
        "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                  "inputs": {"length_m": "g1", "width_m": "g2", "depth_m": "g3"}}]},
        "The excavation is 72 m3.")
    assert [g.get("checked") for g in row["sheet"]["givens"]] == ["user", "user", "user"]
    assert "72 m3" in shown and "incomplete" not in shown
    assert [f["step"] for f in row["figures"] if f.get("step")][:1] == ["s1"]


@pytest.mark.parametrize("length", ["12 m", "12000 mm", {"value": 12, "unit": "m"}])
def test_the_users_quantity_written_in_the_plan_is_credited(monkeypatch, length):
    shown, row = _volume_turn(monkeypatch, f"unit-lit-ok-{length}", {
        "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                  "inputs": {"length_m": length, "width_m": "3 m", "depth_m": "2 m"}}]},
        "The excavation is 72 m3.")
    assert "72 m3" in shown and "incomplete" not in shown
    assert [f["step"] for f in row["figures"] if f.get("step")][:1] == ["s1"]


# --- a figure from a workflow's answer keeps its sign for the given check ---

def _level_workflow(monkeypatch, answer):
    from app.agents.driver import routes

    async def ran(args, *a, **k):
        return {"ok": True, "workflow": "survey_levels", "answer": answer}

    async def nothing_stated(*a, **k):
        return {}

    monkeypatch.setattr(routes, "run", ran)
    monkeypatch.setattr(loop, "_inputs_stated_in_ask", nothing_stated)
    return routes.RUN_WORKFLOW


@pytest.mark.parametrize("level,rise,wrong", [(-43, 10, "53"), (-7, 2.5, "9.5")])
def test_a_workflow_figure_with_its_sign_turned_is_not_credited(monkeypatch, level, rise, wrong):
    """The workflow answers a level below datum. The model writes it as a
    given without the minus: nobody gave that number."""
    conv = f"wf-flip-{rise}"
    run_workflow = _level_workflow(monkeypatch, f"The formation level is {level} m.")
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call(run_workflow, {"workflow": "survey_levels"}, "c0")]},
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": -level, "unit": "m"}, {"id": "g2", "value": rise, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The new level is {wrong} m."},
    ])
    shown = _text(_run(user_message=f"Read the formation level and raise it by {rise} m. What is the new level?",
                       conversation_id=conv, sheet_mode=True))
    assert f"{wrong} m" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert [g.get("checked") for g in row["sheet"]["givens"]] == [None, "user"]
    assert not [f for f in row["figures"] if f.get("step") == "s1"]


@pytest.mark.parametrize("level,rise,new", [(-43, 10, "-33"), (-7, 2.5, "-4.5")])
def test_a_workflow_figure_with_its_sign_kept_is_credited_with_its_step(monkeypatch, level, rise, new):
    conv = f"wf-kept-{rise}"
    run_workflow = _level_workflow(monkeypatch, f"The formation level is {level} m.")
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call(run_workflow, {"workflow": "survey_levels"}, "c0")]},
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": level, "unit": "m"}, {"id": "g2", "value": rise, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": f"The new level is {new} m."},
    ])
    shown = _text(_run(user_message=f"Read the formation level and raise it by {rise} m. What is the new level?",
                       conversation_id=conv, sheet_mode=True))
    assert f"{new} m" in shown and "incomplete" not in shown
    row = trace_store.records(_today(), conv)[0]
    assert [g.get("checked") for g in row["sheet"]["givens"]] == ["calculator", "user"]
    assert [f for f in row["figures"] if f.get("step") == "s1"]


# --- a given written before its passage is fetched is checked again before the step runs ---

@pytest.mark.parametrize("depth,add,total", [(7.25, 2, "9.25"), (13.5, 2, "15.5")])
def test_a_given_written_before_its_passage_is_marked_when_the_step_runs(monkeypatch, depth, add, total):
    from app.agents.runtime import Agent

    conv = f"late-passage-{depth}"

    async def found(self, call, api_key=None, project_id=None, conversation_id=None, **kw):
        return {"ok": True, "result": {"results": [{"text": f"Pile cap depth is {depth} m.",
                                                     "doc_name": "drawing.pdf", "page": 4}]}}

    monkeypatch.setattr(Agent, "_run_tool_call", found)
    _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": depth, "unit": "m"}, {"id": "g2", "value": add, "unit": "m"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("search_project_documents", {"query": "pile cap depth"}, "c2")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c3")]},
        {"content": f"The depth with the blinding is {total} m."},
    ])
    shown = _text(_run(user_message=f"Add {add} m of blinding to the pile cap depth on the drawing.",
                       conversation_id=conv, project_id="late-passage-proj", sheet_mode=True))
    row = trace_store.records(_today(), conv)[0]
    assert [g.get("checked") for g in row["sheet"]["givens"]] == ["passage", "user"]
    assert f"{total} m" in shown


def _level_script(monkeypatch, level, rise, reply, inputs=None, rise_unit="m"):
    """The model writes two givens, plans one arithmetic step on them, runs
    it and replies ``reply``."""
    return _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "value": level, "unit": "m"}, {"id": "g2", "value": rise, "unit": rise_unit}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": inputs or {"operator": "add", "a": "g1", "b": "g2"}}]}, "c1")]},
        {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
        {"content": reply},
    ])


@pytest.mark.parametrize("level,rise,said,credited", [
    (-43, 10, "-33 m", True),      # as code wrote it
    (-43, 10, "33 m", False),      # the minus dropped in the reply
    (43, 10, "53 m", True),
    (43, 10, "-53 m", False),      # a minus code never wrote
    (-7, 2.5, "-4.5 m", True),
    (-7, 2.5, "4.5 m", False),
    (7, 2.5, "9.5 m", True),
    (7, 2.5, "-9.5 m", False),
])
def test_a_step_result_is_credited_to_its_step_only_with_the_sign_code_wrote(monkeypatch, level, rise, said, credited):
    conv = f"sign-{level}-{said}"
    _level_script(monkeypatch, level, rise, f"The new level is {said}.")
    shown = _text(_run(user_message=f"The formation level is {level} m. Raise it by {rise} m: what is the new level?",
                       conversation_id=conv, sheet_mode=True))
    row = trace_store.records(_today(), conv)[0]
    assert [st["result"]["value"] for st in row["sheet"]["steps"]] == [float(level + rise)]
    assert bool([f for f in row["figures"] if f.get("step") == "s1"]) is credited
    # The figure with the wrong sign is not on the screen at all, under any
    # credit: not as the step's, not as "working" on the givens' sizes.
    assert (said in shown) is credited
    assert (said.lstrip("-") in shown) is credited
    assert [f for f in row["figures"] if f.get("working")] == []
    assert len(row["figures"]) == (1 if credited else 0)


@pytest.mark.parametrize("level,rise,reply", [
    (-43, 10, "The new level is -33 m (\u221243 m + 10 m = \u221233 m)."),
    (-7, 2.5, "The new level is -4.5 m (\u22127 m + 2.5 m = \u22124.5 m)."),
    (43, 10, "The new level is 53 m (43 m + 10 m = 53 m)."),
])
def test_a_step_result_restated_with_its_givens_is_shown_whole(monkeypatch, level, rise, reply):
    """The shape a real model writes: the result, then the step restated
    with the givens, the minus typographic. Every figure on it is the
    step's or the user's, so nothing is cut."""
    conv = f"restated-{level}-{rise}"
    _level_script(monkeypatch, level, rise, reply)
    shown = _text(_run(user_message=f"The formation level is {level} m. Raise it by {rise} m: what is the new level?",
                       conversation_id=conv, sheet_mode=True))
    row = trace_store.records(_today(), conv)[0]
    assert shown.strip() == reply
    assert [f.get("step") for f in row["figures"]] == ["s1", None, None, "s1"]
    assert [f["source"] for f in row["figures"]] == ["calculator", "user_input", "user_input", "calculator"]


@pytest.mark.parametrize("level,rise", [(-5, 5), (12.5, -12.5)])
def test_a_step_result_of_zero_is_credited_to_its_step_and_shown(monkeypatch, level, rise):
    conv = f"zero-{level}"
    _level_script(monkeypatch, level, rise, "The new level is 0 m.")
    shown = _text(_run(user_message=f"The formation level is {level} m. Change it by {rise} m: what is the new level?",
                       conversation_id=conv, sheet_mode=True))
    row = trace_store.records(_today(), conv)[0]
    assert [st["result"]["value"] for st in row["sheet"]["steps"]] == [0.0]
    assert "0 m" in shown
    assert [(f["figure"], f.get("step")) for f in row["figures"]] == [("0 m", "s1")]


@pytest.mark.parametrize("level,rise,rise_unit,wrong,unit_first", [
    (-43, 250, "mm", "207 m", False), (-43, 250, "mm", "207 m", True),
    (12, 500, "mm", "512 m", False), (12, 500, "mm", "512 m", True),
    (40, 3, "km", "43 m", False), (40, 3, "km", "43 m", True),
])
def test_a_unit_the_plan_writes_beside_a_given_does_not_change_the_givens_unit(monkeypatch, level, rise, rise_unit,
                                                                               wrong, unit_first):
    """The user gave the rise in mm (or km). The plan names the given and
    writes "m" beside it. The step is not run as metres plus metres, and
    the sum of the two bare numbers is not shown."""
    conv = f"beside-{level}-{rise_unit}-{unit_first}"
    inputs = {"operator": "add", "a": "g1"}
    inputs.update([("b_unit", "m"), ("b", "g2")] if unit_first else [("b", "g2"), ("b_unit", "m")])
    _level_script(monkeypatch, level, rise, f"The new level is {wrong}.", inputs=inputs, rise_unit=rise_unit)
    shown = _text(_run(
        user_message=f"The level is {level} m. Raise it by {rise} {rise_unit}: what is the new level?",
        conversation_id=conv, sheet_mode=True))
    row = trace_store.records(_today(), conv)[0]
    step = row["sheet"]["steps"][0]
    assert step["status"] == "failed" and "result" not in step
    assert step["error"] == f"cannot add m and {rise_unit}: convert one to the other's unit first"
    assert wrong not in shown
    assert [f for f in row["figures"] if f["figure"] == wrong] == []


@pytest.mark.parametrize("level,rise,rise_unit,wrong", [(-43, 250, "mm", "207 m"), (12, 500, "mm", "512 m")])
def test_a_bare_number_of_the_plan_is_held_to_the_unit_written_beside_it(monkeypatch, level, rise, rise_unit, wrong):
    """The plan writes the user's number itself, bare, with another unit
    beside it (250 for the user's 250 mm, run as 250 m). The step runs, but
    its result rests on a quantity nobody gave, so it is not shown."""
    conv = f"bare-{level}-{rise}"
    _level_script(monkeypatch, level, rise, f"The new level is {wrong}.",
                  inputs={"operator": "add", "a": "g1", "b": rise, "b_unit": "m"}, rise_unit=rise_unit)
    shown = _text(_run(
        user_message=f"The level is {level} m. Raise it by {rise} {rise_unit}: what is the new level?",
        conversation_id=conv, sheet_mode=True))
    row = trace_store.records(_today(), conv)[0]
    assert [st["result"]["value"] for st in row["sheet"]["steps"]] == [float(level + rise)]
    assert wrong not in shown
    assert [f for f in row["figures"] if f.get("step") == "s1"] == []


@pytest.mark.parametrize("level,rise,total", [(-43, 10, "-33 m"), (12, 0.5, "12.5 m")])
def test_a_bare_number_of_the_plan_in_the_unit_the_user_gave_it_in_is_credited(monkeypatch, level, rise, total):
    conv = f"bare-ok-{level}"
    _level_script(monkeypatch, level, rise, f"The new level is {total}.",
                  inputs={"operator": "add", "a": "g1", "b": rise, "b_unit": "m"})
    shown = _text(_run(user_message=f"The level is {level} m. Raise it by {rise} m: what is the new level?",
                       conversation_id=conv, sheet_mode=True))
    row = trace_store.records(_today(), conv)[0]
    assert total in shown
    assert [f.get("step") for f in row["figures"]] == ["s1"]

# --- the recorded end events the answer's panel is tested on ----------------
# tests/fixtures/show_working holds real SSE 'end' events of this path;
# the panel's node and browser tests read them. Each is re-run here through
# loop.stream with the same scripted model and must come out the same, so
# the panel is never tested on a hand-made end event.

_FIXTURES = __import__("pathlib").Path(__file__).resolve().parent / "fixtures" / "show_working"

_RECORDED_TURNS = {
    "three_step_chain.json": (
        "each wall is 12 m long and 0.6 m high and there are 7 walls with 2 coats; what area do I paint",
        [{"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "what": "wall length", "value": 12, "unit": "m", "source": "user"},
                       {"id": "g2", "what": "wall height", "value": 0.6, "unit": "m", "source": "user"},
                       {"id": "g3", "what": "walls", "value": 7, "unit": "nr", "source": "user"},
                       {"id": "g4", "what": "coats", "value": 2, "unit": "nr", "source": "spec"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "g1", "b": "g2"}},
                     {"id": "s2", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "s1", "b": "g3"}},
                     {"id": "s3", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "s2", "b": "g4"}}]}, "c1")]},
         {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
         {"content": "", "tool_calls": [_call("run_step", {"step": "s2"}, "c3")]},
         {"content": "", "tool_calls": [_call("run_step", {"step": "s3"}, "c4")]},
         {"content": "The painted area is 100.8 m2."}],
        "ws-rec-1", ["user", "user", "user", "user"]),
    "stopped_on_missing_input.json": (
        "the pit is 1.75 m deep; work out the loose excavation volume",
        [{"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "what": "pit depth", "value": 1.75, "unit": "m", "source": "user"},
                       {"id": "g2", "what": "bulking factor", "value": 1.25, "unit": "-",
                        "source": "general knowledge"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "g1"}},
                     {"id": "s2", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "s1", "b": "g2"}}],
            "missing": [{"for": "s1", "input": "b", "label": "pit plan area"}]}, "c1")]},
         {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
         {"content": "What is the pit plan area?"}],
        "ws-rec-2", ["user", None]),
    "plan_literals.json": (
        "the trench holds 40 m3 of spoil; multiply by 1.05 for bulking and add the 2.5 m3 from the pit",
        [{"content": "", "tool_calls": [_call("sheet_update", {
            "givens": [{"id": "g1", "what": "trench spoil", "value": 40, "unit": "m3", "source": "user"}],
            "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "g1", "b": {"value": 1.05, "unit": "-"}}},
                     {"id": "s2", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "s1", "b": {"value": 2.5, "unit": "m3"}}}]}, "c1")]},
         {"content": "", "tool_calls": [_call("run_step", {"step": "s1"}, "c2")]},
         {"content": "", "tool_calls": [_call("run_step", {"step": "s2"}, "c3")]},
         {"content": "The loose spoil is 44.5 m3."}],
        "ws-rec-3", ["user"]),
}


def _without_turn(sheet):
    return {k: v for k, v in sheet.items() if k != "turn"}


@pytest.mark.parametrize("name", sorted(_RECORDED_TURNS))
def test_the_panel_fixtures_are_real_end_events(monkeypatch, name):
    message, replies, conv, marks = _RECORDED_TURNS[name]
    _scripted(monkeypatch, json.loads(json.dumps(replies)))
    events = _run(user_message=message, conversation_id=conv, project_id=None, sheet_mode=True)
    end = [e for e in events if e.get("type") == "end"][0]
    recorded = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert _text(events) == recorded["content"]
    assert end["ended"] == recorded["ended"]
    assert _without_turn(end["sheet"]) == _without_turn(recorded["sheet"])
    # Code's own check on each given is what the panel shows as its source:
    # a figure the user typed is marked "user"; one the model wrote is not.
    assert [g.get("checked") for g in end["sheet"]["givens"]] == marks


@pytest.mark.parametrize("seconds,elapsed,left,whole", [
    ("150", 0.0, 150, 150),
    ("150", 140.0, 10, 150),
    ("150", 0.4, 149, 150),     # rounded down: never more than is left
    ("10.5", 3.0, 7, 10),       # 7.5 left of 10.5: both rounded down
    ("10", 10.0, 0, 10),        # exactly spent
    ("10", 12.5, 0, 10),        # past the ceiling: never a negative figure
])
def test_sheet_note_states_the_seconds_left_of_the_seconds_ceiling(monkeypatch, seconds, elapsed, left, whole):
    monkeypatch.setenv("DRIVER_SHEET_MAX_SECONDS", seconds)
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", "7")
    from app.agents.driver import sheet

    work = sheet.new_sheet("n@1", "q", None)
    rules = loop._sheet_note(work, 3, elapsed).split("\n", 1)[0]
    assert f"round 3 of 7 for this turn, with {left} of {whole} seconds left;" in rules
    assert "no tool round starts after round 7 or once the seconds have run out" in rules


@pytest.mark.parametrize("seconds,elapsed", [("inf", 0.0), ("1e309", 40.0), ("Infinity", 1e9)])
def test_sheet_note_states_only_the_rounds_when_the_setting_leaves_no_seconds_ceiling(monkeypatch, seconds, elapsed):
    monkeypatch.setenv("DRIVER_SHEET_MAX_SECONDS", seconds)
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", "7")
    from app.agents.driver import sheet

    work = sheet.new_sheet("n@1", "q", None)
    rules = loop._sheet_note(work, 3, elapsed).split("\n", 1)[0]
    assert "This is round 3 of 7 for this turn; no tool round starts after round 7, so write" in rules
    assert "seconds" not in rules.split("This is round 3 of 7")[1]


@pytest.mark.parametrize("seconds,answer", [("inf", "I need the slab size to work that out."), ("1e309", "Nothing to add.")])
def test_a_sheet_turn_runs_when_the_setting_leaves_no_seconds_ceiling(monkeypatch, seconds, answer):
    # The whole turn, through the loop's own door: the note is built before
    # the first model call, so a note that cannot be built ends every turn.
    monkeypatch.setenv("DRIVER_SHEET_MAX_SECONDS", seconds)
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", "7")
    seen = _scripted(monkeypatch, [{"content": answer}])
    events = _run(user_message="how much concrete?", conversation_id=f"ws-inf-{seconds}", sheet_mode=True)
    assert _text(events) == answer and len(seen) == 1
    note = [m["content"] for m in seen[0]["messages"] if m["role"] == "system" and m["content"].startswith("Working sheet")]
    assert len(note) == 1 and "round 1 of 7 for this turn; no tool round starts after round 7" in note[0]


@pytest.mark.parametrize("seconds,left,whole", [("nan", 1, 1), ("-5", 1, 1), ("0", 1, 1), ("soon", 150, 150)])
def test_sheet_note_states_the_ceiling_the_loop_falls_back_to_for_a_setting_that_is_no_duration(
        monkeypatch, seconds, left, whole):
    monkeypatch.setenv("DRIVER_SHEET_MAX_SECONDS", seconds)
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", "7")
    from app.agents.driver import sheet

    work = sheet.new_sheet("n@1", "q", None)
    rules = loop._sheet_note(work, 3, 0.0).split("\n", 1)[0]
    assert f"round 3 of 7 for this turn, with {left} of {whole} seconds left;" in rules


@pytest.mark.parametrize("pause,seconds,rounds,tool_rounds", [
    (20.0, "150", "12", 8),   # the seconds end the turn at round 8 of 12
    (3.0, "10", "40", 4),     # the seconds end it at round 4 of 40
    (0.1, "10", "5", 5),      # negative: the rounds run out first, the seconds stay nearly whole
])
def test_every_round_states_the_seconds_the_loop_will_still_allow(monkeypatch, pause, seconds, rounds, tool_rounds):
    """The note offered with tools each round states the seconds the loop's own
    clock leaves; the round whose note says 0 seconds or the last round is the
    last one that runs, so no note promises a round the loop will not start."""
    import time as real_time
    import types

    monkeypatch.setenv("DRIVER_SHEET_MAX_SECONDS", seconds)
    monkeypatch.setenv("DRIVER_SHEET_MAX_ROUNDS", rounds)
    clock = {"now": 0.0}
    monkeypatch.setattr(loop, "time", types.SimpleNamespace(monotonic=lambda: clock["now"], time=real_time.time))
    notes = []

    async def slow(agent, messages, offered, model="", timeout=120.0):
        if offered:
            notes.append(next(m["content"] for m in messages
                              if m["role"] == "system" and m["content"].startswith("Working sheet")))
        clock["now"] += pause
        return {"status": "success", "message": {"content": "", "tool_calls": [
            _call("sheet_update", {"givens": [{"id": "g1", "value": len(notes), "unit": "m"}]}, f"c{len(notes)}")]}}

    monkeypatch.setattr(llm, "call", slow)
    conv = f"sl-{pause}-{rounds}"
    _run(user_message="q", conversation_id=conv, sheet_mode=True)
    assert trace_store.records(_today(), conv)[0]["ended"] == "ceiling"
    assert len(notes) == tool_rounds
    whole, spent = int(float(seconds)), 0.0
    for i, note in enumerate(notes, start=1):
        left = max(0, int(float(seconds) - spent))
        spent += pause
        assert f"round {i} of {rounds} for this turn, with {left} of {whole} seconds left;" in note.split("\n", 1)[0]
    # The last round run: the next round's start would be past the seconds,
    # or there is no next round.
    assert tool_rounds * pause > float(seconds) or tool_rounds == int(rounds)
