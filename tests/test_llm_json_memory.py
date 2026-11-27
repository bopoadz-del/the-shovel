"""The model's own JSON memory: write_memory / read_memory / list_memory on
the driver path, stored in the database, one project per note, the project
taken from the turn. Through the public chat route with a scripted model;
every test builds its own projects and notes and runs two different inputs."""
from __future__ import annotations

import json
import logging
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.driver import llm
from app.core import llm_memory
from app.main import app
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _driver(monkeypatch):
    monkeypatch.setenv("DRIVER_MODE", "request")

    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        reply = await llm.call(agent, messages, offered, model=model)
        if reply.get("status") == "success":
            text = reply["message"].get("content") or ""
            if text:
                yield "content", text
        yield "done", reply

    monkeypatch.setattr(llm, "stream", fake_stream)


def _user(client, tag):
    email = f"mem-{tag}-{_RUN}-{uuid.uuid4().hex[:6]}@x.com"
    client.post("/v1/users/register", json={"email": email, "password": PW_06})
    token = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _project(client, h, name):
    r = client.post("/v1/projects", json={"name": name}, headers=h)
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _call(name, args, cid):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def _script(monkeypatch, replies, seen):
    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append({"messages": [dict(m) for m in messages],
                     "tools": [t["function"]["name"] for t in offered]})
        return {"status": "success", "message": replies.pop(0)}
    monkeypatch.setattr(llm, "call", fake_call)


def _turn(client, h, pid, message="go", conversation=True):
    body = {"message": message, "project_id": pid, "history": []}
    if conversation:
        body["conversation_id"] = f"ws-{pid}-{uuid.uuid4().int % 10**13}"
    r = client.post("/v1/chat/stream", json=body, headers={**h, "X-Fork-Driver": "on"})
    return r


def _tool_results(seen):
    """Every tool message the model received, by call id, parsed."""
    out = {}
    for m in seen[-1]["messages"]:
        if m.get("role") == "tool":
            out[m["tool_call_id"]] = json.loads(m["content"])
    return out


def _run_one(client, monkeypatch, h, pid, name, args):
    """One turn: the model makes one call, then answers. Returns the result
    it received for that call."""
    seen = []
    _script(monkeypatch, [{"content": "", "tool_calls": [_call(name, args, "m1")]},
                          {"content": "Done."}], seen)
    r = _turn(client, h, pid)
    assert r.status_code == 200, r.text
    return _tool_results(seen)["m1"], seen


NOTES = [
    ("paths", {"ask_first": "which bill", "tool_for": {"rates": "bill_rows_aggregate"}}),
    ("how-to.json", [{"kind": "schedule", "start_with": "list_project_documents"}]),
]


@pytest.mark.parametrize("name,data", NOTES)
def test_a_note_written_in_one_conversation_is_read_in_another_of_the_project(client, monkeypatch, name, data):
    h = _user(client, "rw")
    pid = _project(client, h, "Memory RW")
    wrote, seen = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": name, "data": json.dumps(data)})
    assert wrote["ok"] is True
    stem = name[:-5] if name.endswith(".json") else name
    assert wrote["result"]["written"] == stem
    assert {"write_memory", "read_memory", "list_memory"} <= set(seen[0]["tools"])
    # A later turn, another conversation, same project.
    got, _ = _run_one(client, monkeypatch, h, pid, "read_memory", {"name": name})
    assert got["ok"] is True
    assert got["result"]["data"] == data
    assert got["result"]["note"].startswith("Your own note, written ")
    assert got["result"]["written_at"][:10] in got["result"]["note"]
    listed, _ = _run_one(client, monkeypatch, h, pid, "list_memory", {})
    assert [n["name"] for n in listed["result"]["notes"]] == [stem]


@pytest.mark.parametrize("name,data", NOTES)
def test_another_project_cannot_read_or_list_the_note(client, monkeypatch, name, data):
    h = _user(client, "iso")
    mine, other = _project(client, h, "Memory A"), _project(client, h, "Memory B")
    _run_one(client, monkeypatch, h, mine, "write_memory", {"name": name, "data": json.dumps(data)})
    got, _ = _run_one(client, monkeypatch, h, other, "read_memory", {"name": name})
    assert got["ok"] is False and "no note named" in got["result"]["error"]
    listed, _ = _run_one(client, monkeypatch, h, other, "list_memory", {})
    assert listed["result"]["notes"] == []


@pytest.mark.parametrize("name,data", NOTES)
def test_the_project_comes_from_the_turn_never_from_the_arguments(client, monkeypatch, name, data):
    h = _user(client, "arg")
    mine, other = _project(client, h, "Memory C"), _project(client, h, "Memory D")
    # The model names another project in its arguments: the call is refused
    # by the strict schema shape or the note lands in the turn's project.
    wrote, _ = _run_one(client, monkeypatch, h, mine, "write_memory",
                        {"name": name, "data": json.dumps(data), "project_id": other})
    assert wrote["ok"] is True
    assert llm_memory.listing(other) == []
    assert [n["name"] for n in llm_memory.listing(mine)] == [wrote["result"]["written"]]


@pytest.mark.parametrize("name,data", NOTES)
def test_another_users_project_cannot_be_read_through_the_route(client, monkeypatch, name, data):
    owner, stranger = _user(client, "own"), _user(client, "str")
    pid = _project(client, owner, "Memory owner")
    _run_one(client, monkeypatch, owner, pid, "write_memory", {"name": name, "data": json.dumps(data)})
    seen = []
    _script(monkeypatch, [{"content": "", "tool_calls": [_call("read_memory", {"name": name}, "m1")]},
                          {"content": "Done."}], seen)
    # In a conversation of that project: refused at the door, no model call.
    assert _turn(client, stranger, pid).status_code == 404
    assert seen == []
    # Without a conversation: the route drops a project the caller cannot
    # access, so the turn has no project and the model's read is refused.
    r = _turn(client, stranger, pid, conversation=False)
    assert r.status_code == 200, r.text
    reached = [json.loads(m["content"]) for m in seen[-1]["messages"] if m.get("role") == "tool"]
    assert len(reached) == 1
    assert reached[0]["ok"] is False and "needs a project" in reached[0]["result"]["error"]
    assert "data" not in reached[0]["result"]
    # The owner still reads it whole.
    assert llm_memory.read(pid, name)["data"] == data


@pytest.mark.parametrize("name", ["../etc/passwd", "a" * 65, "with space", "", "notes.txt"])
def test_a_bad_name_is_an_error_the_model_receives(client, monkeypatch, name):
    h = _user(client, "bad")
    pid = _project(client, h, "Memory bad name")
    got, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": name, "data": "{}"})
    assert got["ok"] is False and "not allowed" in got["result"]["error"]
    assert llm_memory.listing(pid) == []


@pytest.mark.parametrize("data", ["{not json", "plain words", "42", '"a string"'])
def test_data_that_is_not_a_json_document_is_an_error_the_model_receives(client, monkeypatch, data):
    h = _user(client, "nj")
    pid = _project(client, h, "Memory non json")
    got, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": "n1", "data": data})
    assert got["ok"] is False
    assert "JSON" in got["result"]["error"]
    assert llm_memory.listing(pid) == []


@pytest.mark.parametrize("limit", [64, 200])
def test_an_oversize_note_is_an_error_and_the_limit_is_a_setting(client, monkeypatch, limit):
    monkeypatch.setenv("LLM_MEMORY_MAX_BYTES", str(limit))
    h = _user(client, "big")
    pid = _project(client, h, "Memory big")
    fits = json.dumps({"k": "x" * (limit - 8)})   # exactly `limit` bytes compact
    assert len(json.dumps(json.loads(fits), separators=(",", ":")).encode()) == limit
    ok, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": "edge", "data": fits})
    assert ok["ok"] is True and ok["result"]["bytes"] == limit
    over = json.dumps({"k": "x" * (limit - 7)})
    got, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": "big", "data": over})
    assert got["ok"] is False and f"limit is {limit} bytes" in got["result"]["error"]
    assert [n["name"] for n in llm_memory.listing(pid)] == ["edge"]


@pytest.mark.parametrize("bad", ["{broken", json.dumps({"k": "x" * 400})])
def test_a_failed_write_leaves_the_old_note_whole(client, monkeypatch, bad):
    monkeypatch.setenv("LLM_MEMORY_MAX_BYTES", "300")
    h = _user(client, "keep")
    pid = _project(client, h, "Memory keep")
    old = {"start_with": "list_project_documents", "n": 1}
    _run_one(client, monkeypatch, h, pid, "write_memory", {"name": "keep", "data": json.dumps(old)})
    got, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": "keep", "data": bad})
    assert got["ok"] is False
    assert llm_memory.read(pid, "keep")["data"] == old


@pytest.mark.parametrize("limit", [1, 2])
def test_the_document_count_per_project_is_a_setting(client, monkeypatch, limit):
    monkeypatch.setenv("LLM_MEMORY_MAX_DOCS", str(limit))
    h = _user(client, "cnt")
    pid = _project(client, h, "Memory count")
    for i in range(limit):
        assert _run_one(client, monkeypatch, h, pid, "write_memory",
                        {"name": f"n{i}", "data": "{}"})[0]["ok"] is True
    # Replacing an existing note is not a new document.
    assert _run_one(client, monkeypatch, h, pid, "write_memory",
                    {"name": "n0", "data": '{"v": 2}'})[0]["ok"] is True
    got, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": "extra", "data": "{}"})
    assert got["ok"] is False and "limit" in got["result"]["error"]
    assert len(llm_memory.listing(pid)) == limit


@pytest.mark.parametrize("docs", [
    ({"a": 1, "path": ["x"] * 30}, {"b": 2, "path": ["y"] * 30}),
    ([{"k": "first"}] * 20, [{"k": "second"}] * 25),
])
def test_two_writes_at_once_leave_one_whole_document(docs):
    pid = f"mem-race-{uuid.uuid4().hex[:8]}"
    barrier = threading.Barrier(2)
    errors = []

    def write(doc):
        try:
            barrier.wait()
            llm_memory.write_retrying(pid, "race", json.dumps(doc), "t#1")
        except Exception as exc:  # noqa: BLE001 -- surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(d,)) for d in docs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert llm_memory.read(pid, "race")["data"] in [docs[0], docs[1]]
    assert len(llm_memory.listing(pid)) == 1


@pytest.mark.parametrize("name,data", NOTES)
def test_no_note_reaches_the_prompt_unless_the_model_reads_it(client, monkeypatch, name, data):
    h = _user(client, "inj")
    pid = _project(client, h, "Memory inject")
    marker = f"marker-{uuid.uuid4().hex[:10]}"
    payload = {"marker": marker, "body": data}
    _run_one(client, monkeypatch, h, pid, "write_memory", {"name": name, "data": json.dumps(payload)})
    seen = []
    _script(monkeypatch, [{"content": "Answer without notes."}], seen)
    assert _turn(client, h, pid).status_code == 200
    assert marker not in json.dumps(seen[0]["messages"])


@pytest.mark.parametrize("name,data", NOTES)
def test_every_memory_call_is_in_the_turns_trace(client, monkeypatch, caplog, name, data):
    h = _user(client, "trc")
    pid = _project(client, h, "Memory trace")
    with caplog.at_level(logging.INFO, logger="app.agents.driver.loop"):
        wrote, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": name, "data": json.dumps(data)})
        _run_one(client, monkeypatch, h, pid, "read_memory", {"name": name})
    traces = [c["trace"] for line in caplog.messages if line.startswith("DRIVER_TURN ")
              for step in json.loads(line[len("DRIVER_TURN "):])["steps"] if isinstance(step, dict)
              for c in step.get("calls") or [] if c.get("trace")]
    stem = wrote["result"]["written"]
    size = wrote["result"]["bytes"]
    assert [(t["memory"], t["name"], t["bytes"]) for t in traces] == [("write", stem, size), ("read", stem, size)]
    assert all(t["turn"].endswith("#1") for t in traces)


@pytest.mark.parametrize("stated", [None, "not a dict", ["a", "list"]])
def test_a_call_whose_tool_states_no_trace_keeps_the_records_shape(monkeypatch, caplog, stated):
    """Only a tool that states its own trace dict adds a "trace" key to the
    turn's call record; any other result leaves the record's keys as before."""
    from tests.lane5_core_support import call, drive, install_driver_stream
    from app.agents.runtime import Agent

    install_driver_stream(monkeypatch)
    given = {"memory": "list", "name": None, "bytes": 3, "turn": "t#1", "ok": True}
    results = {"stated_tool": {"ok": True, "trace": given}, "plain_tool": {"ok": True}}
    if stated is not None:
        results["plain_tool"]["trace"] = stated

    async def fake_run_tool_call(self, tool_call, *a, **k):
        return results[tool_call["function"]["name"]]

    monkeypatch.setattr(Agent, "_run_tool_call", fake_run_tool_call)
    with caplog.at_level(logging.INFO, logger="app.agents.driver.loop"):
        drive(monkeypatch, "go", [
            {"content": "", "tool_calls": [call("plain_tool", {}, "p1"), call("stated_tool", {}, "s1")]},
            {"content": "Done."},
        ])
    lines = [m for m in caplog.messages if m.startswith("DRIVER_TURN ")]
    calls = json.loads(lines[-1][len("DRIVER_TURN "):])["steps"][0]["calls"]
    by_tool = {c["tool"]: c for c in calls}
    assert "trace" not in by_tool["plain_tool"]
    assert by_tool["stated_tool"]["trace"] == given
    assert set(by_tool["stated_tool"]) - set(by_tool["plain_tool"]) == {"trace"}


@pytest.mark.parametrize("name,data", NOTES)
def test_the_admin_endpoint_lists_a_projects_notes(client, monkeypatch, name, data):
    h = _user(client, "adm")
    pid = _project(client, h, "Memory admin")
    _run_one(client, monkeypatch, h, pid, "write_memory", {"name": name, "data": json.dumps(data)})
    # The project's own (non-admin) owner is refused.
    assert client.get(f"/v1/admin/projects/{pid}/memory", headers=h).status_code == 403
    from app.dependencies import require_api_key

    app.dependency_overrides[require_api_key] = lambda: {"user_id": "test-admin", "role": "admin"}
    try:
        got = client.get(f"/v1/admin/projects/{pid}/memory")
    finally:
        app.dependency_overrides.pop(require_api_key, None)
    assert got.status_code == 200, got.text
    notes = got.json()["notes"]
    assert [n["data"] for n in notes] == [data] and notes[0]["written_by_turn"].endswith("#1")


@pytest.mark.parametrize("supports,expected", [(True, True), (False, False), (None, False)])
def test_the_strict_flag_goes_only_to_a_provider_that_declares_it(supports, expected):
    from app.agents.base import memory
    from app.agents.runtime import Agent

    agent = Agent(name="p", description="p", system_prompt="", allowed_blocks=[])
    cfg = {"default_model": "m"} if supports is None else {"default_model": "m", "supports_strict_tools": supports}
    payload = llm._payload(cfg, agent, [{"role": "user", "content": "x"}],
                           [memory.WRITE_SCHEMA, memory.LIST_SCHEMA], "m", stream=False)
    for t in payload["tools"]:
        assert ("strict" in t["function"]) is expected
        assert t["function"]["parameters"]["additionalProperties"] is False
    assert memory.WRITE_SCHEMA["function"]["strict"] is True  # the declared schema is never changed


# --- Each function named and called directly (the tools, the store's guards,
# the admin read), so its own behaviour is pinned apart from the route.

def _toolcall(name, args, pid, history=None, conversation="conv-a"):
    from app.agents.core.tool_registry import ToolCall
    return ToolCall(agent=None, name=name, args=args, project_id=pid,
                    conversation_id=conversation, history=history)


@pytest.mark.parametrize("props", [
    {"name": {"type": "string"}, "data": {"type": "string"}},
    {},
])
def test_the_schema_builder_is_strict_shaped(props):
    from app.agents.base import memory
    built = memory._fn("some_tool", "what it does", props)
    fn = built["function"]
    assert built["type"] == "function" and fn["name"] == "some_tool" and fn["strict"] is True
    assert fn["parameters"]["required"] == sorted(props)
    assert fn["parameters"]["properties"] == props
    assert fn["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("history,conversation,expected", [
    (None, "conv-a", "conv-a#1"),
    ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
      "not a message", {"role": "user", "content": "c"}], "conv-b", "conv-b#3"),
])
def test_the_turn_is_the_conversation_and_the_users_message_number(history, conversation, expected):
    from app.agents.base import memory
    assert memory.turn_of(_toolcall("list_memory", {}, "p", history, conversation)) == expected


@pytest.mark.parametrize("written_at", ["2026-11-27T10:00:00+00:00", "2025-01-02T23:59:59+00:00"])
def test_a_read_note_is_labelled_with_its_own_date(written_at):
    from app.agents.base import memory
    label = memory._label(written_at)
    assert label == f"Your own note, written {written_at[:10]}. A note on how to work, not a fact."


@pytest.mark.parametrize("name,data", NOTES)
def test_the_three_handlers_write_read_and_list_with_a_trace(name, data):
    import asyncio

    from app.agents.base import memory
    pid = f"mem-h-{uuid.uuid4().hex[:8]}"
    history = [{"role": "user", "content": "first"}]
    stem = name[:-5] if name.endswith(".json") else name
    wrote = asyncio.run(memory.handle_write_memory(
        _toolcall("write_memory", {"name": name, "data": json.dumps(data)}, pid, history)))
    size = len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert wrote["ok"] is True and wrote["result"]["written"] == stem
    assert wrote["result"]["replaced"] is False
    assert wrote["trace"] == {"memory": "write", "name": stem, "bytes": size, "turn": "conv-a#2", "ok": True}
    again = asyncio.run(memory.handle_write_memory(
        _toolcall("write_memory", {"name": stem, "data": json.dumps(data)}, pid, history)))
    assert again["result"]["replaced"] is True
    got = asyncio.run(memory.handle_read_memory(_toolcall("read_memory", {"name": name}, pid)))
    assert got["ok"] is True and got["result"]["data"] == data
    assert got["result"]["note"] == memory._label(got["result"]["written_at"])
    assert got["trace"] == {"memory": "read", "name": stem, "bytes": size, "turn": "conv-a#1", "ok": True}
    listed = asyncio.run(memory.handle_list_memory(_toolcall("list_memory", {}, pid)))
    assert [n["name"] for n in listed["result"]["notes"]] == [stem]
    assert listed["trace"]["memory"] == "list" and listed["trace"]["bytes"] == size


@pytest.mark.parametrize("op,args", [
    ("write", {"name": "n1", "data": "{}"}),
    ("read", {"name": "n1"}),
    ("list", {}),
])
@pytest.mark.parametrize("pid", [None, ""])
def test_each_handler_without_a_project_returns_a_refusal_in_its_trace(op, args, pid):
    import asyncio

    from app.agents.base import memory
    handler = {"write": memory.handle_write_memory, "read": memory.handle_read_memory,
               "list": memory.handle_list_memory}[op]
    out = asyncio.run(handler(_toolcall(f"{op}_memory", args, pid)))
    assert out["ok"] is False and "needs a project" in out["result"]["error"]
    assert out["trace"] == {"memory": op, "name": args.get("name"), "bytes": 0,
                            "turn": "conv-a#1", "ok": False}
    refused = memory._refused(_toolcall(f"{op}_memory", args, pid), op, "why", 7)
    assert refused["trace"]["name"] is None and refused["result"]["error"] == "why"


@pytest.mark.parametrize("name", [42, None])
def test_a_name_that_is_not_text_is_refused(name):
    with pytest.raises(llm_memory.MemoryRefused, match="must be a text"):
        llm_memory.clean_name(name)


@pytest.mark.parametrize("pid", ["", None])
def test_the_store_refuses_every_call_without_a_project(pid):
    with pytest.raises(llm_memory.MemoryRefused, match="needs a project"):
        llm_memory.write(pid, "n1", "{}")
    with pytest.raises(llm_memory.MemoryRefused, match="needs a project"):
        llm_memory.read(pid, "n1")
    with pytest.raises(llm_memory.MemoryRefused, match="needs a project"):
        llm_memory.listing(pid)


@pytest.mark.parametrize("first,second,makes_dir", [
    ("sqlite:///{tmp}/a/mem.db", "sqlite:///{tmp}/b/mem.db", True),
    ("postgresql://h1/db", "postgresql://h2/db", False),
])
def test_init_db_creates_the_table_once_per_database(monkeypatch, tmp_path, first, second, makes_dir):
    created = []

    class _Table:
        def create(self, bind=None, checkfirst=False):
            created.append(checkfirst)

    class _Model:
        __table__ = _Table()

    urls = [first.format(tmp=tmp_path.as_posix()), second.format(tmp=tmp_path.as_posix())]
    current = {"url": urls[0]}
    monkeypatch.setattr(llm_memory, "LlmMemory", _Model)
    monkeypatch.setattr(llm_memory, "get_database_url", lambda: current["url"])
    monkeypatch.setattr(llm_memory, "_initialized_for_url", None)
    entered = []

    class _Lock:
        def __enter__(self):
            entered.append(current["url"])

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(llm_memory, "_lock", _Lock())
    llm_memory.init_db()
    llm_memory.init_db()
    assert created == [True]
    # Once a database is set up, a call returns before taking the lock.
    assert entered == [urls[0]]
    current["url"] = urls[1]
    llm_memory.init_db()
    assert created == [True, True]
    assert entered == urls
    assert (tmp_path / "a").is_dir() is makes_dir and (tmp_path / "b").is_dir() is makes_dir


@pytest.mark.parametrize("tool", [{"type": "function"}, {"type": "function", "function": "text"},
                                  {"type": "function", "function": {"name": "t", "parameters": {}}}])
def test_a_tool_without_a_strict_flag_passes_through_unchanged(tool):
    assert llm._unstrict(tool) is tool


@pytest.mark.parametrize("extra", [{"description": "d"}, {}])
def test_the_strict_flag_is_taken_off_and_the_declared_tool_kept(extra):
    tool = {"type": "function", "function": {"name": "t", "strict": True, **extra}}
    out = llm._unstrict(tool)
    assert out["function"] == {"name": "t", **extra}
    assert tool["function"]["strict"] is True


@pytest.mark.parametrize("name,data", NOTES)
def test_the_admin_read_returns_the_notes_whole_and_needs_an_admin(name, data):
    from fastapi import HTTPException

    from app.routers.admin import admin_project_memory
    pid = f"mem-adm-{uuid.uuid4().hex[:8]}"
    llm_memory.write(pid, name, json.dumps(data), "c#1")
    got = admin_project_memory(pid, auth={"user_id": "a", "role": "admin"})
    assert got["project_id"] == pid and got["count"] == 1
    assert [n["data"] for n in got["notes"]] == [data]
    with pytest.raises(HTTPException) as refused:
        admin_project_memory(pid, auth={"user_id": "u", "role": "user"})
    assert refused.value.status_code == 403


def _streamed_text(r):
    """The answer text the user sees: every token event of the stream."""
    out = []
    for line in r.text.splitlines():
        if line.startswith("data:"):
            try:
                event = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "token":
                out.append(str(event.get("content") or ""))
    return "".join(out)


@pytest.mark.parametrize("call,answer", [
    (("write_memory", {"name": "paths", "data": json.dumps({"ask_first": "which bill"})}),
     "The slab needs 12 cubic metres of concrete."),
    (("write_memory", {"name": "how-to", "data": json.dumps([{"start_with": "list_project_documents"}])}),
     "Start from the project document list."),
    (("read_memory", {"name": "paths"}), "The slab needs 12 cubic metres of concrete."),
    (("list_memory", {}), "Ask which bill first."),
])
def test_the_users_answer_never_credits_or_shows_the_notes(client, monkeypatch, call, answer):
    h = _user(client, "ans")
    pid = _project(client, h, "Memory answer")
    llm_memory.write(pid, "paths", json.dumps({"ask_first": "which bill"}), "c#1")
    seen = []
    _script(monkeypatch, [{"content": "", "tool_calls": [_call(call[0], call[1], "m1")]},
                          {"content": answer}], seen)
    r = _turn(client, h, pid)
    assert r.status_code == 200, r.text
    assert _tool_results(seen)["m1"]["ok"] is True
    text = _streamed_text(r)
    assert answer in text
    assert "Calculated with" not in text
    assert "Working notes" not in text and "ask_first" not in text and "start_with" not in text


@pytest.mark.parametrize("data", ['{"x": NaN}', "[Infinity, 1]", '{"y": -Infinity}'])
def test_nan_and_infinity_are_refused_and_the_owner_still_reads_the_project(client, monkeypatch, data):
    from app.routers.admin import admin_project_memory
    h = _user(client, "nan")
    pid = _project(client, h, "Memory nan")
    llm_memory.write(pid, "kept", json.dumps({"ok": 1}), "c#1")
    got, _ = _run_one(client, monkeypatch, h, pid, "write_memory", {"name": "bad", "data": data})
    assert got["ok"] is False and "not a JSON number" in got["result"]["error"]
    assert [n["name"] for n in llm_memory.listing(pid)] == ["kept"]
    shown = admin_project_memory(pid, auth={"user_id": "a", "role": "admin"})
    assert [n["data"] for n in shown["notes"]] == [{"ok": 1}]
    json.dumps(shown, allow_nan=False)  # the admin page's answer is strict JSON


@pytest.mark.parametrize("value", [{"x": float("nan")}, [1, float("inf")], {"y": [float("-inf")]}])
def test_a_document_given_as_a_value_with_nan_or_infinity_is_refused(value):
    with pytest.raises(llm_memory.MemoryRefused, match="NaN and Infinity"):
        llm_memory.parse_data(value)


@pytest.mark.parametrize("value,stored", [('{"x": 1.5}', '{"x":1.5}'), ("[1e3, -2]", "[1000.0,-2]")])
def test_finite_numbers_stay_accepted(value, stored):
    assert llm_memory.parse_data(value)[0] == stored


@pytest.mark.parametrize("first,second", [({"v": 1}, {"v": 2}), ([1], [2, 3])])
def test_a_write_that_loses_the_insert_race_replaces_the_winners_note(monkeypatch, first, second):
    from sqlalchemy.exc import IntegrityError

    pid = f"mem-retry-{uuid.uuid4().hex[:8]}"
    real = llm_memory.write
    calls = []

    def racing(project_id, name, data, turn=""):
        calls.append(turn)
        if len(calls) == 1:
            # Another writer stores the same new note first; this insert collides.
            real(project_id, name, json.dumps(first), "other#1")
            raise IntegrityError("INSERT", {}, Exception("duplicate key"))
        return real(project_id, name, data, turn)

    monkeypatch.setattr(llm_memory, "write", racing)
    done = llm_memory.write_retrying(pid, "race", json.dumps(second), "mine#2")
    assert calls == ["mine#2", "mine#2"]
    assert done["created"] is False
    note = llm_memory.read(pid, "race")
    assert note["data"] == second and note["written_by_turn"] == "mine#2"


def _tool_msg(name, result):
    return {"role": "tool", "name": name, "tool_call_id": f"id-{name}",
            "content": json.dumps({"name": name, "ok": True, "result": result})}


@pytest.mark.parametrize("figure", [12, 2160000])
def test_a_note_is_never_evidence_for_a_figure(figure):
    from app.agents import citation_provenance as cp
    note = _tool_msg("read_memory", {"status": "success", "data": {"slab_m3": figure}})
    calc = _tool_msg("payment_certificate", {"status": "success", "net_due": figure})
    tools = lambda msgs: [r.tool for r in cp.build_evidence(None, msgs).records if r.kind == "tool_run"]
    assert tools([note]) == []
    assert tools([note, calc]) == ["payment_certificate"]


@pytest.mark.parametrize("tier", [1, 3])
def test_a_note_that_names_a_source_is_never_a_source_row(tier):
    from app.agents.driver.sources import cited_entry_rows
    entry = {"id": "r-1", "statement": "Cover is 50 mm to footings.", "credibility_tier": tier,
             "provenance": {"source": "Model's own guess", "verified_against_standard": True}}
    answer = "Cover is 50 mm to footings."
    assert cited_entry_rows([_tool_msg("read_memory", {"data": entry})], answer) == []
    shown = cited_entry_rows([_tool_msg("knowledge_lookup", {"entries": [entry]})], answer)
    assert [(r["doc_name"], r["credibility_tier"]) for r in shown] == [("Model's own guess", tier)]


@pytest.mark.parametrize("name,expected", [
    ("write_memory", True), ("read_memory", True), ("list_memory", True),
    ("search_project_documents", False), ("payment_certificate", False),
    ("no_such_tool", False), (None, False), ("", False), (["write_memory"], False), ({"n": 1}, False),
])
def test_only_a_tool_declaring_notes_is_notes(name, expected):
    from app.agents.core.tool_registry import is_notes
    assert is_notes(name) is expected


@pytest.mark.parametrize("calls,expected", [
    ([("write_memory", {"name": "n", "data": "{}"}), ("payment_certificate", {"gross_valuation": 10})],
     1),
    ([("list_memory", {}), ("read_memory", {"name": "n"})], 0),
])
def test_the_credit_lines_skip_notes_tools_only(calls, expected):
    from app.agents.driver.sources import tool_credit_lines
    text = tool_credit_lines(calls)
    assert text.count("Calculated with") == expected
    assert "Working notes" not in text


@pytest.mark.parametrize("junk", [None, "a plain string"])
def test_message_records_skip_a_non_message_and_a_notes_tool(junk):
    from app.agents.citation_provenance import _message_records
    note = _tool_msg("write_memory", {"status": "success", "written": "n"})
    calc = _tool_msg("payment_certificate", {"status": "success", "net_due": 5})
    records = _message_records([junk, note, calc])
    assert [(r.kind, r.tool) for r in records] == [("tool_run", "payment_certificate")]


@pytest.mark.parametrize("token", ["NaN", "-Infinity"])
def test_not_json_constant_refuses_the_token_by_name(token):
    with pytest.raises(ValueError, match=f"^{token} is not a JSON number$"):
        llm_memory._not_json_constant(token)


@pytest.fixture
def failing_tool():
    from app.agents.core.tool_registry import _TOOLS, tool

    name = "memtest_item_check"

    @tool(name, owner="base", display_name="Item check")
    async def handle(_call):
        return {"name": name, "ok": False, "error": "Could not use these items: red, green, blue",
                "result": {"items": ["red", "green", "blue"], "item_count": 0}}

    yield name
    _TOOLS.pop(name, None)


@pytest.mark.parametrize("figure", ["9", "418"])
@pytest.mark.parametrize("notes_call", [
    None,
    ("write_memory", {"name": "paths", "data": json.dumps({"ask_first": "which bill"})}),
    ("read_memory", {"name": "paths"}),
    ("list_memory", {}),
])
def test_a_notes_call_does_not_back_a_figure_when_the_tool_failed(client, monkeypatch, failing_tool,
                                                                  figure, notes_call):
    """A turn whose only real tool failed shows the failure and no figure --
    the same with a successful notes call beside it as without one."""
    h = _user(client, "nb")
    pid = _project(client, h, "Notes back nothing")
    llm_memory.write(pid, "paths", json.dumps({"ask_first": "which bill"}), "c#1")
    calls = [_call(failing_tool, {}, "t1")]
    if notes_call:
        calls.append(_call(notes_call[0], notes_call[1], "m1"))
    seen = []
    _script(monkeypatch, [{"content": "", "tool_calls": calls},
                          {"content": f"0 items were found. The total is {figure}."}], seen)
    r = _turn(client, h, pid, message="how many items are usable?")
    assert r.status_code == 200, r.text
    if notes_call:
        assert _tool_results(seen)["m1"]["ok"] is True
    text = _streamed_text(r)
    assert f"The total is {figure}" not in text
    assert "Could not use these items: red, green, blue" in text


@pytest.mark.parametrize("data,where", [("{bad", "at character 1"), ("[1,2", "at character 4")])
def test_data_that_is_not_json_says_where_it_breaks(data, where):
    with pytest.raises(llm_memory.MemoryRefused, match=where):
        llm_memory.parse_data(data)


@pytest.mark.parametrize("words", ["my card is 4111 1111 1111 1111", "ring Ahmed on 050 123 4567 about the bill"])
@pytest.mark.parametrize("op", ["write_memory", "read_memory"])
def test_a_refused_name_never_puts_its_words_in_the_turns_log(client, monkeypatch, caplog, words, op):
    """A refused name is free text from the model and can hold the user's
    words: the turn's log line keeps no part of it."""
    h = _user(client, "leak")
    pid = _project(client, h, "Memory refused name")
    args = {"name": words, "data": "{}"} if op == "write_memory" else {"name": words}
    with caplog.at_level(logging.INFO, logger="app.agents.driver.loop"):
        got, _ = _run_one(client, monkeypatch, h, pid, op, args)
    assert got["ok"] is False
    lines = [line for line in caplog.messages if line.startswith("DRIVER_TURN ")]
    assert lines and all(words not in line for line in lines)
    traces = [c["trace"] for line in lines
              for step in json.loads(line[len("DRIVER_TURN "):])["steps"] if isinstance(step, dict)
              for c in step.get("calls") or [] if c.get("trace")]
    assert [(t["name"], t["ok"]) for t in traces] == [(None, False)]


@pytest.mark.parametrize("name,kept", [("paths", "paths"), ("how-to.json", "how-to"),
                                       ("with space", None), ("../etc/passwd", None), (42, None)])
def test_the_trace_keeps_a_name_only_when_it_is_a_valid_one(name, kept):
    from app.agents.base import memory

    assert memory._traced_name(name) == kept


@pytest.mark.parametrize("limit,already", [(3, 2), (2, 1)])
def test_new_notes_written_at_once_never_pass_the_projects_limit(monkeypatch, limit, already):
    """Several turns each create a different new note with one place left:
    exactly one gets it."""
    monkeypatch.setenv("LLM_MEMORY_MAX_DOCS", str(limit))
    pid = f"mem-limit-{uuid.uuid4().hex[:8]}"
    for i in range(already):
        llm_memory.write(pid, f"held{i}", "{}", "t#0")
    barrier = threading.Barrier(6)
    refused, errors = [], []

    def create(i):
        try:
            barrier.wait()
            llm_memory.write_retrying(pid, f"new{i}", json.dumps({"i": i}), "t#1")
        except llm_memory.MemoryRefused:
            refused.append(i)
        except Exception as exc:  # noqa: BLE001 -- surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=create, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(llm_memory.listing(pid)) == limit and len(refused) == 5


@pytest.mark.parametrize("dialect,locks", [("postgresql", 1), ("sqlite", 0)])
def test_the_new_note_lock_is_taken_where_two_writers_can_count_at_once(dialect, locks):
    import types

    ran = []
    session = types.SimpleNamespace(
        get_bind=lambda: types.SimpleNamespace(dialect=types.SimpleNamespace(name=dialect)),
        execute=lambda statement, params: ran.append((str(statement), params)))
    llm_memory._one_new_note_at_a_time(session, "proj-7")
    assert len(ran) == locks
    if locks:
        assert "pg_advisory_xact_lock" in ran[0][0] and ran[0][1] == {"project": "llm_memory:proj-7"}


@pytest.mark.parametrize("sent,stored", [("Paths.JSON", "Paths"), ("how-to.Json", "how-to")])
def test_a_json_suffix_in_any_case_comes_off_the_stored_name(sent, stored):
    pid = f"mem-case-{uuid.uuid4().hex[:8]}"
    assert llm_memory.write(pid, sent, '{"v": 1}', "t#1")["name"] == stored
    assert llm_memory.read(pid, stored)["data"] == {"v": 1}
    assert llm_memory.read(pid, sent)["name"] == stored
    assert [n["name"] for n in llm_memory.listing(pid)] == [stored]


@pytest.mark.parametrize("name,expected", [("Paths.JSON", "Paths"), ("a_1", "a_1"),
                                           ("x" * 65, None), ("two words", None), (["l"], None)])
def test_valid_name_gives_the_stored_name_or_none(name, expected):
    assert llm_memory.valid_name(name) == expected


@pytest.mark.parametrize("setting", ["0", "-3"])
def test_a_limit_set_to_zero_or_below_falls_back_to_the_default(monkeypatch, setting):
    monkeypatch.setenv("LLM_MEMORY_MAX_DOCS", setting)
    monkeypatch.setenv("LLM_MEMORY_MAX_BYTES", setting)
    assert llm_memory.max_docs() == llm_memory.DEFAULT_MAX_DOCS
    assert llm_memory.max_bytes() == llm_memory.DEFAULT_MAX_BYTES
    pid = f"mem-lim-{uuid.uuid4().hex[:8]}"
    assert llm_memory.write(pid, "kept", '{"a": [1, 2, 3]}', "t#1")["created"] is True
    assert llm_memory.read(pid, "kept")["data"] == {"a": [1, 2, 3]}


@pytest.mark.parametrize("written,listed", [(["zeta", "alpha", "mid"], ["alpha", "mid", "zeta"]),
                                            (["b2", "a1"], ["a1", "b2"])])
def test_notes_list_in_name_order_for_the_model_and_the_owner(written, listed):
    """SQLite returns a project's rows in primary-key (name) order even with
    no ORDER BY; its reverse_unordered_selects pragma turns that order round,
    so only a query that asks for name order lists them in name order."""
    from sqlalchemy import event

    from app.core.db import get_engine
    from app.routers.admin import admin_project_memory
    engine = get_engine()
    pid = f"mem-ord-{uuid.uuid4().hex[:8]}"
    for name in written:
        llm_memory.write(pid, name, json.dumps({"n": name}), "t#1")

    def reverse_unordered(conn, cursor, statement, params, context, executemany):
        cursor.execute("PRAGMA reverse_unordered_selects = ON")

    sqlite = engine.dialect.name == "sqlite"
    if sqlite:
        event.listen(engine, "before_cursor_execute", reverse_unordered)
    try:
        names = [n["name"] for n in llm_memory.listing(pid)]
        got = admin_project_memory(pid, auth={"user_id": "a", "role": "admin"})
    finally:
        if sqlite:
            event.remove(engine, "before_cursor_execute", reverse_unordered)
            engine.dispose()
    assert names == listed
    assert [n["name"] for n in got["notes"]] == listed
    assert [n["data"] for n in got["notes"]] == [{"n": name} for name in listed]


@pytest.mark.parametrize("doc,size", [({"k": "خرسانة"}, 20), ({"k": "béton armé"}, 20)])
def test_a_notes_size_is_its_utf8_byte_count_and_the_text_reads_back_unchanged(doc, size):
    pid = f"mem-utf-{uuid.uuid4().hex[:8]}"
    assert llm_memory.write(pid, "word", json.dumps(doc), "t#1")["bytes"] == size
    got = llm_memory.read(pid, "word")
    assert got["bytes"] == size and got["data"] == doc
