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
