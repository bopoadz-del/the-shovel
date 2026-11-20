"""S1: "Ask about this document" from the sidebar list.

* A document id the client pins is an attached document of the turn: the
  model is told which document "this document" means, with no marker in
  the message, after the same access check a composer upload goes through.
* Each card carries one plain status line chosen by the status family.
* The chat pane's totals come from the server: the documents the project
  holds and how many of them search reaches.
* A stored-only document read only up to the fetch cap is named in the
  answer as read in its opening part.

Every project, user and document here is created by the test.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents import runtime as runtime_mod
from app.agents.runtime import (
    Agent,
    _note_opening_part_reads,
    _opening_part_reads,
    _tool_result_content,
)
from app.core import projects as store
from app.main import app
from tests._fixture_values import PW_06, STUB_TOKEN_02

_RUN = uuid.uuid4().hex[:8]


# ── the pinned id reaches the model as an attached document ──────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, suffix: str) -> dict:
    email = f"ask-{suffix}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _project(client, headers, name: str) -> str:
    res = client.post("/v1/projects", json={"name": name}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


def _upload(client, headers, pid: str, name: str, body: bytes) -> dict:
    res = client.post(
        f"/v1/projects/{pid}/documents",
        headers=headers,
        files={"file": (name, body, "text/plain")},
    )
    assert res.status_code in (200, 201), res.text
    return res.json()["document"]


@pytest.fixture
def seen_messages(monkeypatch):
    seen: list[list[dict]] = []

    async def _fake_llm(self, messages, api_key, project_id=None, **kwargs):
        seen.append([dict(m) for m in messages])
        return {"status": "success",
                "choice": {"message": {"content": "Read it.", "tool_calls": []}},
                "raw": {}}

    monkeypatch.setattr(Agent, "_call_llm", _fake_llm)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    return seen


def _attached_note(seen: list[list[dict]]) -> str:
    notes = [str(m.get("content") or "") for turn in seen for m in turn
             if "ATTACHED DOCUMENT(S) THIS TURN" in str(m.get("content") or "")]
    return "\n".join(notes)


@pytest.mark.parametrize("pick", [0, 1])
def test_a_pinned_id_alone_tells_the_model_which_document(client, seen_messages, pick):
    headers = _register(client, f"pin{pick}")
    pid = _project(client, headers, f"Pin-{pick}-{_RUN}")
    docs = [
        _upload(client, headers, pid, "drainage_method_statement.txt",
                b"Backfill trenches in 150 mm layers."),
        _upload(client, headers, pid, "hoarding_inspection_log.txt",
                b"Hoarding panel 14 replaced on day 3."),
    ]
    chosen, other = docs[pick], docs[1 - pick]
    res = client.post("/v1/chat/stream", headers=headers, json={
        "message": "What does this document say?",
        "project_id": pid,
        "conversation_id": f"ws-{pid}-{1000 + pick}",
        "document_ids": [chosen["id"]],
    })
    assert res.status_code == 200, res.text
    note = _attached_note(seen_messages)
    assert chosen["original_name"] in note and chosen["id"] in note
    assert other["original_name"] not in note


@pytest.mark.parametrize("name,body", [
    ("valuation_no_7.txt", b"Gross 1,250,000"),
    ("night_works_permit.txt", b"Permit valid 22:00 to 05:00"),
])
def test_a_pinned_id_from_another_users_project_is_refused(client, seen_messages, name, body):
    tag = name[:5]
    owner = _register(client, f"owner{tag}")
    stranger = _register(client, f"stranger{tag}")
    theirs = _project(client, owner, f"Theirs-{tag}-{_RUN}")
    mine = _project(client, stranger, f"Mine-{tag}-{_RUN}")
    doc = _upload(client, owner, theirs, name, body)
    res = client.post("/v1/chat/stream", headers=stranger, json={
        "message": "What does this document say?",
        "project_id": mine,
        "conversation_id": f"ws-{mine}-2000",
        "document_ids": [doc["id"]],
    })
    assert res.status_code == 404
    assert _attached_note(seen_messages) == ""


@pytest.mark.parametrize("ids,expected", [
    (["d-1"], ["d-1"]),
    (["d-2", "d-1", "d-2"], ["d-2", "d-1"]),
])
def test_client_ids_join_the_marker_documents_once(monkeypatch, ids, expected):
    from app.routers.chat import _attach_client_documents

    rows = {"d-1": {"id": "d-1", "original_name": "grout_test_cubes.txt"},
            "d-2": {"id": "d-2", "original_name": "pile_cap_schedule.txt"}}
    monkeypatch.setattr("app.core.projects.get_document", lambda did: rows.get(did))
    marker_doc = {"id": "d-9", "original_name": "from_marker.txt"}
    out = _attach_client_documents([marker_doc], ids + [None, "missing"])
    assert [d["id"] for d in out] == ["d-9"] + expected


# ── one plain status line per card ───────────────────────────────────────────

@pytest.mark.parametrize("doc,state,words", [
    ({"chunk_count": 12}, "searchable", "Searchable"),
    ({"searchable": True}, "searchable", "Searchable"),
    ({"ingest_status": "PARTIAL_OCR", "pages_ocrd": 3, "pages_in_file": 9, "chunk_count": 4},
     "searchable", "Searchable in part"),
    ({"ingest_status": "EXTRACT_FAILED"}, "failed", "Failed: Extraction returned no usable text"),
    ({"bytes_state": "missing"}, "failed", "Failed:"),
    ({"ingest_status": "UNVERIFIED"}, "reading", "Waiting to be read"),
    ({"metadata": {"indexing": {"status": "pending", "detail": "queued"}}}, "reading",
     "Waiting to be read"),
    ({"ingest_status": "NOT_INDEXED_POLICY",
      "metadata": {"indexing": {"status": "not_indexed",
                                "detail": "Only an admin adds documents."}}},
     "stored", "Stored, not searchable: Only an admin adds documents. You can still ask about it."),
    ({"ingest_status": "UNSUPPORTED_TYPE"}, "stored",
     "Stored, not searchable: This file type is not added"),
])
def test_each_status_family_gets_its_plain_line(doc, state, words):
    out = store.document_state_for(dict(doc))
    assert out["state"] == state
    assert out["line"].startswith(words)


@pytest.mark.parametrize("doc", [
    {"chunk_count": 3},
    {"ingest_status": "NOT_INDEXED_POLICY"},
])
def test_the_panel_note_and_the_line_agree(doc):
    row = store.attach_panel_status(dict(doc))
    assert row["document_state"] == store.document_state_for(dict(doc))
    assert ("panel_status" in row) == (row["document_state"]["state"] != "searchable")


# ── the totals the chat pane states ──────────────────────────────────────────

@pytest.mark.parametrize("names", [
    ("earthworks_cut_fill.txt", "superseded_drawing_list.txt", "crane_lift_plan.txt"),
    ("pump_test_record.txt", "old_access_route_plan.txt", "tower_crane_base.txt"),
])
def test_held_count_leaves_out_hidden_rows_and_the_page_does_not_set_it(names):
    tag = names[0][:5]
    pid = store.create_project(f"Totals-{tag}-{_RUN}")["id"]
    kept = store.add_document(pid, names[0], content_sha256=f"a{tag}{_RUN}")
    gone = store.add_document(pid, names[1], content_sha256=f"b{tag}{_RUN}")
    old = store.add_document(pid, names[2], content_sha256=f"c{tag}{_RUN}")
    store.add_document(pid, names[2], content_sha256=f"d{tag}{_RUN}",
                       reingest_of=old["id"])
    assert store.count_documents(pid) == 4
    assert store.count_documents(pid, visible_only=True) == 4 - 1
    store.exclude_document(gone["id"], by="owner", reason="test", at="2026-11-15T00:00:00Z")
    assert store.count_documents(pid) == 4
    assert store.count_documents(pid, visible_only=True) == 2
    assert kept["id"]


@pytest.mark.parametrize("indexed", [[], ["rebar_bending_schedule.txt"]])
def test_project_detail_states_held_and_searchable_totals(client, monkeypatch, indexed):
    headers = _register(client, f"totals{len(indexed)}")
    pid = _project(client, headers, f"Totals-api-{len(indexed)}-{_RUN}")
    names = ("formwork_striking_times.txt", "rebar_bending_schedule.txt")
    ids = {name: _upload(client, headers, pid, name, f"contents of {name}".encode())["id"]
           for name in names}

    class _Chunks:
        """The chunk store as the route reads it: indexed documents only."""
        def count(self, project_id):
            return 7 * len(indexed)

        def count_by_doc(self, project_id):
            return {ids[n]: 7 for n in indexed}

    monkeypatch.setattr("app.core.rag.vector_store.get_store", lambda *a, **k: _Chunks())
    proj = client.get(f"/v1/projects/{pid}", headers=headers).json()
    assert proj["held_document_count"] == 2
    assert proj["searchable_document_count"] == len(indexed)
    states = {d["original_name"]: d.get("document_state") for d in proj["documents"]}
    assert set(states) == set(names)
    for name, state in states.items():
        if name in indexed:
            assert state == {"state": "searchable", "line": "Searchable"}
        else:
            assert state["state"] == "stored"
            assert "You can still ask about it." in state["line"]


# ── a stored-only read cut at the fetch cap ──────────────────────────────────

def _fetch_turn(tool_result: dict) -> list[dict]:
    inner = tool_result["result"]
    return [
        {"role": "user", "content": "what is the retention period"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "fetch_document",
                         "arguments": json.dumps({"document_id": inner.get("document_id")})}}]},
        {"role": "tool", "name": "fetch_document", "tool_call_id": "c1",
         "content": _tool_result_content(inner, max_chars=8000)},
    ]


def _real_fetch(monkeypatch, name: str, text: str, chunks: list[str] | None) -> dict:
    """Run the real fetch_document tool on one invented document."""
    doc = {"id": "doc-" + name[:6], "original_name": name, "file_path": "/nowhere"}
    monkeypatch.setattr("app.core.projects.list_documents", lambda pid, **_kw: [doc])
    monkeypatch.setattr("app.core.doc_index.extract_document_text",
                        lambda file_path, filename: text)
    if chunks is None:
        monkeypatch.setattr("app.core.rag.retriever.available", lambda: False)
    else:
        class _Store:
            def doc_chunk_texts(self, pid, ids):
                return {doc["id"]: chunks}
        monkeypatch.setattr("app.core.rag.retriever.available", lambda: True)
        monkeypatch.setattr("app.core.rag.vector_store.get_store", lambda *a, **k: _Store())
    from app.agents.runtime import AGENT_REGISTRY, load_agents
    if "project-assistant" not in AGENT_REGISTRY:
        load_agents()
    agent = AGENT_REGISTRY["project-assistant"]
    call = {"function": {"name": "fetch_document", "arguments": {"document_id": doc["id"]}}}
    return asyncio.run(agent._run_tool_call(call, project_id="p-1"))


@pytest.mark.parametrize("name", ["subcontract_terms_full.txt", "site_rules_handbook.txt"])
def test_a_stored_only_read_cut_at_the_cap_is_named(monkeypatch, name):
    out = _real_fetch(monkeypatch, name, "Clause text. " * 4000, chunks=None)
    assert out["ok"] is True and out["result"]["truncated"] is True
    messages = _fetch_turn(out)
    assert _opening_part_reads(messages) == [name]
    answer = _note_opening_part_reads("Retention is 5 percent.", messages)
    assert answer.startswith("Retention is 5 percent.")
    assert f"I read only the opening part of {name}." in answer
    assert _note_opening_part_reads(answer, messages) == answer


@pytest.mark.parametrize("text,chunks", [
    ("Short clause.", None),
    ("ignored", ["Indexed clause. " * 4000]),
])
def test_a_whole_read_or_an_indexed_read_gets_no_note(monkeypatch, text, chunks):
    out = _real_fetch(monkeypatch, "retention_clause.txt", text, chunks)
    assert out["ok"] is True
    messages = _fetch_turn(out)
    assert _opening_part_reads(messages) == []
    assert _note_opening_part_reads("Retention is 5 percent.", messages) == (
        "Retention is 5 percent.")


@pytest.mark.parametrize("name", ["old_minutes.txt", "progress_report_wk3.txt"])
def test_a_read_from_an_earlier_turn_is_not_this_turns(name):
    cut = {"result": {"document_id": "d-4", "filename": name,
                      "truncated": True, "source": "extracted", "content": "x"}}
    messages = _fetch_turn(cut) + [{"role": "user", "content": "and the next question"}]
    assert _opening_part_reads(_fetch_turn(cut)) == [name]
    assert _opening_part_reads(messages) == []
    assert runtime_mod._note_opening_part_reads("", _fetch_turn(cut)) == ""


@pytest.mark.parametrize("tool", ["search_project_documents", "boq_processor"])
def test_only_a_document_fetch_counts_as_a_cut_read(tool):
    """Another tool's result can carry the same keys; it is not a read of
    the document's own text, so it never earns the note."""
    lookalike = {"document_id": "d-6", "filename": "tender_addendum_2.txt",
                 "truncated": True, "source": "extracted", "content": "x"}
    messages = _fetch_turn({"result": lookalike})
    messages[1]["tool_calls"][0]["function"]["name"] = tool
    messages[2]["name"] = tool
    assert _opening_part_reads(messages) == []
    messages[2]["name"] = "fetch_document"
    assert _opening_part_reads(messages) == ["tender_addendum_2.txt"]
