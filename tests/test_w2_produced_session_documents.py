"""W2 (exam 11.4, backend half): a file the product produces in a chat
session is kept as that session's document.

* Exporting a session's answer keeps the file under that session, tagged
  ``produced``, and under no other session.
* The kept file is stored and askable by pin, never indexed: the product's
  own output is not evidence for a later answer.
* The same bytes produced twice in one session are kept once; different
  output is kept as a second file.
* A storage failure never breaks the download.
* The kept file downloads again byte for byte.

Every project, user, session and answer here is created by the test.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, suffix: str) -> dict:
    email = f"w2p-{suffix}-{_RUN}@x.com"
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


def _answer(cid: str, pid: str, text: str) -> None:
    from app.core import agent_memory

    agent_memory.get_or_create_conversation(cid, "project_assistant", pid)
    agent_memory.append_message(cid, "user", "Summarise the pour sequence.")
    agent_memory.append_message(cid, "assistant", text)


def _export(client, headers, pid, cid, fmt):
    # Addressed by route name so the request provably reaches the handler
    # under test (exports.export_conversation_message), not a look-alike path.
    path = app.url_path_for(
        "export_conversation_message", project_id=pid, conversation_id=cid,
    )
    return client.post(f"{path}?format={fmt}", headers=headers)


def _session_docs(client, headers, pid, sid):
    res = client.get(
        f"/v1/projects/{pid}/conversations/{sid}/documents", headers=headers,
    )
    assert res.status_code == 200, res.text
    return res.json()["documents"]


@pytest.mark.parametrize("fmt", ["docx", "xlsx"])
def test_an_exported_answer_is_kept_under_its_session_tagged_produced(client, fmt):
    headers = _register(client, f"keep{fmt}")
    pid = _project(client, headers, f"W2p-keep-{fmt}-{_RUN}")
    mine, other = f"ws-{pid}-1000", f"ws-{pid}-2000"
    _answer(mine, pid, "Pour bay 1 first, then bay 3, then bay 2.")

    res = _export(client, headers, pid, mine, fmt)
    assert res.status_code == 200, res.text

    listed = _session_docs(client, headers, pid, mine)
    assert len(listed) == 1, listed
    assert listed[0]["origin"] == "produced"
    assert listed[0]["original_name"].endswith(f".{fmt}")
    assert listed[0]["original_name"] in res.headers["content-disposition"]
    assert _session_docs(client, headers, pid, other) == []


@pytest.mark.parametrize("fmt", ["docx", "xlsx"])
def test_a_produced_file_is_stored_never_indexed(client, monkeypatch, fmt):
    from app.core import doc_index, ingest_status as ist
    from app.core import projects as projects_store

    queued: list = []
    monkeypatch.setattr(doc_index, "maybe_eager_index",
                        lambda *a, **k: queued.append(a))
    headers = _register(client, f"noidx{fmt}")
    pid = _project(client, headers, f"W2p-noidx-{fmt}-{_RUN}")
    sid = f"ws-{pid}-3000"
    _answer(sid, pid, "Strike soffit forms after seven days.")

    assert _export(client, headers, pid, sid, fmt).status_code == 200
    (listed,) = _session_docs(client, headers, pid, sid)
    row = projects_store.get_document(listed["id"])
    assert queued == []
    assert row["ingest_status"] == ist.NOT_INDEXED_POLICY
    assert row["ingest_status_reason"] == ist.SESSION_ONLY_POLICY_REASON
    assert int(row.get("chunk_count") or 0) == 0
    assert listed["metadata"]["provenance"] == "produced"


class _MinuteApartClock(datetime):
    """Every ``now()`` is two minutes after the previous one, so each render
    stamps a different export time into the file it writes."""

    calls = 0

    @classmethod
    def now(cls, tz=None):
        cls.calls += 1
        return datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc) + timedelta(
            minutes=2 * cls.calls)


@pytest.mark.parametrize("second_answer,kept", [
    (None, 1),                                        # same answer again
    ("Pour bay 2 first; bay 1 waits for the cube results.", 2),
])
def test_the_same_output_twice_is_kept_once(client, monkeypatch,
                                            second_answer, kept):
    from app.routers import exports

    monkeypatch.setattr(exports, "datetime", _MinuteApartClock)
    headers = _register(client, f"dup{kept}")
    pid = _project(client, headers, f"W2p-dup-{kept}-{_RUN}")
    sid = f"ws-{pid}-4000"
    _answer(sid, pid, "Pour bay 1 first, then bay 3, then bay 2.")
    first = _export(client, headers, pid, sid, "xlsx")
    assert first.status_code == 200, first.text
    if second_answer:
        _answer(sid, pid, second_answer)
    second = _export(client, headers, pid, sid, "xlsx")
    assert second.status_code == 200, second.text

    # The two files differ (each carries its own export time), yet the same
    # answer rendered again is the same output and is kept once.
    assert first.content != second.content
    assert len(_session_docs(client, headers, pid, sid)) == kept


@pytest.mark.parametrize("store_fails,kept", [(False, 1), (True, 0)])
def test_a_storage_failure_never_breaks_the_download(client, monkeypatch,
                                                     store_fails, kept):
    from app.core import byte_store

    if store_fails:
        def _refuse(*_a, **_k):
            raise byte_store.StoreUnavailable("store is down")
        monkeypatch.setattr(byte_store, "write_bytes", _refuse)
    headers = _register(client, f"fail{int(store_fails)}")
    pid = _project(client, headers, f"W2p-fail-{int(store_fails)}-{_RUN}")
    sid = f"ws-{pid}-5000"
    _answer(sid, pid, "Backfill in 200 mm layers.")

    res = _export(client, headers, pid, sid, "docx")
    assert res.status_code == 200, res.text
    assert res.content[:2] == b"PK"
    assert len(_session_docs(client, headers, pid, sid)) == kept


@pytest.mark.parametrize("fmt", ["docx", "xlsx"])
def test_the_kept_file_downloads_again_byte_for_byte(client, fmt):
    headers = _register(client, f"dl{fmt}")
    pid = _project(client, headers, f"W2p-dl-{fmt}-{_RUN}")
    sid = f"ws-{pid}-6000"
    _answer(sid, pid, "Cure the slab for seven days under hessian.")

    res = _export(client, headers, pid, sid, fmt)
    (listed,) = _session_docs(client, headers, pid, sid)
    again = client.get(
        f"/v1/projects/{pid}/documents/{listed['id']}/download", headers=headers,
    )
    assert again.status_code == 200, again.text
    assert again.content == res.content


@pytest.mark.parametrize("who,count", [("owner", 1), ("stranger", 0)])
def test_save_produced_to_session_is_keyed_on_session_and_user(
        client, tmp_path, who, count):
    from app.routers import exports

    headers = _register(client, f"key{who}")
    pid = _project(client, headers, f"W2p-key-{who}-{_RUN}")
    sid = f"ws-{pid}-7000"
    owner = client.get("/v1/users/me", headers=headers).json()
    owner_id = owner.get("user_id") or owner.get("id")
    produced = tmp_path / "rebar_schedule.xlsx"
    produced.write_bytes(b"PK\x03\x04 bar marks 01 to 12")

    doc_id = exports._save_produced_to_session(
        pid, sid, owner_id, str(produced), "rebar_schedule.xlsx",
        source={"render": "schedule_workbook", "staged": {"marks": [1, 12]}})
    assert doc_id

    reader = headers if who == "owner" else _register(client, f"key{who}2")
    if who == "stranger":
        # A stranger cannot open the project's session at all.
        res = client.get(
            f"/v1/projects/{pid}/conversations/{sid}/documents", headers=reader,
        )
        assert res.status_code == 404
        listed = []
    else:
        listed = _session_docs(client, reader, pid, sid)
    assert len([d for d in listed if d["origin"] == "produced"]) == count


def _staged_wbs(names: list[str]) -> dict:
    return {
        "status": "success",
        "brief": "Programme for a two-storey clinic.",
        "activities": [
            {"id": f"A{i + 1}", "name": n, "duration_days": 3 + i,
             "predecessors": [f"A{i}"] if i else []}
            for i, n in enumerate(names)
        ],
        "start_date": "2027-02-01",
    }


@pytest.mark.parametrize("staged,status,kept", [
    (["Excavate pad footings", "Blind and fix rebar", "Pour footings"], 200, 1),
    (None, 404, 0),                                   # nothing staged
])
def test_a_sessions_schedule_workbook_is_kept_as_produced(client, staged,
                                                          status, kept):
    from app.core.conversation_wbs import stage_conversation_wbs

    headers = _register(client, f"wbs{kept}")
    pid = _project(client, headers, f"W2p-wbs-{kept}-{_RUN}")
    sid = f"ws-{pid}-8000"
    if staged:
        assert stage_conversation_wbs(sid, _staged_wbs(staged))

    res = client.post(
        app.url_path_for(
            "export_conversation_schedule", project_id=pid, conversation_id=sid,
        ),
        headers=headers,
    )
    assert res.status_code == status, res.text
    listed = _session_docs(client, headers, pid, sid)
    assert [d["origin"] for d in listed] == ["produced"] * kept
    if kept:
        assert listed[0]["original_name"].endswith("_schedule.xlsx")


@pytest.mark.parametrize("user_turns", [0, 1])
def test_a_session_with_no_answer_produces_and_keeps_nothing(client, user_turns):
    """Nothing to export -> 404, and no Produced document is kept for the session."""
    from app.core import agent_memory

    headers = _register(client, f"none{user_turns}")
    pid = _project(client, headers, f"W2p-none-{user_turns}-{_RUN}")
    sid = f"ws-{pid}-9000"
    agent_memory.get_or_create_conversation(sid, "project_assistant", pid)
    for _ in range(user_turns):
        agent_memory.append_message(sid, "user", "Summarise the pour sequence.")

    res = _export(client, headers, pid, sid, "docx")
    assert res.status_code == 404, res.text
    # The reason names the missing answer, not an index error from a later step.
    detail = res.json()["detail"].lower()
    assert "no assistant messages" in detail and "out of range" not in detail
    assert _session_docs(client, headers, pid, sid) == []



_ASKED = {"render": "message", "format": "xlsx", "project": "Tower A",
          "answer": "Pour bay 1 first, then bay 3, then bay 2."}


@pytest.mark.parametrize("second,same", [
    # the same inputs, written in another key order
    ({"answer": "Pour bay 1 first, then bay 3, then bay 2.", "project": "Tower A",
      "format": "xlsx", "render": "message"}, True),
    # one input moved: another answer
    ({**_ASKED, "answer": "Pour bay 2 first."}, False),
])
def test_a_produced_file_is_identified_by_what_it_was_rendered_from(second, same):
    from app.routers import exports

    assert (exports._produced_source_digest(_ASKED)
            == exports._produced_source_digest(second)) is same


@pytest.mark.parametrize("fmt_b,same", [("xlsx", True), ("docx", False)])
def test_the_same_answer_in_another_format_is_another_output(fmt_b, same):
    from app.routers import exports

    assert (exports._produced_source_digest(_ASKED)
            == exports._produced_source_digest({**_ASKED, "format": fmt_b})) is same
