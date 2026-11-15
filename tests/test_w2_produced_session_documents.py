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

import io
import time
import uuid
import zipfile

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


@pytest.mark.parametrize("second_answer,kept", [
    (None, 1),                                        # same bytes again
    ("Pour bay 2 first; bay 1 waits for the cube results.", 2),
])
def test_the_same_output_twice_is_kept_once(client, second_answer, kept):
    headers = _register(client, f"dup{kept}")
    pid = _project(client, headers, f"W2p-dup-{kept}-{_RUN}")
    sid = f"ws-{pid}-4000"
    _answer(sid, pid, "Pour bay 1 first, then bay 3, then bay 2.")
    assert _export(client, headers, pid, sid, "xlsx").status_code == 200
    if second_answer:
        _answer(sid, pid, second_answer)
    # The second render lands in a later second, so its write time differs:
    # the same answer must still be recognised as the same output.
    time.sleep(1.1)
    assert _export(client, headers, pid, sid, "xlsx").status_code == 200

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
        pid, sid, owner_id, str(produced), "rebar_schedule.xlsx")
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


def _ooxml(parts: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in parts.items():
            zf.writestr(name, body)
    return buf.getvalue()


_CORE = "<cp:coreProperties><dcterms:modified>{}</dcterms:modified></cp:coreProperties>"


@pytest.mark.parametrize("second_sheet,second_stamp,same", [
    ("<row>bay 1</row>", "2026-01-01T00:00:09Z", True),    # only write time moved
    ("<row>bay 2</row>", "2026-01-01T00:00:00Z", False),   # the content moved
])
def test_the_produced_digest_reads_content_not_write_time(second_sheet,
                                                          second_stamp, same):
    from app.routers import exports

    first = _ooxml({"xl/worksheets/sheet1.xml": "<row>bay 1</row>",
                    "docProps/core.xml": _CORE.format("2026-01-01T00:00:00Z")})
    second = _ooxml({"xl/worksheets/sheet1.xml": second_sheet,
                     "docProps/core.xml": _CORE.format(second_stamp)})
    assert first != second
    assert (exports._produced_content_digest(first)
            == exports._produced_content_digest(second)) is same


@pytest.mark.parametrize("first,second,same", [
    (b"plain report v1", b"plain report v1", True),
    (b"plain report v1", b"plain report v2", False),
])
def test_a_non_archive_output_is_compared_byte_for_byte(first, second, same):
    from app.routers import exports

    assert (exports._produced_content_digest(first)
            == exports._produced_content_digest(second)) is same
