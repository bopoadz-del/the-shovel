"""G-2 (grader, #360/#371): two exports of the same output that run at the
same time keep ONE produced document, and two different outputs keep two.

The keep-once step used to read the session's documents and then insert, so
two concurrent exports could both read "not kept" and both store. Here both
exports are held after their read until the other has read too (a barrier
around the real listing), then both insert into the real store: SQLite rows
and the encrypted byte store, nothing mocked at the storage boundary. The
kept row is read back through its stored form, and the bytes the losing
export wrote are not left behind.

Every project, user, session and file here is created by the test.
"""
from __future__ import annotations

import os
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _owner(client, suffix: str) -> tuple[dict, str]:
    email = f"g2-{suffix}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    me = client.get("/v1/users/me", headers=headers).json()
    return headers, me.get("user_id") or me.get("id")


@pytest.mark.parametrize(
    "answers,kept",
    [(("pour C30 bay 1", "pour C30 bay 1"), 1), (("pour C30 bay 1", "pour C40 bay 2"), 2)],
)
def test_concurrent_exports_keep_one_document_per_output(client, tmp_path, monkeypatch, answers, kept):
    from app.core import byte_store, file_crypto
    from app.core import projects as projects_store
    from app.routers import exports

    headers, user_id = _owner(client, f"race{kept}")
    res = client.post("/v1/projects", json={"name": f"G2-race-{kept}-{_RUN}"}, headers=headers)
    assert res.status_code in (200, 201), res.text
    pid = res.json()["id"]
    sid = f"ws-{pid}-1700000000000"
    display = f"answer-{uuid.uuid4().hex[:6]}.xlsx"

    # Each export rendered its own bytes (a stamped export time differs);
    # the render source decides whether they are the same output.
    paths = []
    for i, answer in enumerate(answers):
        p = tmp_path / f"render{i}.xlsx"
        p.write_bytes(f"PK\x03\x04 {answer} exported 21:0{i} UTC".encode())
        paths.append(str(p))

    real_list = projects_store.list_session_documents
    both_read = threading.Barrier(2, timeout=20)

    def list_then_wait(*args, **kwargs):
        rows = real_list(*args, **kwargs)
        both_read.wait()
        return rows

    monkeypatch.setattr(projects_store, "list_session_documents", list_then_wait)
    results: list = [None, None]

    def export(i: int) -> None:
        results[i] = exports._save_produced_to_session(
            pid, sid, user_id, paths[i], display,
            source={"format": "xlsx", "answer": answers[i]},
        )

    threads = [threading.Thread(target=export, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    monkeypatch.undo()

    assert all(r for r in results), results
    assert len(set(results)) == kept, results
    rows = [d for d in projects_store.list_session_documents(pid, sid, user_id)
            if d["origin"] == "produced"]
    assert len(rows) == kept, [d["id"] for d in rows]
    assert {d["id"] for d in rows} == set(results)

    # The kept row reads back through the stored (encrypted) form ...
    for doc_id in set(results):
        stored = projects_store.get_document(doc_id)
        body = file_crypto.read_document(stored["file_path"])
        assert body.startswith(b"PK\x03\x04 pour C")
    # ... and the store holds one file per kept output, none left by a loser.
    root = byte_store.ensure_root()
    on_disk = [n for n in os.listdir(root) if n.endswith(f"_{display}")]
    assert len(on_disk) == kept, on_disk


def _session(client, suffix: str) -> tuple[str, str, str]:
    headers, user_id = _owner(client, suffix)
    res = client.post("/v1/projects", json={"name": f"G2-{suffix}-{_RUN}"}, headers=headers)
    assert res.status_code in (200, 201), res.text
    pid = res.json()["id"]
    return pid, f"ws-{pid}-1700000000000", user_id


def _produced(store, pid, sid, user_id) -> list[str]:
    return [d["id"] for d in store.list_session_documents(pid, sid, user_id)
            if d["origin"] == "produced"]


@pytest.mark.parametrize("earlier_answer,kept", [("pour C30 bay 1", 1), ("pour C40 bay 2", 2)])
def test_a_copy_kept_before_ids_were_derived_is_still_kept_once(client, tmp_path, earlier_answer, kept):
    """A copy kept by the earlier code has a random id; exporting the same
    output again returns it instead of storing a second copy."""
    from app.core import projects as store
    from app.routers import exports

    pid, sid, user_id = _session(client, f"legacy{kept}")
    source = {"format": "xlsx", "answer": "pour C30 bay 1"}
    earlier = store.add_document(pid, "answer.xlsx", metadata={
        "origin": "produced", "provenance": "produced", "uploader_id": user_id,
        "session_id": sid,
        "produced_source_sha256": exports._produced_source_digest(
            {"format": "xlsx", "answer": earlier_answer}),
    })["id"]
    render = tmp_path / "answer.xlsx"
    render.write_bytes(b"PK\x03\x04 pour C30 bay 1")

    got = exports._save_produced_to_session(pid, sid, user_id, str(render), "answer.xlsx", source=source)
    assert len(_produced(store, pid, sid, user_id)) == kept
    assert (got == earlier) is (kept == 1)


@pytest.mark.parametrize("taken", [False, True])
def test_a_row_already_holding_the_derived_id_does_not_stop_the_keep(client, tmp_path, taken):
    """An unrelated document that happens to hold the derived id (an 8-hex
    collision) is left alone and the copy is kept under a fresh id."""
    from app.core import projects as store
    from app.routers import exports

    pid, sid, user_id = _session(client, f"taken{int(taken)}")
    source = {"format": "docx", "answer": "pour C30 bay 1"}
    derived = exports._produced_document_id(
        pid, sid, user_id, exports._produced_source_digest(source))
    if taken:
        store.add_document(pid, "unrelated.pdf", document_id=derived)
    render = tmp_path / "answer.docx"
    render.write_bytes(b"PK\x03\x04 pour C30 bay 1")

    got = exports._save_produced_to_session(pid, sid, user_id, str(render), "answer.docx", source=source)
    assert got and _produced(store, pid, sid, user_id) == [got]
    assert (got == derived) is (not taken)
    assert store.get_document(derived) is not None
