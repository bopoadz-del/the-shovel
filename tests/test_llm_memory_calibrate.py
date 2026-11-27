"""The owner calibrates the model's own notes: keep or discard each one.

A discarded note stays in the table, marked, and the model's read_memory /
list_memory no longer return it; a kept note is still returned; every
decision is recorded with the content it was made on; code never edits a
note's content. Through the public chat route with a scripted model and the
admin routes; every test builds its own projects and notes and runs two
different inputs."""
from __future__ import annotations

import json
import uuid

import pytest
import sqlalchemy as sa

from app.core import llm_memory
from app.dependencies import require_api_key
from app.main import app
from tests.test_llm_json_memory import _driver, _project, _run_one, _user, client  # noqa: F401

NOTES = [
    ("paths", {"ask_first": "which bill", "tool_for": {"rates": "bill_rows_aggregate"}}),
    ("how-to.json", [{"kind": "schedule", "start_with": "list_project_documents"}]),
]


def _stem(name):
    return name[:-5] if name.endswith(".json") else name


class _Admin:
    """Admin auth on the routes for the length of a ``with`` block."""

    def __init__(self, user_id="owner-1"):
        self.user_id = user_id

    def __enter__(self):
        app.dependency_overrides[require_api_key] = lambda: {"user_id": self.user_id, "role": "admin"}

    def __exit__(self, *exc):
        app.dependency_overrides.pop(require_api_key, None)


def _write(client, monkeypatch, h, pid, name, data):
    wrote, _ = _run_one(client, monkeypatch, h, pid, "write_memory",
                        {"name": name, "data": json.dumps(data)})
    assert wrote["ok"] is True, wrote


def _decide(client, pid, name, decision):
    return client.post(f"/v1/admin/projects/{pid}/memory/{name}/review", json={"decision": decision})


def _admin_notes(client, pid):
    got = client.get(f"/v1/admin/projects/{pid}/memory")
    assert got.status_code == 200, got.text
    return {n["name"]: n for n in got.json()["notes"]}


@pytest.mark.parametrize("name,data", NOTES)
def test_a_discarded_note_is_no_longer_returned_to_the_model_but_stays_in_the_table(
        client, monkeypatch, name, data):
    h = _user(client, "disc")
    pid = _project(client, h, "Calibrate discard")
    _write(client, monkeypatch, h, pid, name, data)
    with _Admin():
        r = _decide(client, pid, _stem(name), "discard")
        assert r.status_code == 200, r.text
        assert r.json()["review"] == "discarded"
        notes = _admin_notes(client, pid)
    # Kept in the table, marked, content untouched.
    assert notes[_stem(name)]["review"] == "discarded"
    assert notes[_stem(name)]["data"] == data
    got, _ = _run_one(client, monkeypatch, h, pid, "read_memory", {"name": name})
    assert got["ok"] is False and "no note named" in got["result"]["error"]
    listed, _ = _run_one(client, monkeypatch, h, pid, "list_memory", {})
    assert listed["result"]["notes"] == []


@pytest.mark.parametrize("name,data", NOTES)
def test_a_kept_note_is_still_returned_to_the_model(client, monkeypatch, name, data):
    h = _user(client, "keep")
    pid = _project(client, h, "Calibrate keep")
    _write(client, monkeypatch, h, pid, name, data)
    with _Admin():
        r = _decide(client, pid, name, "keep")
        assert r.status_code == 200, r.text
        assert _admin_notes(client, pid)[_stem(name)]["review"] == "kept"
    got, _ = _run_one(client, monkeypatch, h, pid, "read_memory", {"name": name})
    assert got["ok"] is True and got["result"]["data"] == data
    listed, _ = _run_one(client, monkeypatch, h, pid, "list_memory", {})
    assert [n["name"] for n in listed["result"]["notes"]] == [_stem(name)]


@pytest.mark.parametrize("name,data", NOTES)
def test_every_decision_is_recorded_with_the_content_it_was_made_on(client, monkeypatch, name, data):
    h = _user(client, "log")
    pid = _project(client, h, "Calibrate log")
    _write(client, monkeypatch, h, pid, name, data)
    with _Admin("owner-a"):
        assert _decide(client, pid, _stem(name), "keep").status_code == 200
    with _Admin("owner-b"):
        assert _decide(client, pid, _stem(name), "discard").status_code == 200
    # The model writes the note again: a new version, unmarked, readable.
    newer = {"replaced": data}
    _write(client, monkeypatch, h, pid, name, newer)
    got, _ = _run_one(client, monkeypatch, h, pid, "read_memory", {"name": name})
    assert got["ok"] is True and got["result"]["data"] == newer
    with _Admin():
        assert _admin_notes(client, pid)[_stem(name)]["review"] == ""
        log = client.get(f"/v1/admin/projects/{pid}/memory/reviews").json()
    assert log["count"] == 2
    assert [(e["decision"], e["decided_by"]) for e in log["reviews"]] == [
        ("keep", "owner-a"), ("discard", "owner-b")]
    # The record still holds what was thrown away, not the newer version.
    assert all(e["data"] == data for e in log["reviews"])
    assert all(e["note_written_by_turn"].endswith("#1") for e in log["reviews"])


@pytest.mark.parametrize("path,body", [
    ("memory/paths/review", {"decision": "discard"}),
    ("memory/reviews", None),
])
def test_only_an_admin_can_review_or_read_the_decisions(client, monkeypatch, path, body):
    h = _user(client, "nonadm")
    pid = _project(client, h, "Calibrate auth")
    _write(client, monkeypatch, h, pid, "paths", {"a": 1})
    url = f"/v1/admin/projects/{pid}/{path}"
    r = client.post(url, json=body, headers=h) if body else client.get(url, headers=h)
    assert r.status_code == 403
    # The project's owner was refused: the note is still the model's.
    got, _ = _run_one(client, monkeypatch, h, pid, "read_memory", {"name": "paths"})
    assert got["ok"] is True


@pytest.mark.parametrize("decision", ["delete", "Keep", "", "edit"])
def test_a_decision_other_than_keep_or_discard_is_refused_and_leaves_no_record(
        client, monkeypatch, decision):
    h = _user(client, "baddec")
    pid = _project(client, h, "Calibrate bad decision")
    _write(client, monkeypatch, h, pid, "paths", {"a": decision})
    with _Admin():
        assert _decide(client, pid, "paths", decision).status_code == 422
        assert _admin_notes(client, pid)["paths"]["review"] == ""
        assert client.get(f"/v1/admin/projects/{pid}/memory/reviews").json()["count"] == 0


@pytest.mark.parametrize("name,status", [
    ("never-written", 404),
    ("bad name!", 422),
    ("x" * 65, 422),
])
def test_an_unknown_or_badly_named_note_cannot_be_reviewed(client, monkeypatch, name, status):
    h = _user(client, "unk")
    pid = _project(client, h, "Calibrate unknown")
    _write(client, monkeypatch, h, pid, "paths", {"a": 1})
    with _Admin():
        assert _decide(client, pid, name, "discard").status_code == status
        assert client.get(f"/v1/admin/projects/{pid}/memory/reviews").json()["count"] == 0


@pytest.mark.parametrize("name,data", NOTES)
def test_a_decision_on_one_project_never_marks_another_projects_note(client, monkeypatch, name, data):
    h = _user(client, "cross")
    mine, other = _project(client, h, "Calibrate A"), _project(client, h, "Calibrate B")
    _write(client, monkeypatch, h, mine, name, data)
    _write(client, monkeypatch, h, other, name, data)
    with _Admin():
        assert _decide(client, mine, _stem(name), "discard").status_code == 200
        assert _admin_notes(client, other)[_stem(name)]["review"] == ""
        assert client.get(f"/v1/admin/projects/{other}/memory/reviews").json()["count"] == 0
    got, _ = _run_one(client, monkeypatch, h, other, "read_memory", {"name": name})
    assert got["ok"] is True and got["result"]["data"] == data


@pytest.mark.parametrize("limit", [1, 2])
def test_a_discarded_note_does_not_take_a_place_under_the_count_limit(client, monkeypatch, limit):
    monkeypatch.setenv("LLM_MEMORY_MAX_DOCS", str(limit))
    h = _user(client, "cap")
    pid = _project(client, h, "Calibrate cap")
    for i in range(limit):
        _write(client, monkeypatch, h, pid, f"n{i}", {"i": i})
    # Full: one more is refused.
    over, _ = _run_one(client, monkeypatch, h, pid, "write_memory",
                       {"name": "extra", "data": json.dumps({"x": 1})})
    assert over["ok"] is False and "limit" in over["result"]["error"]
    with _Admin():
        assert _decide(client, pid, "n0", "discard").status_code == 200
    # A kept note still takes its place; a discarded one does not.
    _write(client, monkeypatch, h, pid, "extra", {"x": limit})
    with _Admin():
        notes = _admin_notes(client, pid)
    assert notes["n0"]["review"] == "discarded" and notes["extra"]["data"] == {"x": limit}


@pytest.mark.parametrize("name,data", NOTES)
def test_the_store_marks_and_records_without_touching_the_content(name, data):
    pid = f"cal-{uuid.uuid4().hex[:8]}"
    llm_memory.write(pid, name, json.dumps(data), "c#3")
    before = llm_memory.listing(pid, with_data=True, with_review=True)[0]
    done = llm_memory.review(pid, name, "discard", by="u1")
    assert done == {"name": _stem(name), "decision": "discard", "review": "discarded",
                    "decided_at": done["decided_at"], "decided_by": "u1"}
    after = llm_memory.listing(pid, with_data=True, with_review=True)[0]
    assert after["data"] == before["data"] == data
    assert after["written_at"] == before["written_at"]
    assert (after["review"], after["reviewed_by"]) == ("discarded", "u1")
    assert llm_memory.listing(pid) == []
    with pytest.raises(llm_memory.MemoryRefused):
        llm_memory.read(pid, name)
    llm_memory.review(pid, name, "keep", by="u2")
    assert llm_memory.read(pid, name)["data"] == data
    assert [r["decision"] for r in llm_memory.reviews(pid)] == ["discard", "keep"]


@pytest.mark.parametrize("pid", ["", None])
def test_a_review_without_a_project_is_refused(pid):
    with pytest.raises(llm_memory.MemoryRefused):
        llm_memory.review(pid, "paths", "keep")


@pytest.mark.parametrize("present", [(), ("review",), ("review", "reviewed_at", "reviewed_by")])
def test_init_db_adds_the_mark_columns_a_table_lacks(monkeypatch, tmp_path, present):
    """A database whose llm_memory table predates the mark gets the columns
    it lacks, and its notes read as not reviewed."""
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = sa.create_engine(url)
    extra = "".join(f", {c} VARCHAR NOT NULL DEFAULT ''" for c in present)
    with eng.begin() as conn:
        conn.execute(sa.text(
            "CREATE TABLE llm_memory (project_id VARCHAR NOT NULL, name VARCHAR NOT NULL, "
            "data TEXT NOT NULL, size_bytes INTEGER NOT NULL, written_at VARCHAR NOT NULL, "
            f"written_by_turn VARCHAR NOT NULL DEFAULT ''{extra}, PRIMARY KEY (project_id, name))"))
        conn.execute(sa.text("INSERT INTO llm_memory (project_id, name, data, size_bytes, written_at) "
                             "VALUES ('p', 'old', '{\"a\": 1}', 8, '2026-11-01T00:00:00+00:00')"))
    from sqlalchemy.orm import sessionmaker
    monkeypatch.setattr(llm_memory, "engine", eng)
    monkeypatch.setattr(llm_memory, "SessionLocal", sessionmaker(bind=eng))
    monkeypatch.setattr(llm_memory, "get_database_url", lambda: url)
    monkeypatch.setattr(llm_memory, "_initialized_for_url", None)
    llm_memory.init_db()
    cols = {c["name"] for c in sa.inspect(eng).get_columns("llm_memory")}
    assert {"review", "reviewed_at", "reviewed_by"} <= cols
    assert "llm_memory_review" in sa.inspect(eng).get_table_names()
    assert llm_memory.listing("p", with_review=True)[0]["review"] == ""
    assert llm_memory.read("p", "old")["data"] == {"a": 1}


@pytest.mark.parametrize("name,data", NOTES)
def test_the_review_route_functions_mark_record_and_refuse(name, data):
    """The two admin route functions called directly: a decision returns the
    mark it left; the record lists it; a bad decision and an unknown note
    raise 422 / 404; a non-admin is refused."""
    from fastapi import HTTPException

    from app.routers.admin import (MemoryReviewBody, admin_project_memory_reviews,
                                   admin_review_project_memory)
    pid = f"cal-route-{uuid.uuid4().hex[:8]}"
    llm_memory.write(pid, name, json.dumps(data), "c#2")
    admin = {"user_id": "boss", "role": "admin"}
    done = admin_review_project_memory(pid, name, MemoryReviewBody(decision="discard"), auth=admin)
    assert (done["name"], done["review"], done["decided_by"]) == (_stem(name), "discarded", "boss")
    log = admin_project_memory_reviews(pid, auth=admin)
    assert log["project_id"] == pid and log["count"] == 1
    assert log["reviews"][0]["data"] == data
    for decision, target, status in (("drop", name, 422), ("keep", "absent-note", 404)):
        with pytest.raises(HTTPException) as refused:
            admin_review_project_memory(pid, target, MemoryReviewBody(decision=decision), auth=admin)
        assert refused.value.status_code == status
    for call in (lambda a: admin_review_project_memory(pid, name, MemoryReviewBody(decision="keep"), auth=a),
                 lambda a: admin_project_memory_reviews(pid, auth=a)):
        with pytest.raises(HTTPException) as refused:
            call({"user_id": "u", "role": "user"})
        assert refused.value.status_code == 403
    assert admin_project_memory_reviews(pid, auth=admin)["count"] == 1
