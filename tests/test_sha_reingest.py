"""R3: ingest refuses a duplicate sha256 unless --reingest <old_id>."""
from __future__ import annotations

import hashlib
import importlib

import pytest


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    return importlib.reload(projects), users_mod


def _project(projects, users):
    users.ensure_user_exists("u1")
    projects.create_project(name="P", client="C", user_id="u1")
    return projects.list_projects("u1")[0]


def test_add_document_refuses_duplicate_sha_without_flag(monkeypatch, tmp_path):
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    proj = _project(projects, users)
    sha = hashlib.sha256(b"same-bytes-letter").hexdigest()
    first = projects.add_document(
        project_id=proj["id"],
        original_name="a.docx",
        stored_as="a.docx",
        file_path="/tmp/a.docx",
        size=139671,
        content_sha256=sha,
    )
    with pytest.raises(projects.DuplicateContentError) as exc:
        projects.add_document(
            project_id=proj["id"],
            original_name="b.docx",
            stored_as="b.docx",
            file_path="/tmp/b.docx",
            size=139671,
            content_sha256=sha,
        )
    assert exc.value.existing_id == first["id"]
    assert len(projects.list_documents(proj["id"])) == 1


def test_reingest_flag_supersedes_old_row(monkeypatch, tmp_path):
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    proj = _project(projects, users)
    sha = hashlib.sha256(b"same-bytes-letter").hexdigest()
    old = projects.add_document(
        project_id=proj["id"],
        original_name="old.docx",
        stored_as="old.docx",
        file_path="/tmp/old.docx",
        size=139671,
        content_sha256=sha,
    )
    new = projects.add_document(
        project_id=proj["id"],
        original_name="new.docx",
        stored_as="new.docx",
        file_path="/tmp/new.docx",
        size=139671,
        content_sha256=sha,
        reingest_of=old["id"],
    )
    old_after = projects.get_document(old["id"])
    assert old_after is not None, "hard delete is forbidden"
    assert old_after["superseded_by"] == new["id"]
    assert old_after["retrieval_visible"] is False
    assert new["retrieval_visible"] is True
    assert new["id"] != old["id"]
    assert len(projects.list_documents(proj["id"])) == 2


def test_p1b_ingest_file_refuses_duplicate_sha(monkeypatch, tmp_path):
    """_ingest_file returns DUPLICATE_SHA when sha exists and no --reingest."""
    from scripts.p1b_ingest_drive_server import _ingest_file

    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    proj = _project(projects, users)
    body = b"letter-bytes-for-sha"
    sha = hashlib.sha256(body).hexdigest()
    projects.add_document(
        project_id=proj["id"],
        original_name="letter.docx",
        stored_as="letter.docx",
        file_path=str(tmp_path / "letter.docx"),
        size=len(body),
        content_sha256=sha,
    )

    class _Drive:
        def download_file_bytes(self, _fid):
            return body, None


    _rel, result = _ingest_file(
        {"id": "drive1", "name": "letter.docx", "mimeType": "x", "size": str(len(body))},
        proj["id"],
        tmp_path,
        "run1",
        _Drive(),
    )
    assert result["error"] == "DUPLICATE_SHA"
    assert result["existing_id"]
    assert len(projects.list_documents(proj["id"])) == 1


def test_scoped_reingest_ignores_unrelated_sha(monkeypatch, tmp_path):
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    proj = _project(projects, users)
    old = projects.add_document(
        project_id=proj["id"],
        original_name="old.docx",
        stored_as="old.docx",
        file_path="/tmp/old.docx",
        size=10,
        content_sha256=hashlib.sha256(b"old-bytes").hexdigest(),
    )
    other_sha = hashlib.sha256(b"other-bytes").hexdigest()
    assert projects.scoped_reingest_of(old["id"], other_sha) is None
    assert projects.scoped_reingest_of(
        old["id"], old["content_sha256"],
    ) == old["id"]


def _add(projects, pid, name, payload, *, days_ago=0):
    from datetime import datetime, timedelta, timezone

    from app.core.db import SessionLocal
    from app.core.models import Document

    doc = projects.add_document(
        project_id=pid,
        original_name=name,
        stored_as=name,
        file_path=f"/tmp/{name}",
        size=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
    )
    if days_ago:
        with SessionLocal() as session:
            row = session.get(Document, doc["id"])
            row.uploaded_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
            session.commit()
    return doc


def test_older_copy_of_the_same_document_is_superseded_by_the_newer(monkeypatch, tmp_path):
    """A corrected copy uploaded under the same name hides the older extract.

    Structural, from document metadata: same project, same name once case,
    spacing and copy markers ("(1)", "- Copy") are ignored -> the newest upload
    is live and every older one points at it. Dry-run unless ``apply``."""
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    pid = _project(projects, users)["id"]
    stale = _add(projects, pid, "Example Letter.docx", b"stale extract", days_ago=3)
    live = _add(projects, pid, "example letter (1).docx", b"corrected extract")
    other = _add(projects, pid, "Unrelated Memo.docx", b"memo", days_ago=5)

    plan = projects.supersede_older_duplicates(pid)
    assert plan["applied"] is False
    assert plan["pairs"] == [{"stale_id": stale["id"], "live_id": live["id"]}]
    assert projects.get_document(stale["id"])["retrieval_visible"] is True

    done = projects.supersede_older_duplicates(pid, apply=True)
    assert done["applied"] is True
    assert projects.get_document(stale["id"])["retrieval_visible"] is False
    assert projects.get_document(stale["id"])["superseded_by"] == live["id"]
    assert projects.get_document(live["id"])["retrieval_visible"] is True
    assert projects.get_document(other["id"])["retrieval_visible"] is True
    # Idempotent: nothing left to supersede.
    assert projects.supersede_older_duplicates(pid, apply=True)["pairs"] == []


def test_supersede_older_duplicates_is_a_noop_on_distinct_names(monkeypatch, tmp_path):
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    pid = _project(projects, users)["id"]
    _add(projects, pid, "Letter A.docx", b"a")
    _add(projects, pid, "Letter B.docx", b"b")
    assert projects.supersede_older_duplicates(pid, apply=True) == {
        "applied": True, "project_id": pid, "pairs": [],
    }
