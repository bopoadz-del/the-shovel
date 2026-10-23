"""Deleted projects must STAY deleted across a restart.

Regression for the production "I deleted projects and they came back" bug:
deleting a project removed the Project row but left the legacy on-disk index
(``data/doc_index/<pid>.json`` / legacy db). On the next restart,
``doc_index.init_db()`` re-imported that legacy source and
``_ensure_project_row`` resurrected the project. Plus a real row carrying the
virtual master-corpus alias id duplicated the injected alias in listings.
"""
import json
import os

from app.core import doc_index
from app.core import projects as projects_mod
from app.core.db import SessionLocal
from app.core.models import Project


def _write_legacy_json(pid: str) -> str:
    legacy_dir = doc_index._legacy_index_dir()
    os.makedirs(legacy_dir, exist_ok=True)
    path = os.path.join(legacy_dir, f"{pid}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"documents": []}, fh)
    return path


def test_purge_project_index_removes_legacy_json():
    doc_index.init_db()
    pid = "resurrect-purge-1"
    path = _write_legacy_json(pid)
    assert os.path.isfile(path)
    doc_index.purge_project_index(pid)
    assert not os.path.isfile(path)


def test_deleted_project_does_not_resurrect_on_reimport():
    doc_index.init_db()
    pid = "resurrect-reimport-1"
    _write_legacy_json(pid)
    # First import resurrects (the pre-fix behaviour).
    with SessionLocal() as s:
        doc_index._import_legacy_json_indexes(s)
        s.commit()
    with SessionLocal() as s:
        assert s.get(Project, pid) is not None
    # The fixed delete path: delete the row AND purge the re-import source.
    projects_mod.delete_project(pid)
    doc_index.purge_project_index(pid)
    # A later restart's import must NOT bring it back.
    with SessionLocal() as s:
        doc_index._import_legacy_json_indexes(s)
        s.commit()
    with SessionLocal() as s:
        assert s.get(Project, pid) is None


def test_legacy_import_skips_master_corpus_alias():
    doc_index.init_db()
    alias = projects_mod.MASTER_CORPUS_PROJECT_ID
    if alias == projects_mod.MASTER_CORPUS_SOURCE_PROJECT_ID:
        return  # aliasing disabled — the id IS a real project
    # Start from no such row: another test sharing this worker's database may
    # have left one, and the claim is that the IMPORT does not create it.
    doc_index._purge_spurious_master_corpus_row()
    with SessionLocal() as s:
        assert s.get(Project, alias) is None
    _write_legacy_json(alias)
    with SessionLocal() as s:
        doc_index._import_legacy_json_indexes(s)
        s.commit()
    # The alias is virtual; no REAL row may carry its id.
    with SessionLocal() as s:
        assert s.get(Project, alias) is None


def test_startup_removes_spurious_master_corpus_row():
    doc_index.init_db()
    alias = projects_mod.MASTER_CORPUS_PROJECT_ID
    if alias == projects_mod.MASTER_CORPUS_SOURCE_PROJECT_ID:
        return
    with SessionLocal() as s:
        if s.get(Project, alias) is None:
            s.add(Project(
                id=alias, name=alias, client=None, status="active",
                aconex_connected=False, user_id="system",
                created_at=doc_index._now(),
            ))
            s.commit()
    doc_index._purge_spurious_master_corpus_row()
    with SessionLocal() as s:
        assert s.get(Project, alias) is None
