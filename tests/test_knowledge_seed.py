"""Boot-time seeding of docs/knowledge/*.md into the RAG general-knowledge project."""
import glob
import os

import pytest

from app.core import projects as projects_mod


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    gk = os.getenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material").split(",")[0].strip()
    if gk and projects_mod.get_project(gk) is None:
        projects_mod.create_project(
            "General Knowledge",
            user_id=SYSTEM_USER_ID,
            project_id=gk,
            origin="system_seed",
        )
    return tmp_path


def test_seed_knowledge_ingests_and_is_idempotent(fresh_db, monkeypatch):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material")
    # Stub the embedding-backed indexer: record calls instead of running them.
    import app.core.doc_index as di
    calls = []
    monkeypatch.setattr(
        di, "index_document",
        lambda pid, did, **kw: (calls.append((pid, did)), {"indexed": 1})[1],
    )

    from app.core.knowledge_seed import seed_knowledge, KNOWLEDGE_DIR, _gk_project_id
    expected = len(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md")))
    assert expected >= 2, "expected the units + FIDIC knowledge docs at least"

    seed_knowledge()
    gk = _gk_project_id()
    assert projects_mod.get_project(gk) is not None, "GK project not created"
    docs = projects_mod.list_documents(gk)
    assert len(docs) == expected, f"expected {expected} docs, got {len(docs)}"
    assert len(calls) == expected, "index_document not called for each doc"
    names = {(d.get("original_name") or d.get("name")) for d in docs}
    assert "boq_units_of_measurement.md" in names
    assert "fidic_contracts_red_white.md" in names

    # Second run: idempotent by content sha -> nothing new ingested.
    calls.clear()
    seed_knowledge()
    assert len(projects_mod.list_documents(gk)) == expected
    assert calls == []


def test_seed_knowledge_disabled_when_no_gk_project(fresh_db, monkeypatch):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.knowledge_seed import seed_knowledge
    seed_knowledge()  # must be a no-op, never raise


def test_seed_knowledge_heals_system_seed_to_shared(fresh_db, monkeypatch):
    """Live curated_kb rows are origin=system_seed; boot must share them."""
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material")
    import app.core.doc_index as di
    monkeypatch.setattr(di, "index_document", lambda *a, **k: {"indexed": 1})

    from app.core.knowledge_seed import seed_knowledge

    gk = "training_material"
    before = projects_mod.get_project(gk)
    assert before is not None
    assert before["origin"] == "system_seed"
    seed_knowledge()
    after = projects_mod.get_project(gk)
    assert after["origin"] == "admin_drive_approved"
    assert after["is_approved"] is True
    # Idempotent: a second boot does not raise or change the grant.
    seed_knowledge()
    again = projects_mod.get_project(gk)
    assert again["origin"] == "admin_drive_approved"


def test_seed_knowledge_creates_gk_as_admin_drive_approved(fresh_db, monkeypatch):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "new_gk_corpus")
    import app.core.doc_index as di
    monkeypatch.setattr(di, "index_document", lambda *a, **k: {"indexed": 1})

    from app.core.knowledge_seed import seed_knowledge

    assert projects_mod.get_project("new_gk_corpus") is None
    seed_knowledge()
    created = projects_mod.get_project("new_gk_corpus")
    assert created is not None
    assert created["origin"] == "admin_drive_approved"
    assert created["is_approved"] is True
    projects_mod.delete_project("new_gk_corpus")


def test_seed_knowledge_does_not_insert_docs_without_project(fresh_db, monkeypatch):
    """Postgres FK: never add_document(training_material) if the project row is gone."""
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material")
    added = []
    monkeypatch.setattr(projects_mod, "get_project", lambda _pid: None)
    monkeypatch.setattr(
        projects_mod,
        "create_project",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("create failed")),
    )
    monkeypatch.setattr(
        projects_mod, "add_document",
        lambda *a, **k: added.append(1) or {},
    )
    from app.core.knowledge_seed import seed_knowledge
    seed_knowledge()
    assert added == []
