"""STREAM D3 part 2: the index stores the origin a document declares.

A knowledge copy generated from an encoded table says so on a line of its own
(``doc_typing.declared_origin``). The seed writes that origin onto the
document row, so a reader can name the table instead of calling the copy a
separate document. Every input is constructed here; nothing is copied from a
product answer or a shipped document.
"""
from __future__ import annotations

import os

import pytest

from app.core import doc_typing
from app.core import projects as projects_mod

_DECLARING = [
    ("# Copy one\n\nGenerated from `app/lib/data/table_a.json` by a script.\n\n| a | b |\n",
     "app/lib/data/table_a.json"),
    ("Title\nGenerated from `config/b/rates.yaml`; edit the source.\nbody\n",
     "config/b/rates.yaml"),
]
_SILENT = [
    "# Notes\n\nIndicative values from a published survey.\n",
    "The figures were generated from `x.json` in prose, not declared.\n",
]


@pytest.mark.parametrize("text,source", _DECLARING)
def test_a_declared_origin_is_added_to_the_row_metadata(text, source):
    given = {"drive_path": "a/b/c.md"}
    out = doc_typing.with_declared_origin(given, text)
    assert out[doc_typing.ORIGIN_METADATA_KEY] == {"kind": "encoded_table", "source": source}
    assert out["drive_path"] == "a/b/c.md"
    assert doc_typing.ORIGIN_METADATA_KEY not in given  # caller's mapping untouched
    assert doc_typing.stored_origin(out) == {"kind": "encoded_table", "source": source}


@pytest.mark.parametrize("text", _SILENT)
def test_a_document_that_declares_nothing_stores_nothing(text):
    assert doc_typing.with_declared_origin(None, text) is None
    kept = doc_typing.with_declared_origin({"k": 1}, text)
    assert kept == {"k": 1}
    assert doc_typing.stored_origin(kept) == {"kind": "document", "source": None}


@pytest.mark.parametrize("stored", [None, {}, {doc_typing.ORIGIN_METADATA_KEY: {"kind": "encoded_table"}},
                                    {doc_typing.ORIGIN_METADATA_KEY: "app/x.json"}])
def test_a_row_without_a_whole_stored_origin_speaks_for_itself(stored):
    assert doc_typing.stored_origin(stored) == {"kind": "document", "source": None}


@pytest.fixture
def seed_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    if projects_mod.get_project("training_material") is None:
        projects_mod.create_project("General Knowledge", user_id=SYSTEM_USER_ID,
                                    project_id="training_material", origin="system_seed")
    import app.core.doc_index as di
    monkeypatch.setattr(di, "index_document", lambda *a, **k: {"indexed": 1})
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    import app.core.knowledge_seed as ks
    monkeypatch.setattr(ks, "KNOWLEDGE_DIR", str(kdir))
    return kdir


def _stored(name):
    for d in projects_mod.list_documents("training_material", all_sessions=True):
        if (d.get("original_name") or d.get("name")) == name:
            row = projects_mod.get_document(d["id"])
            return doc_typing.stored_origin(row.get("metadata") or row.get("metadata_"))
    raise AssertionError(f"{name} was not seeded")


@pytest.mark.parametrize("declaring,silent", list(zip(_DECLARING, _SILENT)))
def test_the_seed_stores_each_copys_declared_origin(seed_env, declaring, silent):
    text, src = declaring
    (seed_env / "copy.md").write_text(text, encoding="utf-8")
    (seed_env / "note.md").write_text(silent, encoding="utf-8")
    from app.core.knowledge_seed import seed_knowledge
    seed_knowledge()
    assert _stored("copy.md") == {"kind": "encoded_table", "source": src}
    assert _stored("note.md") == {"kind": "document", "source": None}
