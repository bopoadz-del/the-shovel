"""Stage 4 wiring — a user-uploaded doc routes to user_session at index time.

Flag on + doc metadata provenance="user_upload" -> chunks written with
knowledge_layer="user_session". A seed/import doc (no provenance) in a project
-> project_record. Flag off -> (None, None). Store/embedder/doc-store faked.
"""
import numpy as np
from app.core.rag import retriever as ret


class _FakeEmbedder:
    dim = 256

    def encode(self, chunks):
        return np.ones((len(chunks), self.dim), dtype=np.float32)


def _wire(monkeypatch, captured, doc):
    def fake_upsert(pid, did, chunks, emb, *, knowledge_layer=None, authority=None):
        captured.update(knowledge_layer=knowledge_layer, authority=authority)
        return len(chunks)

    class _FakeStore:
        upsert_chunks = staticmethod(fake_upsert)

    monkeypatch.setattr(ret, "available", lambda: True)
    monkeypatch.setattr(ret, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(ret, "get_store", lambda dim=None: _FakeStore())
    # patch the doc-store lookup used by _doc_name_and_provenance
    import app.core.projects as _projects
    monkeypatch.setattr(_projects, "get_document", lambda d: doc, raising=False)
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")


def test_user_upload_routes_to_user_session(monkeypatch):
    monkeypatch.setenv("RAG_LAYERED", "1")
    captured = {}
    _wire(monkeypatch, captured, {
        "original_name": "site_photos_notes.pdf",
        "metadata": {"provenance": "user_upload", "uploader_id": "u_amir"},
    })
    ret.index_chunks("example_infra_pack", "d1", ["a chunk"])
    assert captured["knowledge_layer"] == "user_session"


def test_project_import_stays_project_record(monkeypatch):
    monkeypatch.setenv("RAG_LAYERED", "1")
    captured = {}
    _wire(monkeypatch, captured, {
        "original_name": "priced_boq.xlsx",
        "metadata": None,  # seed/import: no provenance marker
    })
    ret.index_chunks("example_infra_pack", "d1", ["a chunk"])
    assert captured["knowledge_layer"] == "project_record"


def test_flag_off_ignores_provenance(monkeypatch):
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    captured = {}
    _wire(monkeypatch, captured, {
        "original_name": "x.pdf",
        "metadata": {"provenance": "user_upload"},
    })
    ret.index_chunks("example_infra_pack", "d1", ["a chunk"])
    assert captured["knowledge_layer"] is None
