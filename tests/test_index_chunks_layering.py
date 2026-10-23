"""Stage 2 wiring — index_chunks classifies + threads layer/authority.

Flag on: the doc's chunks are written with (knowledge_layer, authority) from
classify(). Flag off: (None, None) — today's behaviour, byte-for-byte.
Store/embedder are faked so this asserts only the wiring, no DB/model.
"""
import numpy as np
import pytest
from app.core.rag import retriever as ret


class _FakeEmbedder:
    dim = 256

    def encode(self, chunks):
        return np.ones((len(chunks), self.dim), dtype=np.float32)


def _wire(monkeypatch, captured):
    def fake_upsert(pid, did, chunks, emb, *, knowledge_layer=None, authority=None):
        captured.update(project_id=pid, knowledge_layer=knowledge_layer,
                        authority=authority)
        return len(chunks)

    class _FakeStore:
        upsert_chunks = staticmethod(fake_upsert)

    monkeypatch.setattr(ret, "available", lambda: True)
    monkeypatch.setattr(ret, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(ret, "get_store", lambda dim=None: _FakeStore())
    import app.core.projects as _projects
    monkeypatch.setattr(_projects, "get_document",
                        lambda d: {"original_name": "priced_boq_sewer.xlsx"},
                        raising=False)
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")


def test_layer_tagged_when_enabled(monkeypatch):
    monkeypatch.setenv("RAG_LAYERED", "1")
    captured = {}
    _wire(monkeypatch, captured)
    n = ret.index_chunks("example_infra_pack", "d1", ["a chunk"])
    assert n == 1
    assert captured["knowledge_layer"] == "project_record"
    assert captured["authority"] == "commercial"


def test_no_layer_when_disabled(monkeypatch):
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    captured = {}
    _wire(monkeypatch, captured)
    ret.index_chunks("example_infra_pack", "d1", ["a chunk"])
    assert captured["knowledge_layer"] is None
    assert captured["authority"] is None
