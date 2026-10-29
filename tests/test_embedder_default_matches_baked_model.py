"""S1 — the store width follows the resolved embedder, and a dim/model mismatch
between the live index and the embedder blocks serving.

Tests pin behaviour by DIMENSION, not by any model-name literal: an unset env
resolves to the one config default (not the legacy 256-dim fallback); the store
width equals whatever dim the resolved embedder reports; and the boot identity
check refuses to serve when the stored index disagrees with the embedder.
No model is loaded and nothing is written to a real corpus.
"""
import pytest

from app.core.rag import embeddings
from app.core.rag import vector_store as vs


class _StubEmbedder:
    def __init__(self, model: str, dim: int):
        self._identity = {"model": model, "dim": dim, "normalized": True}

    @property
    def identity(self):
        return dict(self._identity)

    @property
    def dim(self):
        return self._identity["dim"]

    @property
    def model_name(self):
        return self._identity["model"]


def test_resolve_model_name_precedence_defaults_off_the_one_config_model(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    # Unset -> the one config default, which is NOT the legacy 256-dim model2vec.
    assert embeddings._resolve_model_name(None) == embeddings.DEFAULT_EMBEDDING_MODEL
    assert embeddings.DEFAULT_EMBEDDING_MODEL != embeddings.DEFAULT_MODEL2VEC
    # There is no runtime EMBEDDING_DIM constant any more; the runtime width is
    # the live embedder's dim (see test_store_width_follows_the_resolved_embedder).
    assert not hasattr(embeddings, "EMBEDDING_DIM")
    # Env and an explicit argument take precedence, in that order.
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "x/custom-model")
    assert embeddings._resolve_model_name(None) == "x/custom-model"
    assert embeddings._resolve_model_name("fake") == "fake"


@pytest.mark.parametrize("dim", [384, 256])
def test_store_width_follows_the_resolved_embedder(tmp_path, monkeypatch, dim):
    # (a) env unset + an embedder of dimension D -> the store table width is D,
    # for two different Ds. The width is derived from the embedder, never a
    # hardcoded constant.
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(vs, "get_embedder", lambda model_name=None: _StubEmbedder("m", dim))
    store = vs.get_store(db_path=str(tmp_path / "rag.db"), namespace=f"ns{dim}")
    assert store.dim == dim


def test_boot_identity_mismatch_refuses_to_serve(monkeypatch):
    # (b) stored dim D1 vs embedder dim D2 -> boot check blocks serving.
    class _Mismatch:
        def check_identity(self):
            return {"expected": {"model": "m2", "dim": 16},
                    "stored": {"model": "m1", "dim": 8}}

    monkeypatch.setattr(vs, "get_store", lambda *a, **k: _Mismatch())
    monkeypatch.setattr(vs, "_IDENTITY_SERVING_BLOCKED", False, raising=False)
    result = vs.run_boot_identity_check()
    assert result is not None
    assert vs.identity_serving_blocked() is True


def test_boot_identity_match_serves(monkeypatch):
    class _Match:
        def check_identity(self):
            return None

    monkeypatch.setattr(vs, "get_store", lambda *a, **k: _Match())
    monkeypatch.setattr(vs, "_IDENTITY_SERVING_BLOCKED", False, raising=False)
    assert vs.run_boot_identity_check() is None
    assert vs.identity_serving_blocked() is False


def test_ready_returns_503_when_identity_serving_blocked(monkeypatch):
    # (c) The S1 identity-block flag alone must fail the readiness gate. With
    # the database healthy, a blocked embedder identity makes GET /ready return
    # 503 — driven through the REAL ready() handler reading the REAL
    # identity_serving_blocked(), so the flag is the sole cause of the 503.
    from starlette.responses import Response

    from app.routers import health

    monkeypatch.setattr(
        health, "_evaluate_health",
        lambda: {"checks": {"database": {"ok": True}}},
    )
    monkeypatch.setattr(vs, "_IDENTITY_SERVING_BLOCKED", True, raising=False)
    resp = Response()
    payload = health.ready(resp)
    assert resp.status_code == 503
    assert payload["ready"] is False


def test_real_check_identity_flags_stored_vs_embedder_dim_mismatch(tmp_path, monkeypatch):
    """(b) Drive the REAL VectorStore.check_identity: a row seeded at D1/model m1,
    then the embedder resolves to D2/model m2 -> the real SQL read + comparison
    returns the mismatch dict (no stub)."""
    import numpy as np

    monkeypatch.setattr(vs, "get_embedder", lambda model_name=None: _StubEmbedder("m1", 8))
    store = vs.get_store(db_path=str(tmp_path / "id.db"), namespace="idns")
    assert store.dim == 8
    cls = store._rag_chunk_cls
    with store._session_factory()() as s:
        s.add(cls(
            chunk_id="c1", project_id="p1", doc_id="d1", chunk_index=0, text="t",
            embedding=np.zeros(8, dtype=np.float32), created_at="2026-01-01",
            embedding_model="m1", embedding_dim=8, embedding_normalized=True,
        ))
        s.commit()
    # The embedder now reports a different identity than the stored index.
    store.model_name = "m2"
    store.dim = 16
    result = store.check_identity()
    assert result is not None
    assert result["stored"]["dim"] == 8 and result["stored"]["model"] == "m1"
    assert result["expected"]["dim"] == 16 and result["expected"]["model"] == "m2"


def test_make_rag_chunk_class_is_idempotent_across_import_order():
    """(1) One table registers ONCE on the shared MetaData, order-independent:
    even if the per-namespace cache is lost while Base.metadata persists, a
    second make_rag_chunk_class reuses the existing mapper instead of raising
    'Table ... is already defined for this MetaData instance'."""
    from app.core import models as m

    ns = "orderns"
    c1 = m.make_rag_chunk_class(ns, 8, "m1")
    m._RAG_CHUNK_CLASS_CACHE.pop(ns, None)          # simulate cache/metadata desync
    c2 = m.make_rag_chunk_class(ns, 8, "m1")        # must NOT raise
    assert c1 is c2
    assert c1.__tablename__ == m.rag_chunk_table_name(ns)
    # And the width is the requested one, derived per call — not a module constant.
    assert c1.embedding_identity["dim"] == 8
