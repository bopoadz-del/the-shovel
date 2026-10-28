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


def _install_embedder(monkeypatch, model: str, dim: int):
    """Make the REAL get_embedder() resolve to ``model`` at width ``dim``.

    Uses the real Embedder in its no-model mode with the identity overridden,
    installed in the real process cache under the env-resolved name, so every
    caller (get_store, the import-time registration, models.EMBEDDING_DIM)
    resolves it through the production code path. No model is loaded.
    """
    emb = embeddings.Embedder(model_name="fake")
    emb.model_name = model
    emb._dim = dim
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", model)
    monkeypatch.setattr(embeddings, "_EMBEDDER_CACHE", emb)
    return emb


_BOOT_IN_FRESH_PROCESS = r"""
import json, os, sys
from app.core.rag import embeddings
emb = embeddings.Embedder(model_name="fake")
emb.model_name = os.environ["RAG_EMBEDDING_MODEL"]
emb._dim = int(sys.argv[1])
embeddings._EMBEDDER_CACHE = emb
from app.core.rag import vector_store as vs   # imports models -> registers ns
mismatch = vs.run_boot_identity_check()
print(json.dumps({"blocked": vs.identity_serving_blocked(), "mismatch": mismatch}))
"""


def test_boot_check_blocks_serving_when_the_index_was_built_at_another_dim(
    tmp_path, monkeypatch,
):
    """(b) A namespace indexed at D1, then a FRESH process whose embedder
    resolves to D2: the real run_boot_identity_check opens the real store, the
    real VectorStore.check_identity reads the stored D1 stamp, and serving is
    blocked. Nothing between the check and the flag is stubbed."""
    import json
    import os
    import subprocess
    import sys

    import numpy as np

    ns = "bootd1d2"
    db = tmp_path / "boot.db"
    d1 = embeddings.get_embedder("fake")
    d1_dim = d1.dim
    store = vs.get_store(db_path=str(db), namespace=ns, model_name="fake")
    store.upsert_chunks("p1", "d1", ["t"], np.zeros((1, d1_dim), dtype=np.float32))
    assert store.check_identity() is None  # the index is consistent at D1

    d2_dim = d1_dim + 128
    env = dict(os.environ)
    env.update({
        "DATABASE_URL": f"sqlite:///{db}",
        "RAG_VECTOR_NAMESPACE": ns,
        "RAG_EMBEDDING_MODEL": "emb-d2",
        "ENV": "test",
    })
    env.pop("PYTEST_USE_POSTGRES", None)
    out = subprocess.run(
        [sys.executable, "-c", _BOOT_IN_FRESH_PROCESS, str(d2_dim)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["blocked"] is True
    assert got["mismatch"]["stored"]["dim"] == d1_dim
    assert got["mismatch"]["expected"]["dim"] == d2_dim
    assert got["mismatch"]["expected"]["model"] == "emb-d2"


def test_boot_check_blocks_serving_when_the_namespace_is_mapped_at_another_dim(
    monkeypatch,
):
    """(b, in-process) The namespace is already open at D1 in this process and
    the embedder now resolves to D2: the real run_boot_identity_check -> real
    get_store hits the width guard, and that too blocks serving (it used to be
    swallowed as an "unreadable probe")."""
    monkeypatch.setattr(vs, "_IDENTITY_SERVING_BLOCKED", False, raising=False)
    d1 = embeddings.get_embedder("fake")
    vs.get_store(model_name="fake")                 # active namespace at D1
    vs.reset_store_cache()
    _install_embedder(monkeypatch, "emb-d2", d1.dim + 128)
    result = vs.run_boot_identity_check()
    assert result is not None
    assert result["stored"]["dim"] == d1.dim
    assert result["expected"]["dim"] == d1.dim + 128
    assert vs.identity_serving_blocked() is True


def test_boot_identity_match_serves(monkeypatch):
    # Same identity, empty namespace: the real check finds nothing to refuse.
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


_BUILD_IN_ORDER = r"""
import json, sys
from app.core import models as m
from app.core.rag.embeddings import get_embedder
ident = get_embedder().identity
active = m.rag_chunk_table_name(__import__("os").environ["RAG_VECTOR_NAMESPACE"])
at_import = active in m.Base.metadata.tables
plan = [("orda", 8), ("ordb", 16)]
if sys.argv[1] == "reverse":
    plan.reverse()
for ns, dim in plan:
    m.make_rag_chunk_class(ns, dim, "m-" + ns)
conflicts = []
for ns, dim in plan:
    for bad_dim, bad_model in ((dim * 2, "m-" + ns), (dim, "other")):
        try:
            m.make_rag_chunk_class(ns, bad_dim, bad_model)
            conflicts.append("reused")
        except ValueError:
            conflicts.append("raised")
tables = {}
for name in sorted(m.Base.metadata.tables):
    if name.startswith("chunks_"):
        t = m.Base.metadata.tables[name]
        tables[name] = [[c.name, str(c.type.load_dialect_impl(
            __import__("sqlalchemy.dialects.postgresql", fromlist=["dialect"]).dialect()))
            if hasattr(c.type, "load_dialect_impl") else str(c.type)] for c in t.columns]
print(json.dumps({"at_import": at_import, "conflicts": conflicts,
                  "tables": tables, "active_dim": ident["dim"]}))
"""


def test_chunk_tables_register_at_import_and_build_identically_in_either_order(
    tmp_path,
):
    """(1) Two fresh processes import the models and build two namespaces in
    opposite orders. Both must: have the configured namespace's table on
    Base.metadata straight after import (fails if lazy first-use registration
    returns), end with an identical chunk-table MetaData, and RAISE -- never
    reuse -- on a second width or model for an already-registered namespace."""
    import json
    import os
    import subprocess
    import sys

    runs = {}
    for order in ("forward", "reverse"):
        env = dict(os.environ)
        env.update({
            "DATABASE_URL": f"sqlite:///{tmp_path / (order + '.db')}",
            "RAG_VECTOR_NAMESPACE": "ordactive",
            "RAG_EMBEDDING_MODEL": "fake",
            "ENV": "test",
        })
        env.pop("PYTEST_USE_POSTGRES", None)
        out = subprocess.run(
            [sys.executable, "-c", _BUILD_IN_ORDER, order],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        runs[order] = json.loads(out.stdout.strip().splitlines()[-1])

    for got in runs.values():
        assert got["at_import"] is True
        assert got["conflicts"] == ["raised"] * 4
        assert "chunks_ordactive" in got["tables"]
    assert runs["forward"]["tables"] == runs["reverse"]["tables"]
    widths = {
        name: dict(cols)["embedding"] for name, cols in runs["forward"]["tables"].items()
    }
    assert widths["chunks_orda"] == "VECTOR(8)"
    assert widths["chunks_ordb"] == "VECTOR(16)"
    assert widths["chunks_ordactive"] == f"VECTOR({runs['forward']['active_dim']})"


@pytest.mark.parametrize("dim", [384, 16])
def test_models_embedding_dim_follows_the_resolved_embedder(monkeypatch, dim):
    """(3) models.EMBEDDING_DIM and the legacy EmbeddingVector width are derived
    from get_embedder().dim -- no hand-typed width in models.py."""
    from sqlalchemy.dialects import postgresql

    from app.core import models

    _install_embedder(monkeypatch, f"emb-{dim}", dim)
    assert models.EMBEDDING_DIM == dim
    col = models.RagChunk.__table__.c.embedding
    assert str(col.type.load_dialect_impl(postgresql.dialect())) == f"VECTOR({dim})"
