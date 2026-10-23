"""Production never serves the retired legacy ``chunks`` table.

namespace "" maps to the table the store moved off. A misconfigured
RAG_VECTOR_NAMESPACE in production must fail loud at construction instead of
answering every query from an empty table. Dev and the test suite keep the
legacy namespace available for the tests that exercise it.
"""
import pytest

from app.core.rag import vector_store as vs


@pytest.mark.parametrize("env_name", ["ENV", "ENVIRONMENT"])
def test_production_refuses_the_retired_namespace(monkeypatch, tmp_path, env_name):
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv(env_name, "production")
    with pytest.raises(RuntimeError, match="retired legacy chunks table"):
        vs.VectorStore(db_path=str(tmp_path / "v.db"), namespace="")


def test_production_refuses_it_when_it_comes_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "")
    with pytest.raises(RuntimeError, match="retired legacy chunks table"):
        vs.VectorStore(db_path=str(tmp_path / "v.db"))


def test_outside_production_the_legacy_namespace_still_opens(monkeypatch, tmp_path):
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    store = vs.VectorStore(db_path=str(tmp_path / "v.db"), namespace="")
    assert store._table_name == "chunks"
