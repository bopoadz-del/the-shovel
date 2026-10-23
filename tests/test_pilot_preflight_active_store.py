"""pilot-preflight must report the LIVE vector store, never the retired table.

The bug this locks: `/v1/admin/debug/pilot-preflight` read the embedding
column width off the legacy `chunks` table and compared it to a hardcoded
`vector(256)`. Since the v2 migration, `chunks` is empty and every write
goes to the namespaced `chunks_v2`. So the operator's readiness gate
answered `embedding_dim_ok: True` from a dead table — it would have stayed
green with the live store missing, empty, or the wrong width.

Verified against live prod 2026-08-02: `chunks` = vector(256), 0 rows,
while retrieval served answers the whole session from `chunks_v2`.

Planted truth: the two tables are given DIFFERENT widths, so any code that
reads the legacy table reports 256 and any code that reads the active table
reports 384. Test 1 therefore fails on the old implementation.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import require_api_key
from app.routers import admin as admin_mod

ACTIVE_TABLE = "chunks_v2"
ACTIVE_TYPE = "vector(384)"
LEGACY_TYPE = "vector(256)"
ACTIVE_ROWS = 12345


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _admin_auth():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "system",
        "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _Conn:
    """Fake connection: answers the two query shapes the preflight issues."""

    def __init__(self, types: dict, counts: dict):
        self._types = types
        self._counts = counts

    def execute(self, statement, params=None):
        sql = str(statement)
        if params and "table" in params:
            # the format_type(...) column-width probe
            return _Result(self._types.get(params["table"]))
        if "COUNT(*)" in sql:
            table = sql.rsplit("FROM", 1)[1].strip()
            if table not in self._counts:
                raise RuntimeError(f"relation {table} does not exist")
            return _Result(self._counts[table])
        raise AssertionError(f"unexpected SQL in preflight: {sql}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Engine:
    def __init__(self, conn):
        self._conn = conn

    def connect(self):
        return self._conn


def _install(monkeypatch, *, types, counts, identity):
    """Plant the DB shape the endpoint will see.

    The endpoint imports `create_engine`, `to_psycopg_url` and
    `loaded_embedder_identity` at FUNCTION scope, so each name is resolved
    from its source module at call time — patch the source modules, not
    `admin_mod`, or the patches simply do not bind.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://planted/db")
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "v2")

    import sqlalchemy
    monkeypatch.setattr(
        sqlalchemy, "create_engine", lambda *a, **k: _Engine(_Conn(types, counts))
    )
    import app.core.rag.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "loaded_embedder_identity", lambda: identity)
    import app.core.db as db_mod
    monkeypatch.setattr(db_mod, "to_psycopg_url", lambda u: u, raising=False)


BASE_COUNTS = {
    "users": 6,
    "projects": 45,
    "documents": 2844,
    "chunks": 0,
    "conversations": 920,
    "messages": 2459,
    ACTIVE_TABLE: ACTIVE_ROWS,
}


def test_preflight_reports_active_table_not_legacy(client, monkeypatch):
    """The width reported must come from chunks_v2 (384), not chunks (256)."""
    _install(
        monkeypatch,
        types={"chunks": LEGACY_TYPE, ACTIVE_TABLE: ACTIVE_TYPE},
        counts=dict(BASE_COUNTS),
        identity={"model": "BAAI/bge-small-en-v1.5", "dim": 384, "normalized": True},
    )

    pg = client.get("/v1/admin/debug/pilot-preflight").json()["postgres"]

    assert pg["active_chunk_table"] == ACTIVE_TABLE
    assert pg["active_chunk_embedding_type"] == ACTIVE_TYPE
    # The load-bearing assertion: 256 here means the legacy table was read.
    assert pg["active_chunk_embedding_dim"] == 384
    assert pg["legacy_chunks_embedding_type"] == LEGACY_TYPE
    assert pg["embedding_dim_ok"] is True
    assert pg["row_counts"][ACTIVE_TABLE] == ACTIVE_ROWS
    assert pg["row_counts"]["chunks"] == 0


def test_missing_active_table_is_not_green(client, monkeypatch):
    """Live store absent -> must NOT report ok, even though legacy is intact.

    This is the false-green the old code produced: it never looked at the
    active table, so a missing chunks_v2 still yielded True.
    """
    counts = dict(BASE_COUNTS)
    counts.pop(ACTIVE_TABLE)
    _install(
        monkeypatch,
        types={"chunks": LEGACY_TYPE},  # active table absent entirely
        counts=counts,
        identity={"model": "BAAI/bge-small-en-v1.5", "dim": 384, "normalized": True},
    )

    pg = client.get("/v1/admin/debug/pilot-preflight").json()["postgres"]

    assert pg["active_chunk_embedding_type"] is None
    assert pg["embedding_dim_ok"] is not True
    assert pg["row_counts"][ACTIVE_TABLE] is None


def test_width_mismatch_fails_the_gate(client, monkeypatch):
    """Store width disagreeing with the embedder is the corruption class."""
    _install(
        monkeypatch,
        types={"chunks": LEGACY_TYPE, ACTIVE_TABLE: "vector(256)"},
        counts=dict(BASE_COUNTS),
        identity={"model": "BAAI/bge-small-en-v1.5", "dim": 384, "normalized": True},
    )

    pg = client.get("/v1/admin/debug/pilot-preflight").json()["postgres"]

    assert pg["embedding_dim_ok"] is False
    assert pg["active_chunk_embedding_dim"] == 256
    assert pg["embedder_dim"] == 384


def test_cold_embedder_reports_unknown_not_ok(client, monkeypatch):
    """No measurement -> None. A gate must never read 'ok' from an absence."""
    _install(
        monkeypatch,
        types={"chunks": LEGACY_TYPE, ACTIVE_TABLE: ACTIVE_TYPE},
        counts=dict(BASE_COUNTS),
        identity=None,  # embedder not warm
    )

    pg = client.get("/v1/admin/debug/pilot-preflight").json()["postgres"]

    assert pg["embedder_loaded"] is False
    assert pg["embedder_dim"] is None
    assert pg["embedding_dim_ok"] is None
    # the width is still reported — unknown agreement, not unknown state
    assert pg["active_chunk_embedding_dim"] == 384


@pytest.mark.parametrize(
    "coltype,expected",
    [
        ("vector(384)", 384),
        ("vector(256)", 256),
        ("  vector( 1536 ) ", 1536),
        ("vector", None),      # unsized — cannot assert a width
        (None, None),          # column absent
        ("text", None),        # not a vector column at all
    ],
)
def test_parse_vector_dim(coltype, expected):
    assert admin_mod._parse_vector_dim(coltype) == expected
