"""doc_index storage: one text-free row per entry, and a touched document
moves one row.

Live 2026-10-03 the per-project blob was 48 MB (chunk text of every
document) and every single-document update rewrote it whole (4,479 rewrites
in 80 minutes). These tests pin the mechanism that replaced it: the blob is
a header, entries are rows holding counts (chunks_v2 is the only store of
chunk text), an unchanged entry writes nothing, the per-document path never
reads the project's other entries, and the blob->rows migration is
idempotent.
"""

import importlib

import pytest
from sqlalchemy import event, select

from app.core import doc_index, file_crypto
from app.core import projects as projects_mod
from app.core.db import get_engine
from app.core.models import DocIndex, DocIndexEntry
from tests.conftest import _postgres_test_mode

pytestmark = pytest.mark.skipif(
    _postgres_test_mode(),
    reason="SQLite-backed doc_index storage tests",
)


def _rows(project_id):
    with get_engine().connect() as conn:
        return conn.execute(
            select(
                DocIndexEntry.document_id, DocIndexEntry.kind,
                DocIndexEntry.seq, DocIndexEntry.updated_at, DocIndexEntry.entry_json,
            )
            .where(DocIndexEntry.project_id == project_id)
            .order_by(DocIndexEntry.seq)
        ).all()


def _header(project_id):
    with get_engine().connect() as conn:
        return conn.execute(
            select(DocIndex.index_json).where(DocIndex.project_id == project_id)
        ).scalar_one_or_none()


def _seed(project_id, n=3):
    def _m(current):
        return {
            "project_id": project_id,
            "built_at": "t0",
            "documents": [
                {"document_id": f"d{i}", "filename": f"{i}.txt",
                 "fingerprint": "f", "chunk_count": 1}
                for i in range(n)
            ],
            "skipped": [{"document_id": "s0", "filename": "x.bin",
                         "reason": "unsupported_type", "fingerprint": "f"}],
        }
    doc_index._update_index(project_id, _m)


class _WriteCounter:
    """Counts INSERT/UPDATE/DELETE statements the engine actually executes."""

    def __init__(self):
        self.writes = []

    def __enter__(self):
        self._eng = get_engine()
        event.listen(self._eng, "before_cursor_execute", self._hook)
        return self

    def __exit__(self, *exc):
        event.remove(self._eng, "before_cursor_execute", self._hook)

    def _hook(self, conn, cursor, statement, parameters, context, executemany):
        head = statement.lstrip()[:6].upper()
        if head in ("INSERT", "UPDATE", "DELETE"):
            self.writes.append(statement.lstrip()[:60])


def test_entries_are_rows_and_the_blob_is_a_header(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed("p-rows")

    header = _header("p-rows")
    assert header is not None
    assert "documents" not in header and "skipped" not in header
    assert header["project_id"] == "p-rows"

    rows = _rows("p-rows")
    assert [(r.document_id, r.kind) for r in rows] == [
        ("d0", "document"), ("d1", "document"), ("d2", "document"), ("s0", "skipped"),
    ]

    # The historical dict shape is still what readers get.
    loaded = doc_index._load_index("p-rows")
    assert [d["document_id"] for d in loaded["documents"]] == ["d0", "d1", "d2"]
    assert loaded["documents"][1]["chunk_count"] == 1
    assert [s["document_id"] for s in loaded["skipped"]] == ["s0"]
    assert doc_index.skipped_count("p-rows") == 1


def test_doc_index_entry_has_no_chunk_text(tmp_path, monkeypatch):
    """index_document returns the chunks to its caller; the stored entry
    carries only their count. chunks_v2 is the only store of chunk text."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    importlib.reload(doc_index)

    pid = projects_mod.create_project("Entry Shape")["id"]
    body = ("The structural survey confirmed no subsidence was detected. " * 40).encode()
    path = str(tmp_path / "survey.txt")
    file_crypto.write_document(path, body)
    doc = projects_mod.add_document(pid, "survey.txt", file_path=path, size=len(body))

    result = doc_index.index_document(pid, doc["id"])
    assert result["chunks"], result

    stored = [r for r in _rows(pid) if r.document_id == doc["id"]]
    assert len(stored) == 1
    entry = stored[0].entry_json
    assert "chunks" not in entry
    assert entry["chunk_count"] == len(result["chunks"])
    assert entry["document_id"] == doc["id"]
    assert "survey" in entry["filename"]
    # ...and the same is true of the assembled dict every reader gets.
    loaded = doc_index._load_index(pid)
    assert "chunks" not in loaded["documents"][0]
    assert loaded["documents"][0]["chunk_count"] == len(result["chunks"])


def test_update_index_that_changes_nothing_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed("p-noop")
    before = _rows("p-noop")
    header_before = _header("p-noop")

    def _same(current):
        current["built_at"] = "later"  # bookkeeping only -- not a change
        return current
    with _WriteCounter() as wc:
        doc_index._update_index("p-noop", _same)
    assert wc.writes == []
    assert _rows("p-noop") == before
    assert _header("p-noop") == header_before


def test_skip_does_not_rewrite_index(tmp_path, monkeypatch):
    """Recording the same skip again (the resume loop's case) issues zero
    UPDATE/INSERT/DELETE statements."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed("p-skip")
    skip = {"document_id": "s0", "filename": "x.bin",
            "reason": "unsupported_type", "fingerprint": "f"}
    before = _rows("p-skip")
    with _WriteCounter() as wc:
        assert doc_index._upsert_index_entry("p-skip", "s0", "skipped", skip) is False
    assert wc.writes == [], wc.writes
    assert _rows("p-skip") == before


def test_upsert_entry_touches_one_row_and_skips_an_identical_one(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed("p-one")
    before = {r.document_id: r for r in _rows("p-one")}

    same = {"document_id": "d1", "filename": "1.txt", "fingerprint": "f", "chunk_count": 1}
    assert doc_index._upsert_index_entry("p-one", "d1", "document", same) is False
    assert {r.document_id: r for r in _rows("p-one")} == before

    # A caller still handing over the old shape (chunk text) is slimmed on
    # the way in: the count changes, the text never lands.
    changed = {"document_id": "d1", "filename": "1.txt", "fingerprint": "f",
               "chunks": ["text 1", "more"]}
    with _WriteCounter() as wc:
        assert doc_index._upsert_index_entry("p-one", "d1", "document", changed) is True
    assert all("doc_index" in w for w in wc.writes), wc.writes
    assert sum(1 for w in wc.writes if "doc_index_entries" in w) == 1
    after = {r.document_id: r for r in _rows("p-one")}
    assert after["d1"].seq == before["d1"].seq  # kept its place
    assert after["d1"].entry_json == {"document_id": "d1", "filename": "1.txt",
                                      "fingerprint": "f", "chunk_count": 2}
    untouched = [k for k in before if k != "d1"]
    assert all(after[k] == before[k] for k in untouched), "other rows were rewritten"


def test_upsert_entry_moves_a_document_between_kinds(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed("p-kind")
    skip = {"document_id": "d2", "filename": "2.txt", "reason": "unsupported_type", "fingerprint": "f"}
    doc_index._upsert_index_entry("p-kind", "d2", "skipped", skip)
    loaded = doc_index._load_index("p-kind")
    assert [d["document_id"] for d in loaded["documents"]] == ["d0", "d1"]
    assert sorted(s["document_id"] for s in loaded["skipped"]) == ["d2", "s0"]
    assert doc_index.skipped_count("p-kind") == 2


def test_upsert_entry_on_a_project_without_an_index_creates_the_header(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert doc_index._load_index("p-new") is None
    entry = {"document_id": "d0", "filename": "a.txt", "fingerprint": "f", "chunk_count": 1}
    assert doc_index._upsert_index_entry("p-new", "d0", "document", entry) is True
    loaded = doc_index._load_index("p-new")
    assert loaded["project_id"] == "p-new" and loaded["built_at"]
    assert loaded["documents"] == [entry] and loaded["skipped"] == []


def test_purge_removes_entries_too(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed("p-purge")
    doc_index.purge_project_index("p-purge")
    assert doc_index._load_index("p-purge") is None
    assert _rows("p-purge") == []


def _seed_legacy_blob(project_id):
    """A pre-0018 row: header + both entry lists, chunk text included."""
    from sqlalchemy import insert

    doc_index._ensure_db()
    blob = {
        "project_id": project_id,
        "built_at": "legacy",
        "documents": [
            {"document_id": "L1", "filename": "a.txt", "fingerprint": "f",
             "chunks": ["alpha", "beta"], "rag_indexed": 2},
            {"document_id": "L2", "filename": "b.txt", "fingerprint": "f",
             "chunks": ["gamma"]},
        ],
        "skipped": [
            {"document_id": "L3", "filename": "c.dwg",
             "reason": "skipped_recoverable", "fingerprint": "f"},
        ],
    }
    with doc_index._index_txn(project_id) as conn:
        doc_index._ensure_project_row_on_conn(conn, project_id)
        conn.execute(insert(DocIndex).values(
            project_id=project_id, index_json=blob, updated_at="legacy"))


def test_migration_idempotent(tmp_path, monkeypatch):
    """Blob -> rows conversion: first run converts, second run changes zero
    rows, and no chunk text reaches a row."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed_legacy_blob("p-legacy")

    eng = get_engine()
    with eng.begin() as conn:
        assert doc_index.migrate_blob_entries(conn) == 1

    header = _header("p-legacy")
    assert "documents" not in header and "skipped" not in header
    assert header["built_at"] == "legacy"
    rows = _rows("p-legacy")
    assert [(r.document_id, r.kind) for r in rows] == [
        ("L1", "document"), ("L2", "document"), ("L3", "skipped"),
    ]
    assert all("chunks" not in r.entry_json for r in rows)
    assert rows[0].entry_json["chunk_count"] == 2 and rows[0].entry_json["rag_indexed"] == 2
    assert rows[1].entry_json["chunk_count"] == 1

    snapshot = (rows, header)
    with _WriteCounter() as wc, eng.begin() as conn:
        assert doc_index.migrate_blob_entries(conn) == 0
    assert wc.writes == [], wc.writes
    assert (_rows("p-legacy"), _header("p-legacy")) == snapshot

    loaded = doc_index._load_index("p-legacy")
    assert [d["document_id"] for d in loaded["documents"]] == ["L1", "L2"]
    assert loaded["skipped"][0]["document_id"] == "L3"
