"""Extractor-family canary (E-1 of the corpus order).

The live defect: extractor family ``8535199-sdt`` terminally closed ~3,300
documents as ``attempts_exhausted@8535199-sdt:1:terminal`` on their FIRST
failure -- before that family had ever extracted one document successfully.
Six plain ``.txt`` files sit in the closed class, and a ``.txt`` cannot fail
text extraction, so the closure was systemic, not per-file: nothing gated
the family.

The bound (``INGEST_MAX_ATTEMPTS``, default 1) is correct and unchanged --
identical bytes through an identical extractor give an identical outcome.
The defect is that a brand-new family may stamp terminal closures without
ever having proved it can extract anything. The canary: a fresh family may
not stamp ``attempts_exhausted`` until it has recorded at least one
successful extraction (a row stamped with the current EXTRACTOR_VERSION).

These tests pin the mechanism, not any row: the fresh-family case leaves
the row open, the proven-family case still bounds retries, and the canary
itself never consults the attempt bound.
"""

from __future__ import annotations

import pytest

from app.core import doc_index
from app.core import ingest_status as ist
from app.core import projects as projects_mod
from app.core.doc_index import _stamp_index_ledger
from app.core.ingest_status import ATTEMPTS_MARK
from app.core.projects import extractor_family_proven


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated DATA_DIR + fresh projects DB for each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    return tmp_path


def _register_and_fail(fresh_db, name: str) -> dict:
    """A documents row that has had its one attempt and failed extraction."""
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    doc = projects_mod.add_document(pid, name, file_path=name, size=8)
    projects_mod.record_ingest_attempt(
        doc["id"], ist.attempt_key(doc.get("content_sha256") or doc.get("drive_md5")),
    )
    _stamp_index_ledger(doc["id"], name, 0, extract_failed=True)
    return projects_mod.get_document(doc["id"])


@pytest.mark.parametrize("name", ["plain.txt", "minimal.docx"])
def test_a_fresh_family_leaves_a_failed_row_open(fresh_db, name):
    """Zero successes for the current family: the row must stay open, never
    carry the terminal closure the live corpus got on its first attempt."""
    row = _register_and_fail(fresh_db, name)
    assert row["ingest_status"] == ist.EXTRACT_FAILED
    assert ATTEMPTS_MARK not in (row.get("ingest_status_reason") or "")
    assert ist.is_open(
        row["ingest_status"], row.get("ingest_status_reason"),
        extension=".txt" if name.endswith(".txt") else ".docx",
        extractor_version=row.get("extractor_version"),
    )


@pytest.mark.parametrize("name", ["one.pdf", "two.pdf"])
def test_a_proven_family_still_bounds_retries(fresh_db, name):
    """Once the family has recorded a success, the deliberate bound applies
    again: identical bytes through an identical extractor cannot improve."""
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    proven = projects_mod.add_document(pid, "proven.txt", file_path="proven.txt", size=8)
    projects_mod.stamp_document_index(
        proven["id"], chunk_count=2, ingest_status=ist.INDEXED,
        extractor_version=ist.EXTRACTOR_VERSION,
    )
    row = _register_and_fail(fresh_db, name)
    assert ATTEMPTS_MARK in (row.get("ingest_status_reason") or "")


def test_the_canary_never_reads_the_attempt_bound(fresh_db, monkeypatch):
    """The family-proven check is about recorded successes, not the bound
    (R16: the guard changes, never the bound)."""

    def _boom():
        raise AssertionError("the canary must not read max_attempts()")

    monkeypatch.setattr(ist, "max_attempts", _boom)
    assert extractor_family_proven() is False


@pytest.mark.parametrize("name", ["plain.txt", "minimal.docx"])
def test_the_walk_extracts_a_plain_file(tmp_path, monkeypatch, name):
    """Regression coverage: a .txt and a minimal .docx extract non-empty
    text through the current walk (the smallest possible failing cases of
    the live closure)."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    import docx as _docx

    path = tmp_path / name
    if name.endswith(".txt"):
        path.write_text("Plain text content for the index.\n" * 10, encoding="utf-8")
    else:
        d = _docx.Document()
        d.add_paragraph("A minimal document with plain runs of text.")
        d.save(str(path))
    text, meta = doc_index._extract_with_meta(str(path), name)
    assert len(text.strip()) > 0
    assert "extract_failed" not in meta
