"""Document typing reads the document's own text, not only its name.

Ingest types a row before its text exists, so a name with no type word left
it ``document`` whatever the text said. The backfill now samples the
document's opening chunks; content alone decides only with enough distinct
evidence for one kind and no tie between kinds. Every input is invented here.
"""
from __future__ import annotations

import json

import pytest

from app.core import doc_types
from app.core import doc_typing_backfill as bf
from tests.test_ingest_reconcile import _reload_projects


@pytest.fixture(autouse=True)
def _no_folder_roles(monkeypatch):
    monkeypatch.setenv("DOC_ROLE_FOLDERS", json.dumps({}))


# ── the content-only bar ─────────────────────────────────────────────────────

@pytest.mark.parametrize("content, expected", [
    # two distinct programme phrases, nothing else: content decides
    ("Annex 4. The critical path runs through milestone M3; early start 02 Mar.",
     "schedule"),
    # one programme phrase only: too weak to decide alone
    ("Annex 4. Payment falls due at milestone M3.", "unrecognised"),
])
def test_content_alone_needs_two_distinct_hits(content, expected):
    assert doc_types.classify("annex_04.pdf", content)["name"] == expected


@pytest.mark.parametrize("content, expected", [
    # two kinds score the same on content alone: no kind is preferred
    ("The critical path and milestone dates. Dear Sir, yours faithfully.",
     "unrecognised"),
    # one kind ahead: it wins
    ("The critical path, milestone and early start dates. Dear Sir, yours faithfully.",
     "schedule"),
])
def test_a_content_tie_between_kinds_is_not_broken_by_table_order(content, expected):
    assert doc_types.classify("item_0091.pdf", content)["name"] == expected


@pytest.mark.parametrize("name, expected", [
    # a name match keeps today's rule: one content word may add to it
    ("Weekly_Report_07.pdf", "report"),
    ("Item_07.pdf", "unrecognised"),
])
def test_a_name_match_is_not_held_to_the_content_bar(name, expected):
    assert doc_types.classify(name, "manpower")["name"] == expected


# ── multi-word name keywords across separators ───────────────────────────────

@pytest.mark.parametrize("name, expected", [
    # "inspection and test plan" across underscores beats the one-word match
    ("Inspection_and_Test_Plan_L3.pdf", "itp"),
    ("Inspection_Request_L3.pdf", "inspection_request"),
])
def test_multi_word_keywords_match_across_separators(name, expected):
    assert doc_types.classify(name)["name"] == expected


@pytest.mark.parametrize("name, expected", [
    ("Method_Statement_Blockwork_L2.docx", "method_statement"),
    ("Blockwork_L2_Statement.docx", "unrecognised"),
])
def test_method_statement_is_a_kind_in_the_table(name, expected):
    assert doc_types.classify(name)["name"] == expected


# ── the backfill reads the document's own chunks ─────────────────────────────

def _project_with_doc(monkeypatch, tmp_path, name, texts):
    import numpy as np
    from sqlalchemy import update

    from app.core.db import SessionLocal, engine
    from app.core.models import Document, RagChunk

    pm = _reload_projects(monkeypatch, tmp_path)
    RagChunk.__table__.create(bind=engine, checkfirst=True)
    pm.create_project("cproj", user_id="system", project_id="cproj")
    did = pm.add_document("cproj", name, size=1)["id"]
    with SessionLocal() as s:
        s.execute(update(Document).where(Document.id == did)
                  .values(doc_type="document", doc_role="other"))
        for i, text in enumerate(texts):
            s.add(RagChunk(chunk_id=f"{did}-{i}", project_id="cproj", doc_id=did,
                           chunk_index=i, text=text,
                           embedding=np.zeros(8, dtype=np.float32),
                           created_at="2026-01-01T00:00:00Z"))
        s.commit()
    return did


@pytest.mark.parametrize("texts, counts", [
    # the name says nothing; the opening chunks are a method statement
    (["1. Scope. This method statement covers blockwork to level 2.",
      "2. Sequence of work: set out, lay, cure. 3. Risk assessment attached."],
     {"method_statement": 1}),
    # the same name, text that names no kind: nothing is filled
    (["1. Scope. Blockwork to level 2.", "2. Lay and cure."], {}),
])
def test_backfill_types_a_document_by_its_own_text(monkeypatch, tmp_path, texts, counts):
    did = _project_with_doc(monkeypatch, tmp_path, "DOC-0417-r1.pdf", texts)
    plan = bf.doc_typing_backfill(project_id="cproj")["plan"]
    assert plan["counts"]["doc_type"] == counts
    assert plan["documents"] == ([did] if counts else [])


def test_no_chunks_table_means_no_sample(monkeypatch, tmp_path):
    from app.core.db import SessionLocal

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("cproj", user_id="system", project_id="cproj")
    with SessionLocal() as s:
        assert bf.content_samples(s, "cproj") == {}


@pytest.mark.parametrize("limit, expected", [
    (10_000, "first chunk\nsecond chunk"),
    (8, "first ch"),
])
def test_content_sample_is_opening_chunks_in_order_and_capped(monkeypatch, tmp_path, limit, expected):
    from app.core.db import SessionLocal

    did = _project_with_doc(monkeypatch, tmp_path, "x.pdf", ["first chunk", "second chunk"])
    with SessionLocal() as s:
        assert bf.content_samples(s, "cproj", limit=limit) == {did: expected}
