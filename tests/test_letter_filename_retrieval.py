"""D1: a named-site letter already in the corpus must be retrieved.

Live Master Corpus pack D1 (SHA 567147a / #480): the XBCC Concrete
Batching Plant at Wadi Example letter was in Neon (ids 8199b14b,
5a1e0001) and the handover certificate was too (5a1e0003). Retrieval
returned only Volume 5 Other Documents (geotech, plot agreement, weekly
reports) and the model refused — letter not in the excerpts.

Term rescue treated Volume 5's in-chunk place-name as "already
grounded" and skipped the out-of-pool fetch. The letter's FILENAME is
the discriminator Volume 5 cannot fake.

This fence uses the live letter filename shape and the live signatory
strings already pinned in ``tests/test_docx_signature_extraction.py``.
No new live client figures.
"""
from __future__ import annotations

import importlib

import pytest

from app.core.rag.vector_store import Chunk


LETTER_NAME = (
    "Letter toAICC on Completion and transfer of responsibility "
    "-XBCC Concrete Batching Plant at wadi Example - caw.docx"
)
HANDOVER_NAME = (
    "20230101 XBCC Batch Plant - Temporary Plot Handover Certificate.pdf"
)
VOL5_NAME = "AB-2023-101 Vol 5 Other Documents.pdf"
PLOT_NAME = "Wadi Example - Plot Agreement Proposal.pdf"

LETTER_TEXT = (
    "Letter to ACMC on Completion and transfer of responsibility — "
    "XBCC Concrete Batching Plant at Wadi Example.\n"
    "Signed: Project Signatory\n"
    "Engineer's Representative\n"
    "EXCON Example Limited\n"
)
VOL5_TEXT = (
    "Volume 5 Other Documents. Geotechnical investigation at Wadi Example. "
    "Plot agreement for the batching plant area. Weekly progress reports "
    "covering soils, access, and the temporary plot."
)

D1_QUERY = (
    "Who signed the letter about the XBCC Concrete Batching Plant at "
    "Wadi Example, and in what capacity?"
)
D1_CATALOG_QUERY = (
    "Who signed the letter about the batching plant at Creek Bend, "
    "and in what capacity?"
)
ENGINEER_QUERY = "Who is the Engineer under this contract?"
PARTICULARS_QUERY = (
    "What is the Time for Completion for the whole of the Works?"
)

ACTIVE = "p_master"
LETTER_DOC = "8199b14b"
VOL5_DOC = "vol5other"


def _chunk(cid, doc_id, score, text):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_d1_query_is_a_letter_ask():
    from app.core.rag.retriever import query_asks_for_letter_or_signatory

    assert query_asks_for_letter_or_signatory(D1_QUERY)
    assert query_asks_for_letter_or_signatory(D1_CATALOG_QUERY)
    assert query_asks_for_letter_or_signatory(
        "who put their name to the Creek Bend batching plant letter"
    )


def test_who_is_the_engineer_is_not_a_letter_ask():
    """A9 / contract-role identity stays off this path (#483 owns it)."""
    from app.core.rag.retriever import query_asks_for_letter_or_signatory

    assert not query_asks_for_letter_or_signatory(ENGINEER_QUERY)
    assert not query_asks_for_letter_or_signatory(PARTICULARS_QUERY)
    assert not query_asks_for_letter_or_signatory(
        "What does Schedule 9 of the contract volumes cover?"
    )


def test_letter_filename_overlaps_named_site_and_party():
    from app.core.rag.retriever import (
        distinctive_query_terms,
        filename_looks_like_letter,
        filename_match_bonus,
        filename_query_overlap,
    )

    terms = distinctive_query_terms(D1_QUERY)
    assert filename_looks_like_letter(LETTER_NAME)
    assert not filename_looks_like_letter(VOL5_NAME)
    assert not filename_looks_like_letter(PLOT_NAME)
    letter_overlap = filename_query_overlap(LETTER_NAME, terms)
    vol5_overlap = filename_query_overlap(VOL5_NAME, terms)
    assert letter_overlap >= 0.6, letter_overlap
    assert vol5_overlap == 0.0, vol5_overlap
    assert filename_match_bonus(
        LETTER_NAME, terms, letter_query=True,
    ) > filename_match_bonus(VOL5_NAME, terms, letter_query=True)


def test_plot_agreement_filename_loses_to_the_letter():
    from app.core.rag.retriever import (
        distinctive_query_terms,
        filename_match_bonus,
    )

    terms = distinctive_query_terms(D1_QUERY)
    letter = filename_match_bonus(LETTER_NAME, terms, letter_query=True)
    plot = filename_match_bonus(PLOT_NAME, terms, letter_query=True)
    handover = filename_match_bonus(HANDOVER_NAME, terms, letter_query=True)
    assert letter > plot
    assert letter > handover


def _install_d1_corpus(monkeypatch, *, letter_in_semantic: bool, rescue_docs=None):
    """Semantic pool is Volume 5 (and optionally a buried letter)."""
    from app.core.rag import retriever as ret

    vol5 = _chunk("vol5", VOL5_DOC, 0.88, VOL5_TEXT)
    letter = _chunk("letter", LETTER_DOC, 0.31, LETTER_TEXT)
    semantic = [vol5, letter] if letter_in_semantic else [vol5]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return []

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        if LETTER_DOC in (doc_ids or []):
            return [_chunk("letter", LETTER_DOC, 0.0, LETTER_TEXT)]
        return []

    names = {
        LETTER_DOC: LETTER_NAME,
        VOL5_DOC: VOL5_NAME,
    }
    default_matches = [
        {"id": LETTER_DOC, "original_name": LETTER_NAME, "file_path": f"Misc/{LETTER_NAME}"},
    ]
    seeded = rescue_docs if rescue_docs is not None else default_matches

    def fake_filename_match(pid, terms, min_terms=2, require_letter=False, limit=8):
        out = []
        for doc in seeded:
            blob = f"{doc.get('original_name', '')} {doc.get('file_path', '')}".lower()
            if require_letter and "letter" not in blob:
                continue
            hits = sum(1 for t in (terms or []) if t and t.lower() in blob)
            if hits >= min_terms:
                out.append(doc)
        return out[:limit]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs", fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 2,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""),
                        raising=False)
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        fake_filename_match,
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_d1_retrieves_letter_when_only_vol5_is_in_the_semantic_pool(monkeypatch):
    """The live failure: letter is in Neon, cosine never fetches it."""
    ret = _install_d1_corpus(monkeypatch, letter_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(D1_QUERY, ACTIVE, k=5)
    names = [getattr(c, "source_name", "") for c in chunks]
    texts = " ".join(c.text for c in chunks)
    assert any("letter" in (n or "").lower() for n in names), names
    assert "Project Signatory" in texts
    assert "Engineer's Representative" in texts
    assert "EXCON Example Limited" in texts
    assert chunks[0].doc_id == LETTER_DOC


def test_d1_lifts_a_buried_letter_already_in_the_pool(monkeypatch):
    ret = _install_d1_corpus(monkeypatch, letter_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(D1_QUERY, ACTIVE, k=5)
    assert chunks[0].doc_id == LETTER_DOC
    assert "Project Signatory" in chunks[0].text


def test_mutation_filename_bonus_is_what_lifts_the_letter(monkeypatch):
    """If the bonus is a no-op, Volume 5 wins again — the live ranking."""
    ret = _install_d1_corpus(monkeypatch, letter_in_semantic=True)
    monkeypatch.setattr(ret, "filename_match_bonus", lambda *a, **k: 0.0)
    chunks, _ = ret.retrieve_with_filter(D1_QUERY, ACTIVE, k=5)
    assert chunks[0].doc_id == VOL5_DOC


def test_a_generic_contract_filename_does_not_earn_a_bonus():
    """Dual-query / Contract Data tests name every doc ``contract.pdf``.

    One shared token must not lift those chunks — that is how D1's
    filename bonus would have perturbed unrelated ranking.
    """
    from app.core.rag.retriever import (
        distinctive_query_terms,
        filename_match_bonus,
    )

    terms = distinctive_query_terms(PARTICULARS_QUERY)
    assert filename_match_bonus(
        "contract.pdf", terms, letter_query=False,
    ) == 0.0
    assert filename_match_bonus(
        "contract.pdf", terms, letter_query=True,
    ) == 0.0


def test_particulars_query_is_not_stolen_by_a_letter_filename(monkeypatch):
    ret = _install_d1_corpus(monkeypatch, letter_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(PARTICULARS_QUERY, ACTIVE, k=5)
    # Not a letter ask and <3 overlapping filename terms on Vol 5 — the
    # letter must not be injected into a Contract Data question.
    assert all(c.doc_id != LETTER_DOC for c in chunks)


@pytest.fixture
def project_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod

    importlib.reload(db_mod)
    from app.core import projects as projects_mod

    pm = importlib.reload(projects_mod)
    pm._initialized = False
    pm.init_db()
    return pm


def test_filename_sql_finds_the_s5_letter_among_vol5_decoys(project_store):
    from app.core.rag.retriever import distinctive_query_terms

    p = project_store.create_project("D1 corpus")
    pid = p["id"]
    letter = project_store.add_document(
        pid, LETTER_NAME, file_path=f"Misc/{LETTER_NAME}", size=12,
    )
    project_store.add_document(pid, VOL5_NAME, size=40)
    project_store.add_document(pid, PLOT_NAME, size=8)
    project_store.add_document(pid, HANDOVER_NAME, size=9)
    project_store.add_document(
        pid, "Weekly Report W22 - soils.pdf", size=5,
    )

    terms = distinctive_query_terms(D1_QUERY)
    found = project_store.documents_matching_filename_terms(
        pid, terms, min_terms=2, require_letter=True,
    )
    assert [d["id"] for d in found] == [letter["id"]]
    assert "XBCC" in found[0]["original_name"]
    assert "wadi Example" in found[0]["original_name"]


def test_filename_sql_require_letter_excludes_the_plot_agreement(project_store):
    from app.core.rag.retriever import distinctive_query_terms

    p = project_store.create_project("D1 decoys")
    pid = p["id"]
    project_store.add_document(pid, PLOT_NAME, size=8)
    project_store.add_document(pid, VOL5_NAME, size=40)
    terms = distinctive_query_terms(D1_QUERY)
    found = project_store.documents_matching_filename_terms(
        pid, terms, min_terms=2, require_letter=True,
    )
    assert found == []
