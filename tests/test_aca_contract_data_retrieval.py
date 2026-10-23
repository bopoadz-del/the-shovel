"""A2: Accepted Contract Amount must come from a Contract Data file.

Live Master Corpus A2 retry on tip d7a4ca8: the ask retrieved Long Form
PSA / CPM permit trackers and reported the figure absent. The executed
amount is in ``…_Contract Data.pdf`` as a scanned table (newlines between
Accepted / Contract / Amount). That file has no ``CONTRACT DATA
particulars`` index-time prefix, so the particulars boost never fires.

Filename ``Contract Data`` is the discriminator PSA/CPM cannot fake.

Not #501 (A3/A5 newest-year lock). Fixture figures only — no live
client SAR amounts.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.rag.vector_store import Chunk


CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
A2_ASK = CATALOG["cases"]["A2"]["ask"]
A3_ASK = CATALOG["cases"]["A3"]["ask"]
LIVE_PREFIX = "Answer only from the client project documents. "
DEFINITION = "What does Accepted Contract Amount mean?"

# Fixture-only. Must not be the live Master Corpus figure.
ACA_INCL = "SAR 1,419,753.07"
ACA_EXCL = "SAR 8,640,000.00"

CD_NAME = (
    "AB-2023-101_QP2 Infra P1_Vol 1.0_Cond of Contract "
    "(complete)_Contract Data.pdf"
)
PSA_NAME = "Long Form PSA 01_02_21 Rev 2 with Legal Amendments.docx"
CPM_NAME = "CPM 16-01-2024.pdf"

# Live shape: scanned table, no particulars prefix, label split across lines.
CD_TEXT = (
    "CONTRACT DATA\n"
    "1.1.1\nAccepted \nContract \nAmount\n"
    f"{ACA_EXCL}\n"
    "1.1.1\nAccepted \nContract \nAmount (including VAT)\n"
    f"{ACA_INCL}\n"
)
PSA_TEXT = (
    "The Long Form PSA refers the reader to the Contract Data particulars "
    "for the Accepted Contract Amount. The actual monetary value is not quoted."
)
CPM_TEXT = (
    "Permit tracking register for 16 January 2024. No contract financial data."
)

ACTIVE = "p_master"
CD_DOC = "cd118"
PSA_DOC = "psa"
CPM_DOC = "cpm"


def _chunk(cid, doc_id, score, text):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_a2_catalog_ask_is_frozen():
    assert A2_ASK == "What is the Accepted Contract Amount including VAT?"


def test_aca_ask_shape_and_filename_gate():
    from app.core.rag.retriever import (
        contract_data_chunk_states_aca,
        filename_looks_like_contract_data,
        query_asks_for_accepted_contract_amount,
    )

    assert query_asks_for_accepted_contract_amount(A2_ASK)
    assert query_asks_for_accepted_contract_amount(LIVE_PREFIX + A2_ASK)
    assert not query_asks_for_accepted_contract_amount(DEFINITION)
    assert not query_asks_for_accepted_contract_amount(A3_ASK)

    assert filename_looks_like_contract_data(CD_NAME)
    assert not filename_looks_like_contract_data(PSA_NAME)
    assert not filename_looks_like_contract_data(CPM_NAME)

    assert contract_data_chunk_states_aca(CD_NAME, CD_TEXT, A2_ASK)
    assert not contract_data_chunk_states_aca(PSA_NAME, PSA_TEXT, A2_ASK)
    assert not contract_data_chunk_states_aca(CPM_NAME, CPM_TEXT, A2_ASK)


def _install_a2_corpus(monkeypatch, *, cd_in_semantic: bool):
    from app.core.rag import retriever as ret

    psa = _chunk("psa", PSA_DOC, 0.88, PSA_TEXT)
    cpm = _chunk("cpm", CPM_DOC, 0.81, CPM_TEXT)
    cd = _chunk("cd", CD_DOC, 0.29, CD_TEXT)
    semantic = [psa, cpm, cd] if cd_in_semantic else [psa, cpm]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return []

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        if CD_DOC in (doc_ids or []):
            return [_chunk("cd", CD_DOC, 0.0, CD_TEXT)]
        return []

    def fake_containing_all(self, project_id, needles, k=20):
        return []

    names = {CD_DOC: CD_NAME, PSA_DOC: PSA_NAME, CPM_DOC: CPM_NAME}
    seeded = [
        {
            "id": CD_DOC,
            "original_name": CD_NAME,
            "file_path": f"Contract/{CD_NAME}",
        },
    ]

    def fake_title_match(pid, phrase, limit=8):
        needle = (phrase or "").lower()
        return [
            doc for doc in seeded
            if needle and needle in (
                f"{doc.get('original_name', '')} {doc.get('file_path', '')}".lower()
            )
        ][:limit]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs", fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 3,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""),
                        raising=False)
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        fake_title_match,
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_a2_retrieves_contract_data_when_psa_leads_the_pool(monkeypatch):
    """The live failure: PSA/CPM occupy the top-k; Contract Data is absent."""
    ret = _install_a2_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(A2_ASK, ACTIVE, k=5)
    names = [getattr(c, "source_name", "") for c in chunks]
    blob = " ".join(c.text for c in chunks)
    assert chunks
    assert any("contract data" in (n or "").lower() for n in names), names
    assert ACA_INCL in blob
    assert chunks[0].doc_id == CD_DOC
    assert all(c.doc_id == CD_DOC for c in chunks)


def test_a2_live_prefix_still_elects_contract_data(monkeypatch):
    ret = _install_a2_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(LIVE_PREFIX + A2_ASK, ACTIVE, k=5)
    assert chunks
    assert ACA_INCL in chunks[0].text
    assert chunks[0].doc_id == CD_DOC


def test_a2_drops_psa_when_contract_data_is_already_in_the_pool(monkeypatch):
    ret = _install_a2_corpus(monkeypatch, cd_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(A2_ASK, ACTIVE, k=5)
    assert chunks
    assert all(c.doc_id == CD_DOC for c in chunks)
    assert ACA_INCL in " ".join(c.text for c in chunks)


def test_a3_is_not_stolen_onto_the_aca_rescue(monkeypatch):
    ret = _install_a2_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(A3_ASK, ACTIVE, k=5)
    assert all(c.doc_id != CD_DOC for c in chunks)


def test_definition_question_is_not_forced_onto_contract_data(monkeypatch):
    ret = _install_a2_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(DEFINITION, ACTIVE, k=5)
    assert all(c.doc_id != CD_DOC for c in chunks)


CONTRACT_SUM_SYNONYM = "What is the contract sum before VAT as a number?"


def test_contract_sum_synonym_surfaces_the_aca_row(monkeypatch):
    """A synonym ("contract sum") must still surface the Accepted Contract
    Amount Contract Data row — via the reservation, since query_asks_for_
    accepted_contract_amount is False for this phrasing and the CD chunk sits
    at a low semantic score behind PSA/CPM."""
    ret = _install_a2_corpus(monkeypatch, cd_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(CONTRACT_SUM_SYNONYM, ACTIVE, k=5)
    assert any(c.doc_id == CD_DOC for c in chunks), [c.doc_id for c in chunks]
    assert ACA_EXCL in " ".join(c.text for c in chunks)


def test_mutation_filename_bonus_is_what_lifts_contract_data(monkeypatch):
    ret = _install_a2_corpus(monkeypatch, cd_in_semantic=True)
    monkeypatch.setattr(ret, "filename_looks_like_contract_data", lambda *_a, **_k: False)
    # Isolate the filename bonus from the including-VAT particular path.
    monkeypatch.setattr(ret, "query_asks_for_aca_including_vat", lambda _q: False)
    monkeypatch.setattr(ret, "query_is_aca_including_vat_particular", lambda _q: False)
    chunks, _ = ret.retrieve_with_filter(A2_ASK, ACTIVE, k=5)
    assert chunks[0].doc_id == PSA_DOC
