"""A5 rate and A9 Engineer must surface after the #501 year-lock.

Live Wave-1 retest on tip ``3a5fce5`` (#501 merged):

* **A3 PASS** — 852 days, AB-2023-101. Year-lock is doing its job.
* **A5 FAIL** — not FAIL_WRONG_CONTRACT. The excerpts never stated
  ``0.1% of the Contract Price per calendar day``. The rate lives in a
  different chunk than Time for Completion; a same-year cap / Sub-Clause
  8.8 row satisfied the asked-label check and reservation was a no-op.
* **A9 FAIL** — Client/Consultant parties only (Long Form PSA). Expected
  EXAMPLAR / EXCON Example Limited. The appointment is a filled Engineer row;
  without the index-time particulars prefix the year-lock fence dropped
  the 118 file and no-id PSA chunks filled the pool.

A2 / C2 / D1 are owned elsewhere. Do not invent a signatory. Fixture
figures and party names only.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.rag.vector_store import Chunk


CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
A3_ASK = CATALOG["cases"]["A3"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]
A6_ASK = CATALOG["cases"]["A6"]["ask"]
A9_ASK = CATALOG["cases"]["A9"]["ask"]
LIVE_PREFIX = "Answer only from the client project documents. "
LIVE_A5 = LIVE_PREFIX + A5_ASK
LIVE_A9 = LIVE_PREFIX + A9_ASK

# Fixture-only. Must not be the confidential live pack's other figures.
A5_RATE = "0.1% of the Contract Price per calendar day"
A9_FIRM = "Northwater Engineers (Demo Saudi Limited)"
A3_DAYS = "852 days"
A6_DAYS = "365 days"

DD23_NAME = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Vol 1 - Conditions of Contract.pdf"
)
CD_SCANNED_NAME = (
    "AB-2023-101_QP2 Infra P1_Vol 1.0_Cond of Contract "
    "(complete)_Contract Data.pdf"
)
PSA_NAME = "Long Form PSA 01_02_21 Rev 2 with Legal Amendments.docx"
GC_NAME = DD23_NAME

PREFIXED_RATE = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    f"8.8 Delay Damages for the whole of the Works: {A5_RATE}"
)
PREFIXED_CAP = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    "8.8 Maximum amount of delay damages: 10% of the Accepted Contract Amount"
)
PREFIXED_TFC = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    f"1.1.75 Time for Completion for the whole of the Works: {A3_DAYS}"
)
PREFIXED_DNP = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    f"1.1.27 Defects Notification Period: {A6_DAYS} from the Taking-Over Certificate"
)
GC_8_8 = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.8 Delay Damages. If "
    "the Contractor fails to comply with Sub-Clause 8.2, the Contractor "
    "shall pay delay damages for the whole of the Works at the rate stated "
    "in the Contract Data for every calendar day which shall elapse between "
    "the relevant Time for Completion and the date stated in the Taking-Over "
    "Certificate, up to the maximum amount of delay damages stated in the "
    "Contract Data."
)
# Live A5 shape: scanned table, no particulars prefix, label split across lines.
SCANNED_RATE = (
    "CONTRACT DATA\n"
    "8.8\nDelay \nDamages for the whole of the Works\n"
    f"{A5_RATE}\n"
)
SCANNED_ENGINEER = (
    "CONTRACT DATA\n"
    "1.3.1 (b)\nEngineer\n"
    f"{A9_FIRM}\n"
)
PREFIXED_ENGINEER = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    f"1.3.1 (b) Engineer: {A9_FIRM}"
)
PSA_PARTIES = (
    "Long Form Professional Services Agreement. The parties are:\n"
    "Client: the Employer named in the Particular Conditions.\n"
    "Consultant: the firm engaged to provide design services.\n"
    "The Consultant is not named as the Engineer in this agreement."
)
GC_ENGINEER = (
    'Volume 1 - Conditions of Contract. 1.1.35 "Engineer" means the person '
    "appointed by the Employer to act as the Engineer for the purposes of "
    "the Contract and named in the Contract Data, or any replacement "
    "appointed under Sub-Clause 3.6 within 28 days."
)

ACTIVE = "p_master"
RATE_DOC = "rate118"
CAP_DOC = "cap118"
GC_DOC = "gc118"
TFC_DOC = "tfc118"
DNP_DOC = "dnp118"
ENG_DOC = "eng118"
PSA_DOC = "psa"
SCAN_RATE_DOC = "scanrate"
SCAN_ENG_DOC = "scaneng"


def _chunk(cid, doc_id, score, text, project_id=ACTIVE):
    return Chunk(
        chunk_id=cid,
        project_id=project_id,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_catalog_asks_are_frozen():
    assert A5_ASK == "What are the Delay Damages for the whole of the Works?"
    assert A9_ASK == "Who is the Engineer under this contract?"


def test_rate_and_engineer_predicates_and_ask_shapes():
    from app.core.rag.retriever import (
        chunk_answers_asked_particular,
        chunk_states_delay_damages_rate,
        chunk_states_engineer_identity,
        particulars_row_answers_asked_label,
        query_asks_for_delay_damages_rate,
        query_asks_who_the_engineer_is,
    )

    assert query_asks_for_delay_damages_rate(A5_ASK)
    assert query_asks_for_delay_damages_rate(LIVE_A5)
    assert not query_asks_for_delay_damages_rate(A3_ASK)
    assert not query_asks_for_delay_damages_rate(
        "Calculate the delay damages per calendar day in SAR for the "
        "whole of the Works"
    )
    assert not query_asks_for_delay_damages_rate(
        "What is the maximum amount of delay damages?"
    )

    assert query_asks_who_the_engineer_is(A9_ASK)
    assert query_asks_who_the_engineer_is(LIVE_A9)
    assert not query_asks_who_the_engineer_is(
        "What does Engineer mean under the contract?"
    )
    assert not query_asks_who_the_engineer_is(
        "Who is the Engineer's Representative under this contract?"
    )

    assert chunk_states_delay_damages_rate(PREFIXED_RATE)
    assert chunk_states_delay_damages_rate(SCANNED_RATE)
    assert not chunk_states_delay_damages_rate(PREFIXED_CAP)
    assert not chunk_states_delay_damages_rate(GC_8_8)
    assert not chunk_states_delay_damages_rate(PREFIXED_TFC)
    # Cap still answers the *label* so #501 can year-lock from it when
    # the rate chunk has not been fetched yet.
    assert particulars_row_answers_asked_label(A5_ASK, PREFIXED_CAP)

    assert chunk_states_engineer_identity(PREFIXED_ENGINEER)
    assert chunk_states_engineer_identity(SCANNED_ENGINEER)
    assert not chunk_states_engineer_identity(PSA_PARTIES)
    assert not chunk_states_engineer_identity(GC_ENGINEER)

    assert chunk_answers_asked_particular(A5_ASK, SCANNED_RATE)
    assert chunk_answers_asked_particular(A9_ASK, SCANNED_ENGINEER)
    assert not chunk_answers_asked_particular(A5_ASK, GC_8_8)
    assert not chunk_answers_asked_particular(A9_ASK, PSA_PARTIES)


def test_election_prefers_the_rate_over_the_same_year_cap():
    from app.core.rag.retriever import elect_answer_bearing_contract

    ranked = [
        (DD23_NAME, PREFIXED_CAP),
        (DD23_NAME, GC_8_8),
        (CD_SCANNED_NAME, SCANNED_RATE),
    ]
    assert elect_answer_bearing_contract(A5_ASK, ranked) == "ab-2023-101"
    assert elect_answer_bearing_contract(LIVE_A5, ranked) == "ab-2023-101"


def test_election_picks_scanned_engineer_over_psa_parties():
    from app.core.rag.retriever import elect_answer_bearing_contract

    ranked = [
        (PSA_NAME, PSA_PARTIES),
        (DD23_NAME, GC_ENGINEER),
        (CD_SCANNED_NAME, SCANNED_ENGINEER),
    ]
    assert elect_answer_bearing_contract(A9_ASK, ranked) == "ab-2023-101"
    assert elect_answer_bearing_contract(LIVE_A9, ranked) == "ab-2023-101"


def test_reservation_swaps_a_cap_row_for_the_rate():
    from app.core.rag.retriever import reserve_matching_particulars_row

    cap = _chunk("cap", CAP_DOC, 0.9, PREFIXED_CAP)
    gc = _chunk("gc", GC_DOC, 0.88, GC_8_8)
    rate = _chunk("rate", SCAN_RATE_DOC, 0.31, SCANNED_RATE)
    kept = [cap, gc]
    assert reserve_matching_particulars_row(A5_ASK, kept, [cap, gc, rate]) is True
    assert any(A5_RATE in (c.text or "") for c in kept)


def _install_corpus(monkeypatch, *, semantic, rescue_hits, names):
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers or []).lower()
        out = []
        for chunk in rescue_hits:
            text_l = (chunk.text or "").lower()
            if "delay" in blob and "damages" in blob and "0.1%" in text_l:
                out.append(chunk)
            if "engineer" in blob and "northwater" in text_l:
                out.append(chunk)
            if "1.3.1" in blob and "engineer" in text_l:
                out.append(chunk)
        return out[:k]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, project_id, doc_ids, k_per_doc=12: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 4,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""),
                        raising=False)
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda pid, phrase, limit=8: [],
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_a5_surfaces_the_rate_when_same_year_gc_and_cap_lead(monkeypatch):
    """Live A5 after year-lock: 118 is elected, excerpts have no 0.1%."""
    gc = _chunk("gc", GC_DOC, 0.92, GC_8_8)
    cap = _chunk("cap", CAP_DOC, 0.90, PREFIXED_CAP)
    tfc = _chunk("tfc", TFC_DOC, 0.87, PREFIXED_TFC)
    rate = _chunk("rate", SCAN_RATE_DOC, 0.22, SCANNED_RATE)
    names = {
        GC_DOC: GC_NAME,
        CAP_DOC: DD23_NAME,
        TFC_DOC: DD23_NAME,
        SCAN_RATE_DOC: CD_SCANNED_NAME,
    }
    ret = _install_corpus(
        monkeypatch,
        semantic=[gc, cap, tfc],
        rescue_hits=[rate],
        names=names,
    )
    chunks, _ = ret.retrieve_with_filter(A5_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks, "rate never reached the pool"
    assert A5_RATE in blob
    assert A5_RATE in chunks[0].text
    assert "at the rate stated in the Contract Data" not in blob
    assert "Maximum amount of delay damages" not in blob


def test_a5_live_prefix_still_returns_the_rate(monkeypatch):
    gc = _chunk("gc", GC_DOC, 0.92, GC_8_8)
    cap = _chunk("cap", CAP_DOC, 0.90, PREFIXED_CAP)
    rate = _chunk("rate", SCAN_RATE_DOC, 0.22, SCANNED_RATE)
    names = {
        GC_DOC: GC_NAME, CAP_DOC: DD23_NAME, SCAN_RATE_DOC: CD_SCANNED_NAME,
    }
    ret = _install_corpus(
        monkeypatch, semantic=[gc, cap], rescue_hits=[rate], names=names,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_A5, ACTIVE, k=5)
    assert chunks
    assert A5_RATE in chunks[0].text


def test_a9_surfaces_the_engineer_not_psa_parties(monkeypatch):
    """Live A9: Client/Consultant only. Appointment never reached top-k."""
    psa = _chunk("psa", PSA_DOC, 0.91, PSA_PARTIES)
    gloss = _chunk("gloss", GC_DOC, 0.86, GC_ENGINEER)
    eng = _chunk("eng", SCAN_ENG_DOC, 0.24, SCANNED_ENGINEER)
    names = {
        PSA_DOC: PSA_NAME, GC_DOC: GC_NAME, SCAN_ENG_DOC: CD_SCANNED_NAME,
    }
    ret = _install_corpus(
        monkeypatch,
        semantic=[psa, gloss],
        rescue_hits=[eng],
        names=names,
    )
    chunks, _ = ret.retrieve_with_filter(A9_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks
    assert A9_FIRM in blob
    assert A9_FIRM in chunks[0].text
    assert all(c.doc_id == SCAN_ENG_DOC for c in chunks)
    assert "The parties are:" not in blob
    assert "means the person appointed" not in blob


def test_a9_live_prefix_still_returns_the_engineer(monkeypatch):
    psa = _chunk("psa", PSA_DOC, 0.91, PSA_PARTIES)
    eng = _chunk("eng", SCAN_ENG_DOC, 0.24, SCANNED_ENGINEER)
    names = {PSA_DOC: PSA_NAME, SCAN_ENG_DOC: CD_SCANNED_NAME}
    ret = _install_corpus(
        monkeypatch, semantic=[psa], rescue_hits=[eng], names=names,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_A9, ACTIVE, k=5)
    assert chunks
    assert A9_FIRM in chunks[0].text


def test_a3_and_a6_are_not_stolen_onto_the_rate_or_engineer_rescue(monkeypatch):
    gc = _chunk("gc", GC_DOC, 0.92, GC_8_8)
    tfc = _chunk("tfc", TFC_DOC, 0.84, PREFIXED_TFC)
    dnp = _chunk("dnp", DNP_DOC, 0.83, PREFIXED_DNP)
    rate = _chunk("rate", SCAN_RATE_DOC, 0.22, SCANNED_RATE)
    eng = _chunk("eng", SCAN_ENG_DOC, 0.21, SCANNED_ENGINEER)
    names = {
        GC_DOC: GC_NAME,
        TFC_DOC: DD23_NAME,
        DNP_DOC: DD23_NAME,
        SCAN_RATE_DOC: CD_SCANNED_NAME,
        SCAN_ENG_DOC: CD_SCANNED_NAME,
    }
    ret = _install_corpus(
        monkeypatch,
        semantic=[gc, tfc, dnp],
        rescue_hits=[rate, eng],
        names=names,
    )
    a3, _ = ret.retrieve_with_filter(A3_ASK, ACTIVE, k=5)
    a6, _ = ret.retrieve_with_filter(A6_ASK, ACTIVE, k=5)
    assert all(c.doc_id != SCAN_RATE_DOC for c in a3)
    assert all(c.doc_id != SCAN_ENG_DOC for c in a3)
    assert all(c.doc_id != SCAN_RATE_DOC for c in a6)
    assert all(c.doc_id != SCAN_ENG_DOC for c in a6)
    assert any(A3_DAYS in (c.text or "") for c in a3)
    assert any(A6_DAYS in (c.text or "") for c in a6)


def test_representative_ask_is_not_forced_onto_the_engineer_row(monkeypatch):
    """D1 / Engineer's Representative stays off this rescue. No invented name."""
    psa = _chunk("psa", PSA_DOC, 0.91, PSA_PARTIES)
    eng = _chunk("eng", SCAN_ENG_DOC, 0.24, SCANNED_ENGINEER)
    names = {PSA_DOC: PSA_NAME, SCAN_ENG_DOC: CD_SCANNED_NAME}
    ret = _install_corpus(
        monkeypatch, semantic=[psa], rescue_hits=[eng], names=names,
    )
    chunks, _ = ret.retrieve_with_filter(
        "Who is the Engineer's Representative under this contract?",
        ACTIVE,
        k=5,
    )
    assert all(c.doc_id != SCAN_ENG_DOC for c in chunks)
