"""A6: Defects Notification Period from Contract Data, not PSA/CPM TOC.

Live OLD-pack A6 on Master Corpus (tip 82eb9c5 / #522 C1 just shipped):

    Answer only from the client project documents. What is the Defects
    Notification Period?

Expected: 365 days (Taking-Over Certificate / Contract Data under
AB-2023-101).

Observed: refusal — retrieved excerpts were Long Form PSA chunk #4 and
CPM 22-01-2024 / 16-01-2024 chunk #7 (TOC, recitals, document
registers). None stated a DNP value.

A2/A3/A5/A9 already have exclusive asked-value fences. A6 was not
named off the C1 path and had no duration fence of its own. This
rescue elects the filled 1.1.27 row.
Do not steal A2/A3/A5/A9/C1/E1/F1. Fixture wording only.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.runtime import (
    _CG_REFUSAL,
    _graft_asked_contract_particular,
    _postprocess_answer,
)
from app.core.rag.vector_store import Chunk


CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
A2_ASK = CATALOG["cases"]["A2"]["ask"]
A3_ASK = CATALOG["cases"]["A3"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]
A6_ASK = CATALOG["cases"]["A6"]["ask"]
A9_ASK = CATALOG["cases"]["A9"]["ask"]
C1_ASK = CATALOG["cases"]["C1"]["ask"]
E1_ASK = CATALOG["cases"]["E1"]["ask"]
F1_ASK = (
    "Answer only from the client project documents. Generate a high-level "
    "WBS for the demolition and site clearance scope in this project's BOQ."
)
LIVE_PREFIX = "Answer only from the client project documents. "
LIVE_A6 = LIVE_PREFIX + A6_ASK

A6_DAYS = "365 days"
A3_DAYS = "852 days"
A5_RATE = "0.1% of the Contract Price per calendar day"
A9_FIRM = "Northwater Engineers (Demo Saudi Limited)"
ACA_INCL = "SAR 2,017,680,124.69"

DD23_NAME = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Vol 1 - Conditions of Contract.pdf"
)
CD_SCANNED_NAME = (
    "AB-2023-101_QP2 Infra P1_Vol 1.0_Cond of Contract "
    "(complete)_Contract Data.pdf"
)
PSA_NAME = "Long Form PSA 01_02_21 Rev 2 with Legal Amendments.docx"
CPM_NAME = "CPM 22-01-2024.pdf"
CPM16_NAME = "CPM 16-01-2024.pdf"

PREFIXED_DNP = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    f"1.1.27 Defects Notification Period: {A6_DAYS} from the Taking-Over Certificate"
)
SCANNED_DNP = (
    "CONTRACT DATA\n"
    "1.1.27\nDefects \nNotification \nPeriod\n"
    f"{A6_DAYS} from the Taking-Over Certificate\n"
)
PSA_TOC = (
    "Long Form Professional Services Agreement. Table of contents. "
    "Recitals. 1. Definitions. Defects Notification Period. "
    "Document register. No duration is stated in these recitals."
)
CPM_REGISTER = (
    "CPM contract data sheet. Document register. Table of contents. "
    "Particulars headings: Time for Completion, Defects Notification "
    "Period, Delay Damages. Recitals only — no filled 1.1.27 value."
)
GC_POINTER = (
    "Volume 1 - Conditions of Contract. Sub-Clause 11.1 Defects "
    "Notification Period. The Defects Notification Period is the "
    "period stated in the Contract Data following the date stated "
    "in the Taking-Over Certificate."
)
GLOSSARY = (
    '"Defects Notification Period" means the period for notifying '
    "defects under Clause 11, as stated in the Contract Data."
)
PREFIXED_TFC = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    f"1.1.75 Time for Completion for the whole of the Works: {A3_DAYS}"
)
PREFIXED_RATE = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    f"8.8 Delay Damages for the whole of the Works: {A5_RATE}"
)
SCANNED_ACA = (
    "CONTRACT DATA\n"
    "1.1.1\nAccepted Contract Amount (including VAT)\n"
    f"{ACA_INCL}\n"
)
SCANNED_ENGINEER = (
    "CONTRACT DATA\n"
    "1.3.1 (b)\nEngineer\n"
    f"{A9_FIRM}\n"
)
C1_LIST = (
    "Post Tender Clarifications\n"
    "Tender Addenda\n"
    "Schedule of Project Requirements\n"
)

ACTIVE = "p_master"
PSA_DOC = "psa"
CPM_DOC = "cpm22"
CPM16_DOC = "cpm16"
GC_DOC = "gc118"
DNP_DOC = "dnp118"
SCAN_DNP_DOC = "scandnp"
TFC_DOC = "tfc118"
RATE_DOC = "rate118"
ACA_DOC = "aca118"
ENG_DOC = "eng118"
C1_DOC = "c1list"


def _chunk(cid, doc_id, score, text, chunk_index=0):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def test_a6_catalog_ask_is_frozen():
    assert A6_ASK == "What is the Defects Notification Period?"
    assert CATALOG["cases"]["A6"]["must"] == ["365"]


def test_a6_ask_shape_and_duration_gates():
    from app.core.rag.retriever import (
        chunk_states_defects_notification_period,
        extract_defects_notification_period,
        query_asks_for_defects_notification_period,
        query_wants_contract_data_file,
    )

    assert query_asks_for_defects_notification_period(A6_ASK)
    assert query_asks_for_defects_notification_period(LIVE_A6)
    assert query_asks_for_defects_notification_period(
        "What is the DNP under this contract?"
    )
    assert query_wants_contract_data_file(LIVE_A6)

    assert not query_asks_for_defects_notification_period(A2_ASK)
    assert not query_asks_for_defects_notification_period(A3_ASK)
    assert not query_asks_for_defects_notification_period(A5_ASK)
    assert not query_asks_for_defects_notification_period(A9_ASK)
    assert not query_asks_for_defects_notification_period(C1_ASK)
    assert not query_asks_for_defects_notification_period(E1_ASK)
    assert not query_asks_for_defects_notification_period(F1_ASK)
    assert not query_asks_for_defects_notification_period(
        "What does Defects Notification Period mean under the contract?"
    )

    assert chunk_states_defects_notification_period(PREFIXED_DNP)
    assert chunk_states_defects_notification_period(SCANNED_DNP)
    assert not chunk_states_defects_notification_period(PSA_TOC)
    assert not chunk_states_defects_notification_period(CPM_REGISTER)
    assert not chunk_states_defects_notification_period(GC_POINTER)
    assert not chunk_states_defects_notification_period(GLOSSARY)
    assert not chunk_states_defects_notification_period(PREFIXED_TFC)
    assert not chunk_states_defects_notification_period(C1_LIST)

    assert extract_defects_notification_period(PREFIXED_DNP) == A6_DAYS
    assert extract_defects_notification_period(SCANNED_DNP) == A6_DAYS
    assert extract_defects_notification_period(PSA_TOC) is None
    assert extract_defects_notification_period(GC_POINTER) is None


def _install_a6_corpus(monkeypatch, *, semantic, rescue_hits, names, seeded=None):
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers or []).lower()
        out = []
        for chunk in rescue_hits:
            text_l = (chunk.text or "").lower()
            if "defects" in blob and "notification" in blob and "365" in text_l:
                out.append(chunk)
            if "1.1.27" in blob and "365" in text_l:
                out.append(chunk)
        return out[:k]

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        out = []
        for chunk in rescue_hits:
            if chunk.doc_id in (doc_ids or []):
                out.append(chunk)
        return out[: max(1, int(k_per_doc or 12))]

    seeded = list(seeded or [])

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
        lambda self, project_id, needles, k=20: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_following",
        lambda self, project_id, anchors, n=1: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 6,
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
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def _live_lookalikes_and_dnp():
    psa = _chunk("psa", PSA_DOC, 0.91, PSA_TOC, chunk_index=4)
    cpm = _chunk("cpm", CPM_DOC, 0.89, CPM_REGISTER, chunk_index=7)
    cpm16 = _chunk("c16", CPM16_DOC, 0.88, CPM_REGISTER, chunk_index=7)
    gc = _chunk("gc", GC_DOC, 0.86, GC_POINTER)
    dnp = _chunk("dnp", SCAN_DNP_DOC, 0.22, SCANNED_DNP)
    names = {
        PSA_DOC: PSA_NAME,
        CPM_DOC: CPM_NAME,
        CPM16_DOC: CPM16_NAME,
        GC_DOC: DD23_NAME,
        SCAN_DNP_DOC: CD_SCANNED_NAME,
    }
    seeded = [
        {"id": SCAN_DNP_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]
    return psa, cpm, cpm16, gc, dnp, names, seeded


def test_a6_elects_contract_data_when_psa_cpm_toc_lead(monkeypatch):
    """The live failure: PSA/CPM TOC occupy top-k; 365 days is out of pool."""
    psa, cpm, cpm16, gc, dnp, names, seeded = _live_lookalikes_and_dnp()
    ret = _install_a6_corpus(
        monkeypatch,
        semantic=[psa, cpm, cpm16, gc],
        rescue_hits=[dnp],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_A6, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks, "DNP row never reached the pool"
    assert A6_DAYS in blob
    assert A6_DAYS in chunks[0].text
    assert "Taking-Over Certificate" in chunks[0].text
    assert all(ret.chunk_states_defects_notification_period(c.text) for c in chunks)
    assert all(c.doc_id not in {PSA_DOC, CPM_DOC, CPM16_DOC} for c in chunks)


def test_a6_catalog_ask_still_elects_365_days(monkeypatch):
    psa, cpm, _c16, gc, dnp, names, seeded = _live_lookalikes_and_dnp()
    ret = _install_a6_corpus(
        monkeypatch,
        semantic=[psa, cpm, gc],
        rescue_hits=[dnp],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(A6_ASK, ACTIVE, k=5)
    assert chunks
    assert A6_DAYS in chunks[0].text
    assert chunks[0].doc_id == SCAN_DNP_DOC


def test_a6_drops_toc_when_dnp_is_already_in_the_pool(monkeypatch):
    psa, cpm, cpm16, gc, dnp, names, seeded = _live_lookalikes_and_dnp()
    prefixed = _chunk("pref", DNP_DOC, 0.40, PREFIXED_DNP)
    names = dict(names)
    names[DNP_DOC] = DD23_NAME
    ret = _install_a6_corpus(
        monkeypatch,
        semantic=[psa, cpm, cpm16, gc, prefixed],
        rescue_hits=[dnp],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_A6, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert A6_DAYS in chunks[0].text
    assert "table of contents" not in blob.lower()
    assert "document register" not in blob.lower()


def test_a2_a3_a5_a9_c1_e1_f1_are_not_stolen_onto_the_dnp_row(monkeypatch):
    psa, cpm, _c16, gc, dnp, names, seeded = _live_lookalikes_and_dnp()
    tfc = _chunk("tfc", TFC_DOC, 0.84, PREFIXED_TFC)
    rate = _chunk("rate", RATE_DOC, 0.83, PREFIXED_RATE)
    aca = _chunk("aca", ACA_DOC, 0.82, SCANNED_ACA)
    eng = _chunk("eng", ENG_DOC, 0.81, SCANNED_ENGINEER)
    listed = _chunk("c1", C1_DOC, 0.80, C1_LIST)
    names = dict(names)
    names[TFC_DOC] = DD23_NAME
    names[RATE_DOC] = DD23_NAME
    names[ACA_DOC] = CD_SCANNED_NAME
    names[ENG_DOC] = CD_SCANNED_NAME
    names[C1_DOC] = DD23_NAME
    ret = _install_a6_corpus(
        monkeypatch,
        semantic=[psa, tfc, rate, aca, eng, listed],
        rescue_hits=[dnp],
        names=names,
        seeded=seeded,
    )
    for ask in (A2_ASK, A3_ASK, A5_ASK, A9_ASK, C1_ASK, E1_ASK, F1_ASK):
        chunks, _ = ret.retrieve_with_filter(ask, ACTIVE, k=5)
        assert all(c.doc_id != SCAN_DNP_DOC for c in chunks), ask
        assert all(
            not ret.chunk_states_defects_notification_period(c.text) for c in chunks
        ), ask


def test_election_prefers_ab2023_dnp_over_psa_toc():
    from app.core.rag.retriever import elect_answer_bearing_contract

    ranked = [
        (PSA_NAME, PSA_TOC),
        (CPM_NAME, CPM_REGISTER),
        (CD_SCANNED_NAME, SCANNED_DNP),
    ]
    assert elect_answer_bearing_contract(A6_ASK, ranked) == "ab-2023-101"
    assert elect_answer_bearing_contract(LIVE_A6, ranked) == "ab-2023-101"


def _sys(*chunk_texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=doc{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(chunk_texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def test_graft_a6_states_365_days_when_model_could_not_confirm():
    rag = _sys(PSA_TOC, SCANNED_DNP)
    msgs = [{"role": "user", "content": LIVE_A6}]
    missing = (
        "I could not confirm the Defects Notification Period in the "
        "retrieved excerpts for this question. The retrieved context "
        "contains the Contract Data particulars documents (the Long "
        "Form PSA and the CPM contract data sheets), but the excerpts "
        "shown cover only the table of contents, recitals, and document "
        "registers — none of the retrieved chunks state a Defects "
        "Notification Period value or clause."
    )
    out = _graft_asked_contract_particular(missing, rag, msgs)
    assert A6_DAYS in out
    assert out.startswith("The Defects Notification Period")


def test_graft_a6_does_not_rewrite_a_correct_365_answer():
    rag = _sys(SCANNED_DNP)
    msgs = [{"role": "user", "content": LIVE_A6}]
    ok = (
        "The Defects Notification Period is 365 days from the "
        "Taking-Over Certificate under Contract Data 1.1.27."
    )
    assert _graft_asked_contract_particular(ok, rag, msgs) == ok


def test_graft_a6_replaces_generic_refusal_when_excerpt_has_365():
    rag = _sys(PREFIXED_DNP)
    msgs = [{"role": "user", "content": LIVE_A6}]
    out = _graft_asked_contract_particular(_CG_REFUSAL, rag, msgs)
    assert A6_DAYS in out
    assert "The Defects Notification Period" in out


def test_graft_does_not_steal_a2_a3_a5_a9_c1_e1(monkeypatch):
    rag = _sys(PREFIXED_DNP, PREFIXED_TFC, SCANNED_ACA)
    for ask in (A2_ASK, A3_ASK, A5_ASK, A9_ASK, C1_ASK, E1_ASK, F1_ASK):
        msgs = [{"role": "user", "content": LIVE_PREFIX + ask}]
        body = "I could not confirm this particular in the retrieved excerpts."
        out = _graft_asked_contract_particular(body, rag, msgs)
        if ask == A3_ASK:
            assert A3_DAYS in out
            continue
        if ask == A2_ASK:
            continue
        assert "Defects Notification Period is 365" not in out, ask


def test_postprocess_grafts_a6_on_the_live_refusal():
    rag = _sys(SCANNED_DNP)
    msgs = [{"role": "user", "content": LIVE_A6}]
    missing = (
        "I could not confirm the Defects Notification Period in the "
        "retrieved excerpts."
    )
    out = _postprocess_answer(missing, rag, msgs)
    assert A6_DAYS in out
    assert "Defects Notification Period" in out
