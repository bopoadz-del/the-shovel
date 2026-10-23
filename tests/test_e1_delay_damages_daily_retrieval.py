"""OLD-pack E1: retrieve rate AND ACA so compose can state SAR/day.

Live Master Corpus (tip 8f4b465 / leftover pack E1):

* A5 PASS — rate rescue surfaces ``0.1% of the Contract Price per
  calendar day``.
* A2 PASS — filename / incl-VAT rescue surfaces the ACA.
* E1 FAIL — "Calculate the delay damages per calendar day in SAR for
  the whole of the Works" retrieved Spec TOC / Daywork / insurance
  and refused to compute. Expected SAR 1,754,504.46/day = 0.1% of
  excl-VAT ACA SAR 1,754,504,456.25.

E1 is a monetary-base ask, so A5's exclusive rate rescue is correctly
off (that fence would drop the ACA). This path is a dual-operand
rescue: scanned Contract Data rate + ACA, then the existing compose
graft. Do not steal A2/A3/A5/A6/A9.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.runtime import (
    _CG_REFUSAL,
    _graft_composed_delay_damages_daily,
    _postprocess_answer,
)
from app.core.rag.vector_store import Chunk
from app.lib.construction_formulas_commercial import (
    compose_delay_damages_daily_from_excerpts,
    query_asks_delay_damages_daily_amount,
)


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
F1_ASK = CATALOG["cases"]["F1"]["ask"]
LIVE_PREFIX = "Answer only from the client project documents. "
LIVE_E1 = LIVE_PREFIX + (
    "Calculate the delay damages per calendar day in SAR for the "
    "whole of the Works."
)
LIVE_A5 = LIVE_PREFIX + A5_ASK

# Fixture figures already used by E1 compose / A2 graft tests.
NET_ACA = 1_754_504_456.25
GROSS_ACA = 2_017_680_124.69
DAILY = 1_754_504.46
RATE = "0.1% of the Contract Price per calendar day"
NET_ACA_TXT = f"SAR {NET_ACA:,.2f}"
GROSS_ACA_TXT = f"SAR {GROSS_ACA:,.2f}"

DD23_NAME = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Vol 1 - Conditions of Contract.pdf"
)
CD_SCANNED_NAME = (
    "AB-2023-101_QP2 Infra P1_Vol 1.0_Cond of Contract "
    "(complete)_Contract Data.pdf"
)
SPEC_NAME = "AB-2023-101_Vol 2 Specifications — Table of Contents.pdf"
DAYWORK_NAME = "Daywork Schedule — Labour per calendar day.pdf"
INSURANCE_NAME = "Works All Risks Insurance Policy.pdf"

SCANNED_RATE = (
    "CONTRACT DATA\n"
    "8.8\nDelay \nDamages for the whole of the Works\n"
    f"{RATE}\n"
)
SCANNED_NET_ACA = (
    "CONTRACT DATA\n"
    "1.1.1\nAccepted\nContract\nAmount (excluding VAT)\n"
    f"{NET_ACA_TXT}\n"
)
SCANNED_GROSS_ACA = (
    "CONTRACT DATA\n"
    "1.1.1\nAccepted\nContract\nAmount (including VAT)\n"
    f"{GROSS_ACA_TXT}\n"
)
SPEC_TOC = (
    "Specification table of contents. Section 01 The Works. "
    "Section 02 Materials. Daywork is scheduled separately. "
    "No delay damages rate is stated here."
)
DAYWORK = (
    "Daywork Schedule. Labour per calendar day. Plant per hour. "
    "Rates in SAR for measured dayworks, not liquidated damages."
)
INSURANCE = (
    "Works All Risks insurance policy. Sum insured "
    "SAR 50,000,000.00. Period of insurance for the whole of the Works."
)
GC_8_8 = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.8 Delay Damages. "
    "The Contractor shall pay delay damages for the whole of the Works "
    "at the rate stated in the Contract Data for every calendar day."
)
# Live leftover E1 on c5c6dfa: chunks 9–11 of the bound Contract Data
# volume stated the 0.1% rate AND a FIDIC worked-example ACA of
# SAR 10,000,000. Retrieval treated that as the money operand, so
# compose emitted SAR 10,000/day.
COC_8_8_TOY_ACA = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.8 Delay Damages. "
    "The Contractor shall pay delay damages for the whole of the Works "
    f"at {RATE}. For example, if the Accepted Contract Amount "
    "excluding VAT is SAR 10,000,000.00, the daily amount is "
    "SAR 10,000.00."
)
TOY_ACA_TXT = "SAR 10,000,000.00"
PREFIXED_TFC = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "1.1.75 Time for Completion for the whole of the Works: 852 days"
)
PREFIXED_DNP = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD23_NAME}].\n"
    "1.1.27 Defects Notification Period: 365 days from the Taking-Over Certificate"
)
CANNOT_CALCULATE = (
    "I cannot calculate the delay damages per calendar day from the "
    "reference context provided. Retrieved excerpts include the "
    "Specification table of contents, the Daywork Schedule, and an "
    "insurance policy."
)

ACTIVE = "p_master"
SPEC_DOC = "spec118"
DAY_DOC = "daywork"
INS_DOC = "insure"
RATE_DOC = "scanrate"
ACA_DOC = "scanaca"
GROSS_DOC = "scangross"
GC_DOC = "gc118"
TFC_DOC = "tfc118"
DNP_DOC = "dnp118"


def _chunk(
    cid, doc_id, score, text, project_id=ACTIVE, chunk_index=0, source_name="",
):
    return Chunk(
        chunk_id=cid,
        project_id=project_id,
        doc_id=doc_id,
        chunk_index=chunk_index,
        text=text,
        score=score,
        source_name=source_name or "",
    )


def _sys(*chunk_texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=doc{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(chunk_texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def test_catalog_and_ask_class_do_not_steal_a5():
    from app.core.rag.retriever import (
        query_asks_delay_damages_daily_amount as ret_asks,
        query_asks_for_delay_damages_rate,
        query_needs_a_monetary_base,
        query_wants_contract_data_file,
    )

    assert E1_ASK == (
        "Calculate the delay damages per calendar day in SAR for the "
        "whole of the Works."
    )
    assert query_asks_delay_damages_daily_amount(E1_ASK)
    assert query_asks_delay_damages_daily_amount(LIVE_E1)
    assert ret_asks(E1_ASK) == query_asks_delay_damages_daily_amount(E1_ASK)
    assert query_needs_a_monetary_base(LIVE_E1)
    assert query_wants_contract_data_file(LIVE_E1)

    assert not query_asks_for_delay_damages_rate(E1_ASK)
    assert not query_asks_for_delay_damages_rate(LIVE_E1)
    assert not query_asks_delay_damages_daily_amount(A5_ASK)
    assert not query_asks_delay_damages_daily_amount(LIVE_A5)
    assert not query_asks_delay_damages_daily_amount(A2_ASK)
    assert not query_asks_delay_damages_daily_amount(A3_ASK)
    assert not query_wants_contract_data_file(A5_ASK)


def test_scanned_operands_and_lookalikes():
    from app.core.rag.retriever import (
        chunk_answers_asked_particular,
        chunk_states_accepted_contract_amount,
        chunk_states_delay_damages_rate,
    )

    assert chunk_states_delay_damages_rate(SCANNED_RATE)
    assert chunk_states_accepted_contract_amount(SCANNED_NET_ACA)
    assert chunk_states_accepted_contract_amount(SCANNED_GROSS_ACA)
    assert not chunk_states_accepted_contract_amount(SCANNED_RATE)
    assert not chunk_states_delay_damages_rate(SCANNED_NET_ACA)
    assert not chunk_states_accepted_contract_amount(INSURANCE)
    assert not chunk_states_accepted_contract_amount(DAYWORK)
    assert not chunk_states_delay_damages_rate(GC_8_8)
    assert not chunk_states_accepted_contract_amount(COC_8_8_TOY_ACA)
    assert chunk_states_delay_damages_rate(COC_8_8_TOY_ACA)

    assert chunk_answers_asked_particular(LIVE_E1, SCANNED_RATE)
    assert chunk_answers_asked_particular(LIVE_E1, SCANNED_NET_ACA)
    assert not chunk_answers_asked_particular(LIVE_E1, SPEC_TOC)
    assert not chunk_answers_asked_particular(LIVE_E1, GC_8_8)
    # A5 still answers from the rate only — ACA is the neighboring field.
    assert chunk_answers_asked_particular(A5_ASK, SCANNED_RATE)
    assert not chunk_answers_asked_particular(A5_ASK, SCANNED_NET_ACA)


def _install_e1_corpus(monkeypatch, *, semantic, rescue_hits, names, seeded=None):
    from app.core.rag import retriever as ret

    seeded = list(seeded or [])

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers or []).lower()
        out = []
        for chunk in rescue_hits:
            text_l = (chunk.text or "").lower()
            if "delay" in blob and "damages" in blob and "0.1%" in text_l:
                out.append(chunk)
            if (
                "accepted contract" in blob
                and "accepted" in text_l
                and "sar" in text_l
            ):
                out.append(chunk)
        return out[:k]

    def fake_chunks_for_docs(
        self, project_id, doc_ids, k_per_doc=12, from_end=False,
        all_rows=False,
    ):
        hits = []
        for chunk in rescue_hits:
            if chunk.doc_id in (doc_ids or []):
                hits.append(chunk)
        return hits

    def fake_containing_all(self, project_id, needles, k=20, doc_ids=None):
        needles_l = [str(n).lower() for n in (needles or [])]
        out = []
        for chunk in rescue_hits:
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            text_l = (chunk.text or "").lower()
            if needles_l and all(n in text_l for n in needles_l):
                out.append(chunk)
        return out[: max(1, int(k or 20))]

    def fake_title_match(pid, phrase, limit=8):
        if "contract data" not in (phrase or "").lower():
            return []
        return [d for d in seeded if "contract data" in (d.get("original_name") or "").lower()]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
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


def _live_lookalikes_and_operands():
    spec = _chunk("spec", SPEC_DOC, 0.93, SPEC_TOC)
    day = _chunk("day", DAY_DOC, 0.91, DAYWORK)
    ins = _chunk("ins", INS_DOC, 0.89, INSURANCE)
    rate = _chunk("rate", RATE_DOC, 0.22, SCANNED_RATE)
    aca = _chunk("aca", ACA_DOC, 0.21, SCANNED_NET_ACA)
    names = {
        SPEC_DOC: SPEC_NAME,
        DAY_DOC: DAYWORK_NAME,
        INS_DOC: INSURANCE_NAME,
        RATE_DOC: CD_SCANNED_NAME,
        ACA_DOC: CD_SCANNED_NAME,
    }
    seeded = [
        {"id": RATE_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": ACA_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]
    return spec, day, ins, rate, aca, names, seeded


def test_e1_surfaces_rate_and_excl_aca_when_spec_daywork_insurance_lead(monkeypatch):
    spec, day, ins, rate, aca, names, seeded = _live_lookalikes_and_operands()
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[spec, day, ins],
        rescue_hits=[rate, aca],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks, "operands never reached the pool"
    assert RATE in blob
    assert NET_ACA_TXT in blob
    assert "Specification table of contents" not in blob
    assert "Daywork Schedule" not in blob
    assert "Works All Risks" not in blob
    assert all(
        chunk_id in {RATE_DOC, ACA_DOC}
        for chunk_id in (c.doc_id for c in chunks)
    )


def test_e1_compose_from_retrieved_excerpts_states_the_daily_sar(monkeypatch):
    spec, day, ins, rate, aca, names, seeded = _live_lookalikes_and_operands()
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[spec, day, ins],
        rescue_hits=[rate, aca],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=5)
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["currency"] == "SAR"

    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    grafted = _graft_composed_delay_damages_daily(CANNOT_CALCULATE, rag, msgs)
    assert "1,754,504.46" in grafted
    assert "cannot calculate" not in grafted.lower()
    posted = _postprocess_answer(CANNOT_CALCULATE, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL


def test_e1_live_prefix_and_catalog_ask_both_rescue(monkeypatch):
    spec, day, ins, rate, aca, names, seeded = _live_lookalikes_and_operands()
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[spec, day, ins],
        rescue_hits=[rate, aca],
        names=names,
        seeded=seeded,
    )
    for ask in (E1_ASK, LIVE_E1):
        chunks, _ = ret.retrieve_with_filter(ask, ACTIVE, k=5)
        blob = " ".join(c.text for c in chunks)
        assert RATE in blob and NET_ACA_TXT in blob, ask


def test_a5_still_surfaces_the_rate_without_requiring_aca(monkeypatch):
    spec, day, ins, rate, aca, names, seeded = _live_lookalikes_and_operands()
    gc = _chunk("gc", GC_DOC, 0.94, GC_8_8)
    names = dict(names)
    names[GC_DOC] = DD23_NAME
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[gc, spec, day],
        rescue_hits=[rate, aca],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_A5, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert RATE in blob
    assert RATE in chunks[0].text
    assert "at the rate stated in the Contract Data" not in blob
    # Exclusive A5 fence: the money row is the neighboring field.
    assert NET_ACA_TXT not in blob
    assert compose_delay_damages_daily_from_excerpts(
        LIVE_A5, blob,
    ) is None


def test_a2_a3_a6_a9_are_not_stolen_onto_the_e1_rescue(monkeypatch):
    spec, _day, _ins, rate, aca, names, _seeded = _live_lookalikes_and_operands()
    tfc = _chunk("tfc", TFC_DOC, 0.84, PREFIXED_TFC)
    dnp = _chunk("dnp", DNP_DOC, 0.83, PREFIXED_DNP)
    names = dict(names)
    names[TFC_DOC] = DD23_NAME
    names[DNP_DOC] = DD23_NAME
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[spec, tfc, dnp],
        rescue_hits=[rate, aca],
        names=names,
    )
    a3, _ = ret.retrieve_with_filter(A3_ASK, ACTIVE, k=5)
    a6, _ = ret.retrieve_with_filter(A6_ASK, ACTIVE, k=5)
    a2, _ = ret.retrieve_with_filter(A2_ASK, ACTIVE, k=5)
    a9, _ = ret.retrieve_with_filter(A9_ASK, ACTIVE, k=5)
    for label, chunks in (("A3", a3), ("A6", a6), ("A9", a9)):
        assert all(c.doc_id not in {RATE_DOC, ACA_DOC} for c in chunks), label
    assert any("852 days" in (c.text or "") for c in a3)
    assert any("365 days" in (c.text or "") for c in a6)
    # A2 may pull Contract Data via filename rescue; it must not be
    # fenced onto the delay-damages rate row.
    assert all(RATE not in (c.text or "") for c in a2)


def test_incl_vat_twin_does_not_replace_the_net_base(monkeypatch):
    spec, day, ins, rate, aca, names, seeded = _live_lookalikes_and_operands()
    gross = _chunk("gross", GROSS_DOC, 0.20, SCANNED_GROSS_ACA)
    names = dict(names)
    names[GROSS_DOC] = CD_SCANNED_NAME
    seeded = list(seeded) + [
        {"id": GROSS_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[spec, day, ins],
        rescue_hits=[rate, aca, gross],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=5)
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] == DAILY
    assert GROSS_ACA != out["contract_amount"]


def test_e1_is_not_classified_as_a2():
    from app.core.rag.retriever import query_asks_for_aca_including_vat

    assert not query_asks_for_aca_including_vat(LIVE_E1)
    assert not query_asks_for_aca_including_vat(E1_ASK)
    assert query_asks_delay_damages_daily_amount(LIVE_E1)


def test_e1_surfaces_rate_and_excl_when_coc_and_incl_vat_lead(monkeypatch):
    """Live leftover E1 after #523: CoC 8.8 chunks 9–11 + including-VAT.

    The bound Contract Data volume's Sub-Clause 8.8 windows occupied
    top-k (filename keep) and the last-slot money reserve elected the
    A2 including-VAT twin. Compose never saw 0.1%; the answer was
    only SAR 2,017,680,124.69.
    """
    gc9 = _chunk("gc9", GC_DOC, 0.94, GC_8_8)
    gc10 = _chunk("gc10", GC_DOC, 0.93, GC_8_8)
    gc11 = _chunk("gc11", GC_DOC, 0.92, GC_8_8)
    gross = _chunk("gross", GROSS_DOC, 0.91, SCANNED_GROSS_ACA)
    rate = _chunk("rate", RATE_DOC, 0.22, SCANNED_RATE)
    aca = _chunk("aca", ACA_DOC, 0.21, SCANNED_NET_ACA)
    names = {
        GC_DOC: CD_SCANNED_NAME,
        GROSS_DOC: CD_SCANNED_NAME,
        RATE_DOC: CD_SCANNED_NAME,
        ACA_DOC: CD_SCANNED_NAME,
    }
    seeded = [
        {"id": RATE_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": ACA_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": GROSS_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[gc9, gc10, gc11, gross],
        rescue_hits=[rate, aca, gross],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert RATE in blob
    assert NET_ACA_TXT in blob
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["contract_amount"] == NET_ACA
    assert out["currency"] == "SAR"

    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    live_fail = (
        "The Accepted Contract Amount including VAT is "
        f"{GROSS_ACA_TXT}."
    )
    posted = _postprocess_answer(live_fail, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != live_fail
    assert posted != _CG_REFUSAL


def test_e1_inject_hint_is_compose_not_including_vat():
    from app.core.rag.inject import format_chunks_as_system_message

    rate = _chunk("rate", RATE_DOC, 0.9, SCANNED_RATE)
    aca = _chunk("aca", ACA_DOC, 0.8, SCANNED_NET_ACA)
    gross = _chunk("gross", GROSS_DOC, 0.7, SCANNED_GROSS_ACA)
    e1 = format_chunks_as_system_message(
        [rate, aca, gross], 1, query=LIVE_E1,
    )["content"]
    assert "DELAY DAMAGES PER CALENDAR DAY" in e1
    assert "including VAT. That IS the answer" not in e1
    a2 = format_chunks_as_system_message(
        [gross], 1, query=LIVE_PREFIX + A2_ASK,
    )["content"]
    assert "INCLUDING VAT" in a2
    assert "Lead with that including-VAT figure" in a2


def test_e1_surfaces_excl_vat_when_coc_chunks_carry_toy_ten_million(monkeypatch):
    """Live leftover E1 on c5c6dfa: chunks 9–11 + toy ACA 10,000,000.

    The bound Contract Data volume's Sub-Clause 8.8 windows stated the
    0.1% rate and a FIDIC worked-example ACA of SAR 10,000,000. That
    money satisfied the reservation, so the filled excl-VAT row never
    entered top-k and compose emitted SAR 10,000/day.
    """
    gc9 = _chunk("gc9", GC_DOC, 0.94, COC_8_8_TOY_ACA)
    gc10 = _chunk("gc10", GC_DOC, 0.93, COC_8_8_TOY_ACA)
    gc11 = _chunk("gc11", GC_DOC, 0.92, COC_8_8_TOY_ACA)
    rate = _chunk("rate", RATE_DOC, 0.22, SCANNED_RATE)
    aca = _chunk("aca", ACA_DOC, 0.21, SCANNED_NET_ACA)
    names = {
        GC_DOC: CD_SCANNED_NAME,
        RATE_DOC: CD_SCANNED_NAME,
        ACA_DOC: CD_SCANNED_NAME,
    }
    seeded = [
        {"id": RATE_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": ACA_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=[gc9, gc10, gc11],
        rescue_hits=[rate, aca],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert RATE in blob
    assert NET_ACA_TXT in blob
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["contract_amount"] == NET_ACA
    assert out["currency"] == "SAR"
    assert out["contract_amount"] != 10_000_000.0

    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    live_fail = (
        "Delay damages for the whole of the Works are "
        "SAR 10,000.00 per calendar day "
        f"(0.1% of Accepted Contract Amount {TOY_ACA_TXT})."
    )
    posted = _postprocess_answer(live_fail, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != live_fail
    assert posted != _CG_REFUSAL
    assert "10,000,000.00" not in posted.split("\n", 1)[0]


# Live leftover E1 after #529: the filled excl-VAT row still carried the
# Contract Data "[insert amount]" template cue. Toy-reject stained that
# figure, rescue dropped it, top-k stayed on chunks 9–11, and compose
# fell through to the cost-grounding refusal.
INSERT_STAINED_NET_ACA = (
    "CONTRACT DATA\n1.1.1\nAccepted\nContract\nAmount (excluding VAT)\n"
    f"[insert amount] {NET_ACA_TXT}\n"
)


def test_insert_stained_excl_vat_is_still_an_e1_money_operand():
    from app.core.rag.retriever import chunk_states_accepted_contract_amount
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    assert chunk_has_real_accepted_contract_amount(INSERT_STAINED_NET_ACA)
    assert chunk_states_accepted_contract_amount(INSERT_STAINED_NET_ACA)
    assert not chunk_states_accepted_contract_amount(COC_8_8_TOY_ACA)


def test_e1_surfaces_insert_stained_excl_vat_when_toy_windows_fill_k(
    monkeypatch,
):
    """Live after #529: 8.8 chunks 9–11 occupy every slot as rate windows.

    Toy reject cleared them as the money operand. The filled excl-VAT
    row sat next to ``[insert amount]`` and was stained as a toy, so
    it never entered top-k. Compose then had no rate base and the
    cost-grounding gate refused.
    """
    toys = [
        _chunk(f"gc{i}", GC_DOC, 0.95 - i * 0.01, COC_8_8_TOY_ACA)
        for i in range(5)
    ]
    rate = _chunk("rate", RATE_DOC, 0.22, SCANNED_RATE)
    aca = _chunk("aca", ACA_DOC, 0.21, INSERT_STAINED_NET_ACA)
    names = {
        GC_DOC: CD_SCANNED_NAME,
        RATE_DOC: CD_SCANNED_NAME,
        ACA_DOC: CD_SCANNED_NAME,
    }
    seeded = [
        {"id": RATE_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": ACA_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]
    ret = _install_e1_corpus(
        monkeypatch,
        semantic=toys,
        rescue_hits=[rate, aca],
        names=names,
        seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert RATE in blob
    assert NET_ACA_TXT in blob
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["contract_amount"] == NET_ACA
    assert out["currency"] == "SAR"
    assert out["contract_amount"] != 10_000_000.0

    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(_CG_REFUSAL, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()
    assert "10,000,000.00" not in posted.split("\n", 1)[0]


# Live leftover E1 after #530/#532 (tip eb278c0): sources cited were
# only Contract Data 8.8 chunks 9–11. identifier_search for "accepted
# contract amount excluding vat" matches those worked-example windows
# and LIMIT returns them first. chunks_for_docs first-N (24/40, then
# #532's 400) stays on the Conditions body — the filled 1.1.1 excl-VAT
# row sits in a later appendix of the same complete volume. Compose
# skipped the toy 10M and the cost-grounding gate refused.
# 450 > #532's 400-prefix cap so a prefix-only scan still misses.
LATE_ACA_INDEX = 450
# Live 77a96ac after #533: filled row sits in the MIDDLE of a long
# combined volume (past first-400 AND before last-400). Scanned text
# is "excl. VAT" without a 1.1.1 clause — #533 needles miss.
MIDDLE_ACA_INDEX = 500
MIDDLE_DOC_LEN = 1200
LIVE_SCANNED_EXCL_ACA = (
    "CONTRACT DATA\n"
    "Accepted\nContract\nAmount (excl. VAT)\n"
    f"{NET_ACA_TXT}\n"
)


def _install_live_e1_late_aca_corpus(monkeypatch):
    """Toy 8.8 at chunks 9–11; filled excl-VAT past first-400, same doc."""
    from app.core.rag import retriever as ret
    from app.core.rag.retriever import chunk_states_delay_damages_rate

    toys = [
        _chunk(f"gc{i}", GC_DOC, 0.95 - i * 0.01, COC_8_8_TOY_ACA, chunk_index=9 + i)
        for i in range(3)
    ]
    # Combined GC+CD volume: first-40 / first-400 stay in the Conditions
    # body. Pointer windows are not E1 operands. Pad past #532's 400
    # prefix so the filled 1.1.1 row is only reachable via tail/text.
    dummies = [
        _chunk(f"pre{i}", GC_DOC, 0.10, GC_8_8, chunk_index=i)
        for i in range(LATE_ACA_INDEX)
        if i not in (9, 10, 11)
    ]
    aca = _chunk(
        "aca450", GC_DOC, 0.21, SCANNED_NET_ACA, chunk_index=LATE_ACA_INDEX,
    )
    all_chunks = list(dummies) + list(toys) + [aca]
    names = {GC_DOC: CD_SCANNED_NAME}
    seeded = [
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in toys if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers or []).lower()
        out = []
        for chunk in all_chunks:
            text_l = (chunk.text or "").lower()
            if "delay" in blob and "damages" in blob and "0.1%" in text_l:
                out.append(chunk)
            elif "accepted contract" in blob:
                # Live LIMIT: 8.8 toy windows match the excl-VAT phrase
                # and fill k before the later 1.1.1 row.
                if chunk_states_delay_damages_rate(chunk.text or ""):
                    out.append(chunk)
        return out[:k]

    def fake_chunks_for_docs(
        self, project_id, doc_ids, k_per_doc=12, from_end=False,
        all_rows=False,
    ):
        by_doc: dict[str, list] = {}
        for chunk in all_chunks:
            if chunk.doc_id in (doc_ids or []):
                by_doc.setdefault(chunk.doc_id, []).append(chunk)
        out = []
        for did in doc_ids or []:
            rows = sorted(by_doc.get(did, []), key=lambda c: c.chunk_index)
            if all_rows:
                out.extend(rows)
                continue
            n = max(1, int(k_per_doc or 12))
            out.extend(rows[-n:] if from_end else rows[:n])
        return out

    def fake_containing_all(self, project_id, needles, k=20, doc_ids=None):
        needles_l = [str(n).lower() for n in (needles or [])]
        out = []
        for chunk in all_chunks:
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            text_l = (chunk.text or "").lower()
            if needles_l and all(n in text_l for n in needles_l):
                out.append(chunk)
        return out[: max(1, int(k or 20))]

    def fake_title_match(pid, phrase, limit=8):
        if "contract data" not in (phrase or "").lower():
            return []
        return list(seeded)

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
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
    return ret, toys, aca


def test_e1_surfaces_late_excl_vat_when_id_search_returns_only_8_8_toys(
    monkeypatch,
):
    """Live after #532: toy-only 8.8 top-k, filled excl-VAT past first-400.

    identifier_search returns the 8.8 windows (they name excl-VAT).
    First-24/40/400 chunks_for_docs stay on those early indexes. The
    filled 1.1.1 row must still enter top-k so compose can state SAR/day
    — not the cost-grounding refuse, not 10k/day, not 0.015%.
    """
    ret, _toys, _aca = _install_live_e1_late_aca_corpus(monkeypatch)
    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert RATE in blob
    assert NET_ACA_TXT in blob
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["contract_amount"] == NET_ACA
    assert out["currency"] == "SAR"
    assert out["daily_amount"] != 10_000.0
    assert out["contract_amount"] != 10_000_000.0

    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(_CG_REFUSAL, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()
    assert "10,000.00" not in posted.split("\n", 1)[0]
    assert "0.015%" not in posted
    assert "263,175.67" not in posted


def test_e1_late_aca_helper_scans_past_first_n_on_rate_docs():
    """Direct: fused has only 8.8 toys; first-400 misses chunk 450."""
    from app.core.rag.retriever import (
        _DOC_SCAN_WINDOW,
        recall_composition_operands,
    )
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    toys = [
        _chunk(f"gc{i}", GC_DOC, 0.9, COC_8_8_TOY_ACA, chunk_index=9 + i)
        for i in range(3)
    ]
    dummies = [
        _chunk(f"pre{i}", GC_DOC, 0.1, GC_8_8, chunk_index=i)
        for i in range(LATE_ACA_INDEX)
        if i not in (9, 10, 11)
    ]
    aca = _chunk("aca450", GC_DOC, 0.2, SCANNED_NET_ACA, chunk_index=LATE_ACA_INDEX)
    all_chunks = list(dummies) + list(toys) + [aca]

    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            if all_rows:
                return rows
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            needles_l = [str(n).lower() for n in (needles or [])]
            out = []
            for chunk in all_chunks:
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue
                text_l = (chunk.text or "").lower()
                if needles_l and all(n in text_l for n in needles_l):
                    out.append(chunk)
            return out[: max(1, int(k or 20))]

    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in toys}
    assert not any(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )
    prefix24 = _Store().chunks_for_docs(ACTIVE, [GC_DOC], k_per_doc=24)
    prefix400 = _Store().chunks_for_docs(
        ACTIVE, [GC_DOC], k_per_doc=_DOC_SCAN_WINDOW,
    )
    assert all(c.chunk_index < LATE_ACA_INDEX for c in prefix24)
    assert all(c.chunk_index < LATE_ACA_INDEX for c in prefix400)
    # #532 cap: a bigger first-N still misses the live appendix.
    assert _DOC_SCAN_WINDOW < LATE_ACA_INDEX
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _Store(),
    )
    assert recovered >= 1
    assert any(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )
    excerpts = "\n\n".join(c.text or "" for c, _s, _b in fused.values())
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["contract_amount"] != 10_000_000.0
    assert out["daily_amount"] != 10_000.0
    assert out["daily_amount"] != 263_175.67


def _late_aca_all_chunks():
    toys = [
        _chunk(f"gc{i}", GC_DOC, 0.9, COC_8_8_TOY_ACA, chunk_index=9 + i)
        for i in range(3)
    ]
    dummies = [
        _chunk(f"pre{i}", GC_DOC, 0.1, GC_8_8, chunk_index=i)
        for i in range(LATE_ACA_INDEX)
        if i not in (9, 10, 11)
    ]
    aca = _chunk("aca450", GC_DOC, 0.2, SCANNED_NET_ACA, chunk_index=LATE_ACA_INDEX)
    return list(dummies) + list(toys) + [aca], toys, aca


def test_e1_late_aca_text_match_recovers_when_prefix_and_tail_miss():
    """Needles scoped to the rate-window doc find 1.1.1 past first-400."""
    from app.core.rag.retriever import recall_composition_operands
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    all_chunks, toys, _aca = _late_aca_all_chunks()

    class _TextOnly:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            # Live #532 shape: prefix only, never the appendix.
            if all_rows or from_end:
                return []
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            return rows[: max(1, int(k_per_doc or 12))]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            needles_l = [str(n).lower() for n in (needles or [])]
            out = []
            for chunk in all_chunks:
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue
                text_l = (chunk.text or "").lower()
                if needles_l and all(n in text_l for n in needles_l):
                    out.append(chunk)
            return out[: max(1, int(k or 20))]

    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in toys}
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _TextOnly(),
    )
    assert recovered >= 1
    assert any(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )


def test_e1_late_aca_tail_recovers_when_text_match_is_absent():
    """Last-N of the same volume surfaces the appendix when LIKE is missing."""
    from app.core.rag.retriever import (
        _DOC_SCAN_WINDOW,
        recall_composition_operands,
    )
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    all_chunks, toys, _aca = _late_aca_all_chunks()

    class _TailOnly:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            if all_rows:
                return []
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in toys}
    prefix = _TailOnly().chunks_for_docs(
        ACTIVE, [GC_DOC], k_per_doc=_DOC_SCAN_WINDOW, from_end=False,
    )
    assert all(c.chunk_index < LATE_ACA_INDEX for c in prefix)
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _TailOnly(),
    )
    assert recovered >= 1
    assert any(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )


def _middle_aca_all_chunks():
    """Toys at 9–11; filled excl. VAT at 500; volume length 1200."""
    toys = [
        _chunk(f"gc{i}", GC_DOC, 0.95 - i * 0.01, COC_8_8_TOY_ACA, chunk_index=9 + i)
        for i in range(3)
    ]
    dummies = [
        _chunk(f"pre{i}", GC_DOC, 0.10, GC_8_8, chunk_index=i)
        for i in range(MIDDLE_DOC_LEN)
        if i not in (9, 10, 11, MIDDLE_ACA_INDEX)
    ]
    aca = _chunk(
        "aca500", GC_DOC, 0.21, LIVE_SCANNED_EXCL_ACA, chunk_index=MIDDLE_ACA_INDEX,
    )
    return list(dummies) + list(toys) + [aca], toys, aca


def test_live_scanned_excl_vat_is_a_real_e1_money_operand():
    from app.core.rag.retriever import chunk_states_accepted_contract_amount
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
        parse_accepted_contract_amount,
    )

    assert "1.1.1" not in LIVE_SCANNED_EXCL_ACA
    assert "excluding" not in LIVE_SCANNED_EXCL_ACA.lower()
    assert chunk_has_real_accepted_contract_amount(LIVE_SCANNED_EXCL_ACA)
    assert parse_accepted_contract_amount(LIVE_SCANNED_EXCL_ACA) == (NET_ACA, "SAR")
    assert chunk_states_accepted_contract_amount(LIVE_SCANNED_EXCL_ACA)


def test_e1_533_prefix_tail_needles_miss_middle_excl_vat():
    """#533 hole: first-400, last-400, and 1.1.1+excluding all miss."""
    from app.core.rag.retriever import _DOC_SCAN_WINDOW

    all_chunks, _toys, aca = _middle_aca_all_chunks()
    rows = sorted(
        [c for c in all_chunks if c.doc_id == GC_DOC],
        key=lambda c: c.chunk_index,
    )
    prefix = rows[:_DOC_SCAN_WINDOW]
    tail = rows[-_DOC_SCAN_WINDOW:]
    assert all(c.chunk_index != MIDDLE_ACA_INDEX for c in prefix)
    assert all(c.chunk_index != MIDDLE_ACA_INDEX for c in tail)
    assert aca.chunk_index == MIDDLE_ACA_INDEX
    text_l = (aca.text or "").lower()
    assert "1.1.1" not in text_l
    assert "excluding" not in text_l


def test_e1_full_doc_scan_surfaces_middle_excl_vat_when_533_misses():
    """Walk every chunk of the 8.8 doc — not prefix, tail, or 1.1.1 LIKE."""
    from app.core.rag.retriever import recall_composition_operands
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    all_chunks, toys, _aca = _middle_aca_all_chunks()

    class _Live533Hole:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            if all_rows:
                return rows
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            needles_l = [str(n).lower() for n in (needles or [])]
            if "1.1.1" in needles_l or "excluding" in needles_l:
                return []
            out = []
            for chunk in all_chunks:
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue
                text_l = (chunk.text or "").lower()
                if needles_l and all(n in text_l for n in needles_l):
                    out.append(chunk)
            return out[: max(1, int(k or 20))]

    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in toys}
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _Live533Hole(),
    )
    assert recovered >= 1
    assert any(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )
    excerpts = "\n\n".join(c.text or "" for c, _s, _b in fused.values())
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != 10_000.0
    assert out["daily_amount"] != 263_175.67


def test_e1_excl_vat_needles_recover_when_all_rows_and_prefix_tail_miss():
    """#533 needles required 1.1.1+excluding. Live scanned excl. VAT has neither.

    When all_rows is absent (TypeError / empty), excl+vat LIKE must
    still surface the middle row so compose can state SAR/day.
    """
    from app.core.rag.retriever import recall_composition_operands
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    all_chunks, toys, _aca = _middle_aca_all_chunks()

    class _NeedlesOnly:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            if all_rows:
                return []
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            needles_l = [str(n).lower() for n in (needles or [])]
            if "1.1.1" in needles_l or "excluding" in needles_l:
                return []
            out = []
            for chunk in all_chunks:
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue
                text_l = (chunk.text or "").lower()
                if needles_l and all(n in text_l for n in needles_l):
                    out.append(chunk)
            return out[: max(1, int(k or 20))]

    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in toys}
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _NeedlesOnly(),
    )
    assert recovered >= 1
    assert any(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )
    excerpts = "\n\n".join(c.text or "" for c, _s, _b in fused.values())
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["contract_amount"] == NET_ACA


def test_e1_retrieve_middle_excl_vat_when_top_k_is_only_chunks_9_11(monkeypatch):
    """Live 77a96ac: sources were only 8.8 chunks 9–11. Must still compose."""
    from app.core.rag import retriever as ret
    from app.core.rag.retriever import chunk_states_delay_damages_rate

    all_chunks, toys, _aca = _middle_aca_all_chunks()
    names = {GC_DOC: CD_SCANNED_NAME}
    seeded = [
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in toys if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers or []).lower()
        out = []
        for chunk in toys:
            text_l = (chunk.text or "").lower()
            if "delay" in blob and "damages" in blob and "0.1%" in text_l:
                out.append(chunk)
            elif "accepted contract" in blob and chunk_states_delay_damages_rate(
                chunk.text or "",
            ):
                out.append(chunk)
        return out[:k]

    def fake_chunks_for_docs(
        self, project_id, doc_ids, k_per_doc=12, from_end=False, all_rows=False,
    ):
        by_doc: dict[str, list] = {}
        for chunk in all_chunks:
            if chunk.doc_id in (doc_ids or []):
                by_doc.setdefault(chunk.doc_id, []).append(chunk)
        out = []
        for did in doc_ids or []:
            rows = sorted(by_doc.get(did, []), key=lambda c: c.chunk_index)
            if all_rows:
                out.extend(rows)
                continue
            n = max(1, int(k_per_doc or 12))
            out.extend(rows[-n:] if from_end else rows[:n])
        return out

    def fake_containing_all(self, project_id, needles, k=20, doc_ids=None):
        needles_l = [str(n).lower() for n in (needles or [])]
        if "1.1.1" in needles_l or "excluding" in needles_l:
            return []
        out = []
        for chunk in all_chunks:
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            text_l = (chunk.text or "").lower()
            if needles_l and all(n in text_l for n in needles_l):
                out.append(chunk)
        return out[: max(1, int(k or 20))]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
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
        lambda pid, phrase, limit=8: (
            list(seeded) if "contract data" in (phrase or "").lower() else []
        ),
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)

    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=3)
    blob = " ".join(c.text for c in chunks)
    assert RATE in blob
    assert NET_ACA_TXT in blob
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["contract_amount"] == NET_ACA
    assert out["rate_percent"] == 0.1
    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(_CG_REFUSAL, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()
    assert "10,000.00" not in posted.split("\n", 1)[0]
    assert "0.015%" not in posted
    assert "263,175.67" not in posted


# Live leftover E1 after #535 (4242b86): sources were CoC chunks 9–11
# that already named 0.015% of the filled excl-VAT ACA. The all-chunk
# scan early-exited (fused "had" a real ACA) and compose emitted
# SAR 263,175.67/day. Contract Data 0.1% sat later in the same volume.
COC_015_WITH_ACA = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.7 Delay Damages. "
    "Delay damages for the whole of the Works are 0.015% of the "
    f"Accepted Contract Amount {NET_ACA_TXT} per calendar day."
)
CD_POINT_ONE_RATE = (
    "CONTRACT DATA\n8.8\nDelay Damages for the whole of the Works\n"
    f"{RATE}\n"
)
LOOKALIKE_DAILY = 263_175.67
CD_RATE_INDEX = 480


def _015_early_exit_all_chunks():
    """0.015%+ACA at 9–11; CD 0.1% at 480; standalone excl-VAT at 500."""
    lookalikes = [
        _chunk(
            f"gc{i}", GC_DOC, 0.95 - i * 0.01, COC_015_WITH_ACA, chunk_index=9 + i,
        )
        for i in range(3)
    ]
    dummies = [
        _chunk(f"pre{i}", GC_DOC, 0.10, GC_8_8, chunk_index=i)
        for i in range(MIDDLE_DOC_LEN)
        if i not in (9, 10, 11, CD_RATE_INDEX, MIDDLE_ACA_INDEX)
    ]
    rate = _chunk(
        "cdrate", GC_DOC, 0.20, CD_POINT_ONE_RATE, chunk_index=CD_RATE_INDEX,
    )
    aca = _chunk(
        "aca500", GC_DOC, 0.21, LIVE_SCANNED_EXCL_ACA, chunk_index=MIDDLE_ACA_INDEX,
    )
    return list(dummies) + list(lookalikes) + [rate, aca], lookalikes, rate, aca


def test_coc_015_window_is_not_a_standalone_e1_money_row():
    from app.core.rag.retriever import (
        _has_standalone_excl_vat_aca,
        _daily_rate_preference,
        chunk_states_delay_damages_rate,
    )
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
        parse_delay_damages_rate_percent,
    )

    assert chunk_states_delay_damages_rate(COC_015_WITH_ACA)
    assert chunk_has_real_accepted_contract_amount(COC_015_WITH_ACA)
    assert not _has_standalone_excl_vat_aca(COC_015_WITH_ACA)
    assert _has_standalone_excl_vat_aca(LIVE_SCANNED_EXCL_ACA)
    assert parse_delay_damages_rate_percent(COC_015_WITH_ACA) is None
    assert parse_delay_damages_rate_percent(CD_POINT_ONE_RATE) == 0.1
    assert _daily_rate_preference(COC_015_WITH_ACA) < 2
    assert _daily_rate_preference(CD_POINT_ONE_RATE) >= 2


def test_e1_535_scan_does_not_early_exit_on_015_rate_window_aca():
    """#535 regression: all-chunk scan still runs when 9–11 cite excl-VAT."""
    from app.core.rag.retriever import recall_composition_operands
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    all_chunks, lookalikes, _rate, _aca = _015_early_exit_all_chunks()

    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            if all_rows:
                return rows
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            return []

    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in lookalikes}
    assert all(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _Store(),
    )
    assert recovered >= 1
    excerpts = "\n\n".join(c.text or "" for c, _s, _b in fused.values())
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_retrieve_015_top_k_still_composes_point_one(monkeypatch):
    """Live 4242b86: sources only chunks 9–11 (0.015%+ACA). Must be 0.1%."""
    from app.core.rag import retriever as ret

    all_chunks, lookalikes, _rate, _aca = _015_early_exit_all_chunks()
    names = {GC_DOC: CD_SCANNED_NAME}
    seeded = [
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in lookalikes if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return list(lookalikes)[:k]

    def fake_chunks_for_docs(
        self, project_id, doc_ids, k_per_doc=12, from_end=False, all_rows=False,
    ):
        by_doc: dict[str, list] = {}
        for chunk in all_chunks:
            if chunk.doc_id in (doc_ids or []):
                by_doc.setdefault(chunk.doc_id, []).append(chunk)
        out = []
        for did in doc_ids or []:
            rows = sorted(by_doc.get(did, []), key=lambda c: c.chunk_index)
            if all_rows:
                out.extend(rows)
                continue
            n = max(1, int(k_per_doc or 12))
            out.extend(rows[-n:] if from_end else rows[:n])
        return out

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        lambda *a, **k: [],
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
        lambda pid, phrase, limit=8: (
            list(seeded) if "contract data" in (phrase or "").lower() else []
        ),
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)

    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=3)
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY
    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(_CG_REFUSAL, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()
    assert "263,175.67" not in posted
    assert "0.015%" not in posted.split("\n", 1)[0]


def test_e1_loaded_cd_rows_kill_refuse_even_when_top_k_is_toys():
    """(a) excl-VAT ACA in loaded Contract Data rows must not refuse."""
    from app.core.rag.retriever import recall_composition_operands
    from app.lib.construction_formulas_commercial import (
        chunk_has_real_accepted_contract_amount,
    )

    all_chunks, toys, _aca = _middle_aca_all_chunks()

    class _LoadedRows:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            if all_rows:
                return rows
            return []

    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in toys}
    assert not any(
        chunk_has_real_accepted_contract_amount(c.text or "")
        for c, _s, _b in fused.values()
    )
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _LoadedRows(),
    )
    assert recovered >= 1
    excerpts = "\n\n".join(c.text or "" for c, _s, _b in fused.values())
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    rag = _sys(*(c.text or "" for c, _s, _b in fused.values()))
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(_CG_REFUSAL, rag, msgs)
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()


# Live leftover E1 after #536: top-k is pointer-only Contract Data
# chunks 9–11 (no parseable 0.1% and no excl-VAT ACA). The CoC 0.015%
# path did not fire. Loaded CD volume still has both operands.
REFUSE_PRONE_8_8 = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.8 Delay Damages. "
    "The Contractor shall pay delay damages for the whole of the Works "
    "at the rate stated in the Contract Data for every calendar day."
)


def _refuse_prone_volume_chunks():
    windows = [
        _chunk(
            f"gc{i}", GC_DOC, 0.95 - i * 0.01, REFUSE_PRONE_8_8, chunk_index=9 + i,
        )
        for i in range(3)
    ]
    dummies = [
        _chunk(f"pre{i}", GC_DOC, 0.10, GC_8_8, chunk_index=i)
        for i in range(MIDDLE_DOC_LEN)
        if i not in (9, 10, 11, CD_RATE_INDEX, MIDDLE_ACA_INDEX)
    ]
    rate = _chunk(
        "cdrate", GC_DOC, 0.20, CD_POINT_ONE_RATE, chunk_index=CD_RATE_INDEX,
    )
    aca = _chunk(
        "aca500", GC_DOC, 0.21, LIVE_SCANNED_EXCL_ACA, chunk_index=MIDDLE_ACA_INDEX,
    )
    return list(dummies) + list(windows) + [rate, aca], windows, rate, aca


def _refuse_store(all_chunks):
    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            if all_rows:
                return rows
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            return []

    return _Store()


def test_e1_loaded_cd_volume_helper_composes_when_top_k_is_refuse_prone(
    monkeypatch,
):
    """Even if excerpts are chunks 9–11, the loaded volume supplies both."""
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda pid, phrase, limit=8: [
            {"id": GC_DOC, "original_name": CD_SCANNED_NAME},
        ] if "contract" in (phrase or "").lower() else [],
    )
    rag = _sys(*(c.text for c in windows))
    extra = daily_damages_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _refuse_store(all_chunks),
        rag_context=rag["content"],
        doc_ids=[GC_DOC],
    )
    assert RATE in extra
    assert NET_ACA_TXT in extra
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, extra)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_loaded_cd_volume_helper_does_not_steal_neighbor_asks(monkeypatch):
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    all_chunks, _windows, _rate, _aca = _refuse_prone_volume_chunks()
    store = _refuse_store(all_chunks)
    for ask in (A2_ASK, A3_ASK, A5_ASK, A6_ASK, A9_ASK, C1_ASK, F1_ASK):
        assert daily_damages_excerpts_from_loaded_cd_volume(
            ask, ACTIVE, store, doc_ids=[GC_DOC],
        ) == ""


def test_e1_loaded_cd_volume_helper_keeps_015_reject(monkeypatch):
    """Lookalike-only volume must not become 263,175.67."""
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    lookalikes = [
        _chunk(
            f"gc{i}", GC_DOC, 0.95, COC_015_WITH_ACA, chunk_index=9 + i,
        )
        for i in range(3)
    ]
    aca = _chunk(
        "aca500", GC_DOC, 0.21, LIVE_SCANNED_EXCL_ACA, chunk_index=MIDDLE_ACA_INDEX,
    )
    all_chunks = list(lookalikes) + [aca]
    extra = daily_damages_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _refuse_store(all_chunks),
        doc_ids=[GC_DOC],
    )
    assert extra == ""
    assert compose_delay_damages_daily_from_excerpts(
        LIVE_E1, "\n\n".join(c.text for c in all_chunks),
    ) is None


def test_ensure_e1_kept_can_compose_replaces_refuse_prone_windows():
    from app.core.rag.retriever import ensure_kept_can_compose_daily_damages

    all_chunks, windows, rate, aca = _refuse_prone_volume_chunks()
    kept = list(windows)
    ranked = list(windows) + [rate, aca]
    assert ensure_kept_can_compose_daily_damages(LIVE_E1, kept, ranked) is True
    excerpts = "\n\n".join(c.text or "" for c in kept)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_retrieve_refuse_prone_9_11_still_composes_point_one(monkeypatch):
    """Live fdd60275: sources only chunks 9–11 (pointer). Must still compose."""
    from app.core.rag import retriever as ret

    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    names = {GC_DOC: CD_SCANNED_NAME}
    seeded = [
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in windows if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return list(windows)[:k]

    def fake_chunks_for_docs(
        self, project_id, doc_ids, k_per_doc=12, from_end=False, all_rows=False,
    ):
        by_doc: dict[str, list] = {}
        for chunk in all_chunks:
            if chunk.doc_id in (doc_ids or []):
                by_doc.setdefault(chunk.doc_id, []).append(chunk)
        out = []
        for did in doc_ids or []:
            rows = sorted(by_doc.get(did, []), key=lambda c: c.chunk_index)
            if all_rows:
                out.extend(rows)
                continue
            n = max(1, int(k_per_doc or 12))
            out.extend(rows[-n:] if from_end else rows[:n])
        return out

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        lambda *a, **k: [],
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
        lambda pid, phrase, limit=8: (
            list(seeded) if "contract data" in (phrase or "").lower() else []
        ),
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)

    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=3)
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY
    rag = _sys(*(c.text or "" for c in chunks))
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(_CG_REFUSAL, rag, msgs, project_id=ACTIVE)
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()
    assert "263,175.67" not in posted
    assert "0.015%" not in posted.split("\n", 1)[0]


def _ignore_all_rows_store(all_chunks, owner_pid=None):
    """Live #537 miss: accepts all_rows but still returns first-N / last-N."""

    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False,
        ):
            if owner_pid is not None and project_id != owner_pid:
                return []
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            return []

    return _Store()


def _live_refuse_sys(*chunks):
    """Inject markers the live Sources panel uses (class=project_corpus)."""
    src = "AB-2023-101_QP2 Infra P1_Vol 1.0_Con..."
    body = "\n\n".join(
        f"[doc_id={c.doc_id} chunk={c.chunk_index} "
        f"score={c.score:.3f} class=project_corpus src={src}] {c.text}"
        for c in chunks
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def test_e1_pointer_window_qualifies_doc_without_filename():
    """Truncated Sources name must not hide the bound 8.8 volume."""
    from app.core.rag.retriever import _doc_qualifies_for_late_aca_scan

    assert _doc_qualifies_for_late_aca_scan(REFUSE_PRONE_8_8, "")
    assert _doc_qualifies_for_late_aca_scan(REFUSE_PRONE_8_8, "Vol 1.0_Con...")
    assert not _doc_qualifies_for_late_aca_scan(SPEC_TOC, "")


def test_e1_fetch_does_not_early_exit_when_all_rows_is_ignored():
    """First-N 0.015%+ACA is not enough — walk until both operands exist."""
    from app.core.rag.retriever import _scan_documents_for_operands

    all_chunks, _lookalikes, rate, aca = _015_early_exit_all_chunks()
    extra = _scan_documents_for_operands(
        _ignore_all_rows_store(all_chunks), ACTIVE, [GC_DOC],
    )
    texts = [c.text or "" for c in extra]
    assert any(c.chunk_id == rate.chunk_id for c in extra)
    assert any(c.chunk_id == aca.chunk_id for c in extra)
    assert CD_POINT_ONE_RATE in texts
    assert LIVE_SCANNED_EXCL_ACA in texts


def test_e1_loaded_volume_composes_when_all_rows_ignored_and_operands_mid_doc(
    monkeypatch,
):
    """Live 9ad62cc refuse window: 9–11 + BOQ refuse; operands mid-volume."""
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    rag = _live_refuse_sys(*windows)
    extra = daily_damages_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _ignore_all_rows_store(all_chunks),
        rag_context=rag["content"],
        doc_ids=[GC_DOC],
    )
    assert RATE in extra
    assert NET_ACA_TXT in extra
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, extra)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_loaded_volume_composes_from_015_windows_when_all_rows_ignored(
    monkeypatch,
):
    """#536 lookalike 9–11 still compose 0.1% when the volume has it."""
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    all_chunks, windows, _rate, _aca = _015_early_exit_all_chunks()
    extra = daily_damages_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _ignore_all_rows_store(all_chunks),
        rag_context=_live_refuse_sys(*windows)["content"],
        doc_ids=[GC_DOC],
    )
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, extra)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_loaded_volume_uses_cited_chunk_project_id(monkeypatch):
    """UI Master Corpus id is empty at the store; cited chunk owner has the rows."""
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    source = "p_source"
    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    for chunk in all_chunks:
        chunk.project_id = source
    extra = daily_damages_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _ignore_all_rows_store(all_chunks, owner_pid=source),
        rag_context=_live_refuse_sys(*windows)["content"],
        doc_ids=[GC_DOC],
        extra_pids=[source],
    )
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, extra)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1


def test_e1_retrieve_ignore_all_rows_refuse_window_still_composes(monkeypatch):
    """Live 9ad62cc: Cosine 9–11 only; store ignores all_rows. Must still compose."""
    from app.core.rag import retriever as ret

    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    names = {GC_DOC: "AB-2023-101_QP2 Infra P1_Vol 1.0_Con..."}
    seeded = [
        {"id": GC_DOC, "original_name": CD_SCANNED_NAME, "file_path": CD_SCANNED_NAME},
    ]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in windows if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return list(windows)[:k]

    store = _ignore_all_rows_store(all_chunks)

    def fake_chunks_for_docs(
        self, project_id, doc_ids, k_per_doc=12, from_end=False, all_rows=False,
    ):
        return store.chunks_for_docs(
            project_id, doc_ids, k_per_doc=k_per_doc,
            from_end=from_end, all_rows=all_rows,
        )

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        lambda *a, **k: [],
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
        lambda pid, phrase, limit=8: (
            list(seeded) if "contract data" in (phrase or "").lower() else []
        ),
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)

    chunks, _ = ret.retrieve_with_filter(LIVE_E1, ACTIVE, k=3)
    excerpts = "\n\n".join(c.text or "" for c in chunks)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY
    rag = _live_refuse_sys(*chunks)
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(
        _CG_REFUSAL, rag, msgs,
        audit_rec={"project_id": ACTIVE, "chunks": [
            {"project_id": ACTIVE, "doc_id": GC_DOC, "chunk_index": 9},
        ]},
    )
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()
    assert "263,175.67" not in posted


LIVE_TRUNC_NAME = "AB-2031-007_Package 1_Vol 1.0_Con..."
OCR_NOISE_9_11 = (
    "this contract class=project_corpus scanned page header "
    "drawing title block no extractable rate"
)


def test_a_truncated_volume_name_is_still_scanned_through_the_pool():
    """A truncated Sources name ("<id>_Vol N_Con...") says nothing about the
    document's kind, so it no longer qualifies the volume by NAME (that was
    one corpus's naming convention). The volume is still scanned: when no
    pooled document qualifies, the pooled documents themselves are read."""
    from app.core.rag.retriever import (
        _doc_qualifies_for_late_aca_scan,
        _pool_doc_ids_for_late_aca,
        filename_looks_like_conditions_volume,
    )

    assert not filename_looks_like_conditions_volume(LIVE_TRUNC_NAME)
    assert filename_looks_like_conditions_volume(
        "AB-2031-007_Package 1_Vol 1.0_Cond of Contract"
    )
    assert not filename_looks_like_conditions_volume("Media - Progr.pdf")
    assert not _doc_qualifies_for_late_aca_scan(OCR_NOISE_9_11, "Media - Progr.pdf")
    noise = _chunk("noise", GC_DOC, 0.9, OCR_NOISE_9_11, chunk_index=9)
    noise.source_name = LIVE_TRUNC_NAME
    assert _pool_doc_ids_for_late_aca({"noise": (noise, 0.9, 0.0)}) == [GC_DOC]


def test_e1_pool_falls_back_to_fused_doc_ids_when_ocr_fails_qualify():
    """Cosine 9–11 OCR + missing name still scans those fused docs."""
    from app.core.rag.retriever import (
        _doc_qualifies_for_late_aca_scan,
        _pool_doc_ids_for_late_aca,
        recall_composition_operands,
    )

    windows = [
        _chunk(
            f"ocr{i}", GC_DOC, 0.95 - i * 0.01, OCR_NOISE_9_11,
            chunk_index=9 + i, source_name="",
        )
        for i in range(3)
    ]
    assert all(
        not _doc_qualifies_for_late_aca_scan(c.text, "") for c in windows
    )
    fused = {c.chunk_id: (c, c.score or 0.0, 0.0) for c in windows}
    assert _pool_doc_ids_for_late_aca(fused) == [GC_DOC]

    all_chunks, _w, _rate, _aca = _refuse_prone_volume_chunks()
    recovered = recall_composition_operands(
        LIVE_E1, ACTIVE, fused, _ignore_all_rows_store(all_chunks),
    )
    assert recovered >= 1
    excerpts = "\n\n".join(c.text or "" for c, _s, _b in fused.values())
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY
    assert any(s >= 2.4 for _c, s, _b in fused.values())


def test_e1_token_cap_keeps_zero_score_operands_on_live_ask(monkeypatch):
    """Live 396cc7b: 3 HIGH 9–11 fill the cap; 0.0 operands drop without query."""
    from app.core.rag.inject import apply_token_cap

    # 4000 chars ≈ 1000 tokens. Two HIGH pointers fill a 2000-token cap
    # with no leftover for 0.0 operands — the live 396cc7b drop.
    pad = "x" * (4000 - len(REFUSE_PRONE_8_8))
    pointers = [
        _chunk(
            f"gc{i}", GC_DOC, 0.95 - i * 0.01,
            REFUSE_PRONE_8_8 + pad, chunk_index=9 + i,
            source_name=LIVE_TRUNC_NAME,
        )
        for i in range(3)
    ]
    rate = _chunk("cdrate", GC_DOC, 0.0, CD_POINT_ONE_RATE, chunk_index=480)
    aca = _chunk("aca500", GC_DOC, 0.0, LIVE_SCANNED_EXCL_ACA, chunk_index=500)
    chunks = list(pointers) + [rate, aca]
    monkeypatch.setenv("MAX_RAG_TOKENS", "2000")

    dropped, _ = apply_token_cap(chunks)
    dropped_ids = {c.chunk_id for c in dropped}
    assert "cdrate" not in dropped_ids
    assert "aca500" not in dropped_ids
    assert compose_delay_damages_daily_from_excerpts(
        LIVE_E1, "\n\n".join(c.text or "" for c in dropped),
    ) is None

    kept, _ = apply_token_cap(chunks, query=LIVE_E1)
    kept_ids = {c.chunk_id for c in kept}
    assert "cdrate" in kept_ids
    assert "aca500" in kept_ids
    out = compose_delay_damages_daily_from_excerpts(
        LIVE_E1, "\n\n".join(c.text or "" for c in kept),
    )
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY
    blob = "\n".join(c.text or "" for c in kept)
    assert "263,175.67" not in blob
    assert "0.015%" not in blob


def test_e1_loaded_volume_resolves_owner_pid_when_ui_pid_is_alias(
    monkeypatch,
):
    """master_corpus / empty UI pid; store only serves drive_archive owner."""
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    owner = "drive_archive"
    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    for chunk in all_chunks:
        chunk.project_id = owner
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"id": did, "project_id": owner} if did == GC_DOC else {},
    )
    title_calls: list[tuple] = []

    def _title(pid, phrase, limit=8):
        title_calls.append((pid, phrase))
        return [{"id": f"other-{len(title_calls)}", "original_name": phrase}] * 16

    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase", _title,
    )
    rag = _live_refuse_sys(*windows)
    store = _ignore_all_rows_store(all_chunks, owner_pid=owner)
    for ui_pid in ("master_corpus", ""):
        extra = daily_damages_excerpts_from_loaded_cd_volume(
            LIVE_E1, ui_pid, store, rag_context=rag["content"],
        )
        out = compose_delay_damages_daily_from_excerpts(LIVE_E1, extra)
        assert out is not None, f"ui_pid={ui_pid!r} must still compose"
        assert out["daily_amount"] == DAILY
        assert out["rate_percent"] == 0.1
        assert out["contract_amount"] == NET_ACA
        assert out["daily_amount"] != LOOKALIKE_DAILY
    assert title_calls == [], (
        "cited doc_id already has both operands — do not title-list 16 CoC files"
    )


def test_e1_loaded_volume_does_not_need_title_listing_when_cited_composes(
    monkeypatch,
):
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()

    def _boom(pid, phrase, limit=8):
        raise AssertionError(
            f"title listing must not run when cited volume composes ({phrase})"
        )

    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase", _boom,
    )
    extra = daily_damages_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _ignore_all_rows_store(all_chunks),
        rag_context=_live_refuse_sys(*windows)["content"],
        doc_ids=[GC_DOC],
    )
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, extra)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_scan_project_ids_include_master_corpus_source_alias():
    from app.core.rag.retriever import _late_scan_project_ids
    from app.core.projects import MASTER_CORPUS_SOURCE_PROJECT_ID

    pids = _late_scan_project_ids("master_corpus")
    assert "master_corpus" in pids
    assert MASTER_CORPUS_SOURCE_PROJECT_ID in pids


def test_e1_graft_last_chance_composes_from_cited_rag_when_ui_pid_empty(
    monkeypatch,
):
    """Cold New-chat: rag markers have doc_id, audit pid may be empty."""
    owner = "drive_archive"
    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    for chunk in all_chunks:
        chunk.project_id = owner
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"id": did, "project_id": owner} if did == GC_DOC else {},
    )
    monkeypatch.setattr(
        "app.core.rag.retriever.get_lexical_store",
        lambda: _ignore_all_rows_store(all_chunks, owner_pid=owner),
    )
    rag = _live_refuse_sys(*windows)
    msgs = [{"role": "user", "content": LIVE_E1}]
    posted = _postprocess_answer(
        _CG_REFUSAL, rag, msgs, project_id="",
        audit_rec={"project_id": "", "chunks": [
            {"project_id": owner, "doc_id": GC_DOC, "chunk_index": 9},
        ]},
    )
    assert "1,754,504.46" in posted
    assert posted != _CG_REFUSAL
    assert "upload your priced BOQ" not in posted.lower()
    assert "263,175.67" not in posted
    assert "0.015%" not in posted.split("\n", 1)[0]


# Live leftover E1 after #541 (tip 2ceef76 / cold ×3 on d1adadc):
# the store ignored all_rows AND capped k_per_doc at 400 (k=1_000_000
# still first-400). Prefix 400 + last 400 miss chunk 500 of a 1200-row
# volume. Needles empty. Without a mid-window offset walk, last-chance
# compose stayed empty → cost-grounding refuse or CoC 0.015%.
_E1_STORE_K_CAP = 400


def _capped_k_store(all_chunks, owner_pid=None, k_cap=_E1_STORE_K_CAP):
    """Ignore all_rows; cap k_per_doc; honor offset. Needles empty."""

    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12, from_end=False,
            all_rows=False, offset=0,
        ):
            if owner_pid is not None and project_id != owner_pid:
                return []
            rows = sorted(
                [c for c in all_chunks if c.doc_id in (doc_ids or [])],
                key=lambda c: c.chunk_index,
            )
            n = min(max(1, int(k_per_doc or 12)), k_cap)
            off = max(0, int(offset or 0))
            if from_end:
                end = len(rows) - off
                if end <= 0:
                    return []
                start = max(0, end - n)
                return rows[start:end]
            return rows[off:off + n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            return []

    return _Store()


def test_e1_cap_noise_is_pointer_or_015_not_operands():
    from app.core.rag.retriever import _is_daily_damages_cap_noise

    assert _is_daily_damages_cap_noise(REFUSE_PRONE_8_8)
    assert _is_daily_damages_cap_noise(COC_015_WITH_ACA)
    assert not _is_daily_damages_cap_noise(CD_POINT_ONE_RATE)
    assert not _is_daily_damages_cap_noise(LIVE_SCANNED_EXCL_ACA)
    assert not _is_daily_damages_cap_noise(SCANNED_RATE)


def test_e1_capped_k_prefix_tail_miss_middle_operands():
    """#541 hole: k-cap 400 + ignore all_rows misses chunk 500 of 1200."""
    from app.core.rag.retriever import _DOC_SCAN_WINDOW

    all_chunks, _windows, rate, aca = _refuse_prone_volume_chunks()
    store = _capped_k_store(all_chunks)
    prefix = store.chunks_for_docs(
        ACTIVE, [GC_DOC], k_per_doc=1_000_000, from_end=False,
    )
    tail = store.chunks_for_docs(
        ACTIVE, [GC_DOC], k_per_doc=_DOC_SCAN_WINDOW, from_end=True,
    )
    assert all(c.chunk_index != MIDDLE_ACA_INDEX for c in prefix)
    assert all(c.chunk_index != CD_RATE_INDEX for c in prefix)
    assert all(c.chunk_index != MIDDLE_ACA_INDEX for c in tail)
    assert all(c.chunk_index != CD_RATE_INDEX for c in tail)
    mid = store.chunks_for_docs(
        ACTIVE, [GC_DOC], k_per_doc=_DOC_SCAN_WINDOW, offset=400,
    )
    assert any(c.chunk_id == rate.chunk_id for c in mid)
    assert any(c.chunk_id == aca.chunk_id for c in mid)


def test_e1_fetch_mid_window_recovers_when_k_is_capped():
    """Refuse-prone 9–11 + k-cap 400 must still surface 0.1% and excl-VAT."""
    from app.core.rag.retriever import _scan_documents_for_operands

    all_chunks, _windows, rate, aca = _refuse_prone_volume_chunks()
    extra = _scan_documents_for_operands(
        _capped_k_store(all_chunks), ACTIVE, [GC_DOC],
    )
    texts = [c.text or "" for c in extra]
    assert any(c.chunk_id == rate.chunk_id for c in extra)
    assert any(c.chunk_id == aca.chunk_id for c in extra)
    assert CD_POINT_ONE_RATE in texts
    assert LIVE_SCANNED_EXCL_ACA in texts
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, "\n\n".join(texts))
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["contract_amount"] == NET_ACA
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_fetch_mid_window_recovers_015_top_k_when_k_is_capped():
    """CoC 0.015% 9–11 + k-cap 400 must still compose 0.1%, not 263175.67."""
    from app.core.rag.retriever import _scan_documents_for_operands

    all_chunks, _windows, rate, aca = _015_early_exit_all_chunks()
    extra = _scan_documents_for_operands(
        _capped_k_store(all_chunks), ACTIVE, [GC_DOC],
    )
    excerpts = "\n\n".join(c.text or "" for c in extra)
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, excerpts)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["daily_amount"] != LOOKALIKE_DAILY
    assert any(c.chunk_id == rate.chunk_id for c in extra)
    assert any(c.chunk_id == aca.chunk_id for c in extra)


def test_e1_loaded_volume_composes_when_k_is_capped(monkeypatch):
    from app.core.rag.retriever import daily_damages_excerpts_from_loaded_cd_volume

    all_chunks, windows, _rate, _aca = _refuse_prone_volume_chunks()
    extra = daily_damages_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _capped_k_store(all_chunks),
        rag_context=_live_refuse_sys(*windows)["content"],
        doc_ids=[GC_DOC],
    )
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, extra)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_token_cap_force_keeps_huge_operand_over_leftover_cap(monkeypatch):
    """A 0.1% scanned page larger than leftover cap must still be kept."""
    from app.core.rag.inject import apply_token_cap

    huge_rate = _chunk(
        "cdrate", GC_DOC, 0.0,
        CD_POINT_ONE_RATE + ("x" * 9000),
        chunk_index=480,
    )
    aca = _chunk("aca500", GC_DOC, 0.0, LIVE_SCANNED_EXCL_ACA, chunk_index=500)
    pointers = [
        _chunk(
            f"gc{i}", GC_DOC, 0.95 - i * 0.01,
            REFUSE_PRONE_8_8 + ("y" * 4000),
            chunk_index=9 + i,
        )
        for i in range(3)
    ]
    monkeypatch.setenv("MAX_RAG_TOKENS", "2000")
    kept, _ = apply_token_cap(list(pointers) + [huge_rate, aca], query=LIVE_E1)
    kept_ids = {c.chunk_id for c in kept}
    assert "cdrate" in kept_ids
    assert "aca500" in kept_ids
    out = compose_delay_damages_daily_from_excerpts(
        LIVE_E1, "\n\n".join(c.text or "" for c in kept),
    )
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_token_cap_evicts_015_windows_when_operands_are_protected(monkeypatch):
    """HIGH 0.015% 9–11 must not stay once 0.1% + excl-VAT are protected."""
    from app.core.rag.inject import apply_token_cap

    lookalikes = [
        _chunk(
            f"gc{i}", GC_DOC, 0.95 - i * 0.01, COC_015_WITH_ACA, chunk_index=9 + i,
        )
        for i in range(3)
    ]
    rate = _chunk("cdrate", GC_DOC, 0.0, CD_POINT_ONE_RATE, chunk_index=480)
    aca = _chunk("aca500", GC_DOC, 0.0, LIVE_SCANNED_EXCL_ACA, chunk_index=500)
    monkeypatch.setenv("MAX_RAG_TOKENS", "6000")
    kept, _ = apply_token_cap(list(lookalikes) + [rate, aca], query=LIVE_E1)
    blob = "\n".join(c.text or "" for c in kept)
    assert CD_POINT_ONE_RATE in blob
    assert LIVE_SCANNED_EXCL_ACA in blob
    assert "0.015%" not in blob
    assert "263,175.67" not in blob
    out = compose_delay_damages_daily_from_excerpts(LIVE_E1, blob)
    assert out is not None
    assert out["daily_amount"] == DAILY
    assert out["rate_percent"] == 0.1
    assert out["daily_amount"] != LOOKALIKE_DAILY


def test_e1_token_cap_protect_survives_one_bad_chunk(monkeypatch):
    """One preference exception must not wipe the other operand's protect."""
    from app.core.rag import inject as inj
    from app.core.rag import retriever as ret

    rate = _chunk("cdrate", GC_DOC, 0.0, CD_POINT_ONE_RATE, chunk_index=480)
    aca = _chunk("aca500", GC_DOC, 0.0, LIVE_SCANNED_EXCL_ACA, chunk_index=500)
    junk = _chunk("junk", GC_DOC, 0.99, "not a rate window", chunk_index=1)
    orig = ret._daily_rate_preference

    def _boom(text):
        if "not a rate window" in (text or ""):
            raise RuntimeError("preference boom")
        return orig(text)

    monkeypatch.setattr(ret, "_daily_rate_preference", _boom)
    monkeypatch.setenv("MAX_RAG_TOKENS", "2000")
    pad = "x" * 4000
    pointers = [
        _chunk(
            f"gc{i}", GC_DOC, 0.95,
            REFUSE_PRONE_8_8 + pad, chunk_index=9 + i,
        )
        for i in range(3)
    ]
    kept, _ = inj.apply_token_cap(
        [junk] + list(pointers) + [rate, aca], query=LIVE_E1,
    )
    kept_ids = {c.chunk_id for c in kept}
    assert "cdrate" in kept_ids
    assert "aca500" in kept_ids
