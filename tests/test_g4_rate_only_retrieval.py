"""G4: D529.3 removal of storm water culverts is Rate Only — no amount.

Live OLD-pack G4 on Master Corpus (tip 4d8ddb79 / was a65cebb5):

    Answer only from the client project documents. What is the total
    amount for removal of storm water culverts (D529.3)?

Expected: Rate Only — no amount exists. Observed: generic
acknowledgement ("I'm ready to help. Please let me know what specific
information…") plus the 3348/3348 coverage footer. No Rate Only /
D529.3 election. No invented total. No tool-JSON leak.

Cosine prefers priced lookalikes (D549.2 fence, D599.5 carriageway,
an Excluded culvert that shares "storm water"). Elect the Rate Only
row as written. Do not invent a money total.

Not #504 (E1 compose), not #505 (F2 duration), not #506 (G1 Schedule
10). Fixture wording only.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.runtime import (
    _CG_REFUSAL,
    _graft_rate_only_item,
    _postprocess_answer,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk


CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
G4_ASK = CATALOG["cases"]["G4"]["ask"]
LIVE_PREFIX = "Answer only from the client project documents. "
LIVE_G4 = LIVE_PREFIX + G4_ASK
A2_ASK = CATALOG["cases"]["A2"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]
G1_ASK = CATALOG["cases"]["G1"]["ask"]
B5_ASK = (
    "What is the amount for removal of existing chain link fence (D549.2)?"
)

# Fixture-only. Must not invent a live client total.
RATE_ONLY_MARK = "Rate Only"
D529 = "D529.3"
RATE_ONLY_ROW = (
    f"{D529} Removal of storm water culverts — m 1,370.00 {RATE_ONLY_MARK}"
)
OCR_RATE_ONLY = (
    "D 529.3 Removal of storm water culverts\n"
    "m 1370.00 Rate Only"
)
PRICED_FENCE = (
    "D549.2 Removal of existing chain link fence 80 m 15.00 1,200.00"
)
PRICED_CARRIAGEWAY = (
    "D599.5 Breaking out existing carriageway including road markings "
    "1,200 m2 18.00 21,600.00"
)
EXCLUDED_CULVERT = (
    "J |Breakout and remove existing storm water culverts D599.5 "
    "| sum 1 Excluded"
)
PRICED_CULVERT = (
    "D529.1 Removal of storm water culverts 50 m 1,370.00 68,500.00"
)
MIXED_PAGE = (
    f"| {D529} | Removal of storm water culverts | — | m | 1,370.00 "
    f"| {RATE_ONLY_MARK} |\n"
    "| D549.2 | Removal of existing chain link fence | 80 | m | 15.00 "
    "| 1,200.00 |\n"
)

BOQ_NAME = (
    "AB-2023-101_QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_"
    "Bill of Quantities (Priced).pdf"
)
FENCE_NAME = "Demolition BOQ page d-3-3 fence.pdf"
CWAY_NAME = "Demolition BOQ page d-3-3 carriageway.pdf"
EXCL_NAME = "Provisional sums Excluded culverts.pdf"

ACTIVE = "p_master"
RO_DOC = "ro529"
FENCE_DOC = "fence"
CWAY_DOC = "cway"
EXCL_DOC = "excl"
PRICED_CULV_DOC = "pc529"
MIX_DOC = "mix"

GENERIC_GREETING = (
    "I'm ready to help. Please let me know what specific information "
    "you need from the project documents."
)
INVENTED_TOTAL = (
    "The total amount for removal of storm water culverts (D529.3) "
    "is SAR 68,500.00."
)


def _chunk(cid, doc_id, score, text):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_g4_catalog_ask_is_frozen():
    assert G4_ASK == (
        "What is the total amount for removal of storm water culverts "
        "(D529.3)?"
    )


def test_g4_ask_shape_and_rate_only_gate():
    from app.core.rag.retriever import (
        chunk_states_rate_only_item,
        extract_asked_cesmm_codes,
        query_asks_for_boq_item_amount,
    )

    assert query_asks_for_boq_item_amount(G4_ASK)
    assert query_asks_for_boq_item_amount(LIVE_G4)
    assert query_asks_for_boq_item_amount(
        "removal of storm water culverts D529.3 — what is the total amount?"
    )
    assert query_asks_for_boq_item_amount(B5_ASK)
    assert not query_asks_for_boq_item_amount(A2_ASK)
    assert not query_asks_for_boq_item_amount(A5_ASK)
    assert not query_asks_for_boq_item_amount(G1_ASK)
    assert not query_asks_for_boq_item_amount(
        "What is the unit rate for D529.3?"
    )
    assert not query_asks_for_boq_item_amount(
        "Calculate the delay damages per calendar day in SAR for the "
        "whole of the Works."
    )
    assert not query_asks_for_boq_item_amount(
        "What is the value of the Parent Company Guarantee?"
    )

    assert extract_asked_cesmm_codes(G4_ASK) == ["d529.3"]
    assert extract_asked_cesmm_codes(LIVE_G4) == ["d529.3"]
    assert extract_asked_cesmm_codes("rate for D 529.3 please") == ["d529.3"]

    codes = ["d529.3"]
    assert chunk_states_rate_only_item(RATE_ONLY_ROW, codes)
    assert chunk_states_rate_only_item(OCR_RATE_ONLY, codes)
    assert chunk_states_rate_only_item(MIXED_PAGE, codes)
    assert not chunk_states_rate_only_item(PRICED_FENCE, codes)
    assert not chunk_states_rate_only_item(PRICED_CARRIAGEWAY, codes)
    assert not chunk_states_rate_only_item(EXCLUDED_CULVERT, codes)
    assert not chunk_states_rate_only_item(PRICED_CULVERT, codes)
    # Same page: D549.2 is priced. Do not stain it with D529.3 Rate Only.
    assert not chunk_states_rate_only_item(MIXED_PAGE, ["d549.2"])


def _install_g4_corpus(monkeypatch, *, rate_only_in_semantic: bool):
    from app.core.rag import retriever as ret

    fence = _chunk("fence", FENCE_DOC, 0.91, PRICED_FENCE)
    cway = _chunk("cway", CWAY_DOC, 0.88, PRICED_CARRIAGEWAY)
    excl = _chunk("excl", EXCL_DOC, 0.86, EXCLUDED_CULVERT)
    priced = _chunk("pc", PRICED_CULV_DOC, 0.84, PRICED_CULVERT)
    ro = _chunk("ro", RO_DOC, 0.22, RATE_ONLY_ROW)
    semantic = [fence, cway, excl, priced]
    if rate_only_in_semantic:
        semantic.append(ro)

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        # Empty on purpose: the live miss is cosine + term-rescue treating
        # "storm water" overlap as already-grounded. Identifier fusion
        # must not hide that — rescue is chunks_containing_all.
        return []

    def fake_containing_all(self, project_id, needles, k=20):
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        blob = RATE_ONLY_ROW.lower().replace(" ", "")
        if cleaned and all(
            n.replace(" ", "") in blob or n in RATE_ONLY_ROW.lower()
            for n in cleaned
        ):
            return [_chunk("ro", RO_DOC, 0.0, RATE_ONLY_ROW)]
        return []

    names = {
        RO_DOC: BOQ_NAME,
        FENCE_DOC: FENCE_NAME,
        CWAY_DOC: CWAY_NAME,
        EXCL_DOC: EXCL_NAME,
        PRICED_CULV_DOC: BOQ_NAME,
    }

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, project_id, doc_ids, k_per_doc=12: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 5,
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


def test_g4_retrieves_rate_only_when_priced_lookalikes_lead(monkeypatch):
    """The live failure: priced fence / carriageway / Excluded occupy top-k."""
    ret = _install_g4_corpus(monkeypatch, rate_only_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(G4_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks, "Rate Only row never reached the pool"
    assert RATE_ONLY_MARK in blob
    assert RATE_ONLY_MARK in chunks[0].text
    assert D529 in chunks[0].text or "D 529.3" in chunks[0].text
    assert "68,500" not in blob
    assert "1,200.00" not in blob
    assert "Excluded" not in blob
    assert all(
        ret.chunk_states_rate_only_item(c.text, ["d529.3"]) for c in chunks
    )


def test_g4_live_prefix_still_elects_rate_only(monkeypatch):
    ret = _install_g4_corpus(monkeypatch, rate_only_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(LIVE_G4, ACTIVE, k=5)
    assert chunks
    assert RATE_ONLY_MARK in chunks[0].text
    assert chunks[0].doc_id == RO_DOC


def test_g4_drops_lookalikes_when_rate_only_is_already_in_the_pool(monkeypatch):
    ret = _install_g4_corpus(monkeypatch, rate_only_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(G4_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert RATE_ONLY_MARK in chunks[0].text
    assert "68,500" not in blob
    assert all(c.doc_id == RO_DOC for c in chunks)


def test_a2_a5_g1_are_not_stolen_onto_the_rate_only_rescue(monkeypatch):
    ret = _install_g4_corpus(monkeypatch, rate_only_in_semantic=False)
    for ask in (A2_ASK, A5_ASK, G1_ASK):
        chunks, _ = ret.retrieve_with_filter(ask, ACTIVE, k=5)
        assert all(c.doc_id != RO_DOC for c in chunks), ask


def test_b5_priced_fence_is_not_treated_as_rate_only(monkeypatch):
    """WAVE 2 B5: D549.2 has an amount. Do not elect Rate Only for it."""
    ret = _install_g4_corpus(monkeypatch, rate_only_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(B5_ASK, ACTIVE, k=5)
    assert chunks
    assert all(c.doc_id != RO_DOC for c in chunks)
    assert any("1,200.00" in (c.text or "") for c in chunks)


def test_inject_states_rate_only_on_a_single_class_set():
    """Live miss: mixed-class SOURCE CLASS never fired; the model hedged."""
    msg = format_chunks_as_system_message(
        [_chunk("ro", RO_DOC, 0.9, RATE_ONLY_ROW)],
        4,
        query=LIVE_G4,
    )
    text = msg["content"]
    assert "RATE ONLY" in text
    assert "Rate Only" in text
    assert "generic acknowledgement" in text
    assert "SOURCE CLASS" not in text


def test_inject_does_not_invent_rate_only_from_priced_lookalikes():
    msg = format_chunks_as_system_message(
        [_chunk("fence", FENCE_DOC, 0.9, PRICED_FENCE)],
        4,
        query=LIVE_G4,
    )
    assert "RATE ONLY" not in msg["content"]


def test_inject_does_not_fire_on_a5_even_if_a_rate_only_chunk_is_present():
    msg = format_chunks_as_system_message(
        [_chunk("ro", RO_DOC, 0.9, RATE_ONLY_ROW)],
        4,
        query=A5_ASK,
    )
    assert "RATE ONLY" not in msg["content"]


def _sys(*chunk_texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=doc{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(chunk_texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def test_graft_states_rate_only_when_the_model_greeted():
    rag = _sys(RATE_ONLY_ROW)
    msgs = [{"role": "user", "content": LIVE_G4}]
    out = _graft_rate_only_item(GENERIC_GREETING, rag, msgs)
    assert "Rate Only" in out
    assert "D529.3" in out
    assert "no amount" in out.lower()
    assert "68,500" not in out
    assert "I'm ready to help" not in out


def test_graft_replaces_an_invented_total():
    rag = _sys(RATE_ONLY_ROW)
    msgs = [{"role": "user", "content": LIVE_G4}]
    out = _graft_rate_only_item(INVENTED_TOTAL, rag, msgs)
    assert "Rate Only" in out
    assert "68,500" not in out


def test_graft_is_a_no_op_when_rate_only_is_already_stated():
    rag = _sys(RATE_ONLY_ROW)
    msgs = [{"role": "user", "content": LIVE_G4}]
    already = "D529.3 is Rate Only. No amount exists in the client BOQ."
    assert _graft_rate_only_item(already, rag, msgs) == already


def test_graft_does_not_invent_when_the_excerpt_has_no_rate_only():
    rag = _sys(PRICED_CULVERT)
    msgs = [{"role": "user", "content": LIVE_G4}]
    assert _graft_rate_only_item(GENERIC_GREETING, rag, msgs) == GENERIC_GREETING


def test_postprocess_g4_greeting_states_rate_only():
    rag = _sys(RATE_ONLY_ROW)
    msgs = [{"role": "user", "content": LIVE_G4}]
    out = _postprocess_answer(GENERIC_GREETING, rag, msgs)
    assert "Rate Only" in out
    assert out != _CG_REFUSAL
    assert "upload your priced BOQ" not in out.lower()
    assert "68,500" not in out


def test_mutation_rate_only_predicate_is_what_lifts_the_row(monkeypatch):
    ret = _install_g4_corpus(monkeypatch, rate_only_in_semantic=False)
    monkeypatch.setattr(
        ret, "chunk_states_rate_only_item", lambda *_a, **_k: False,
    )
    # Labelled-row recall also recognises the item row as a filled row for
    # the item code the question names; that recogniser goes off too.
    monkeypatch.setattr(
        ret, "chunk_states_labelled_row", lambda *_a, **_k: False,
    )
    chunks, _ = ret.retrieve_with_filter(G4_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks
    assert chunks[0].doc_id != RO_DOC
    assert RATE_ONLY_MARK not in blob
