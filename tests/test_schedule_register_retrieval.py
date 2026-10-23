"""G1: Schedule 10 of the contract is Not Used — retrieve the register.

Live OLD-pack G1 on Master Corpus (tip a65cebb5): the ask retrieved
Volume 5 / Volume 4 / CPM and answered with a generic acknowledgement
that it would answer from the documents. It did not invent contents,
but it also never stated ``Schedule 10: Not Used``.

The register row is a short index line. Cosine prefers the long volumes
that mention "schedule" at length; term rescue treats that overlap as
already-grounded and skips the out-of-pool fetch.

Not #500 (D2/D4/D5 routing), not #501 (A3/A5 year lock), not #502
(C2 SPE-identity / A2 Contract Data filename). Fixture wording only —
do not invent Schedule 10 contents.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk


CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
G1_ASK = CATALOG["cases"]["G1"]["ask"]
LIVE_PREFIX = "Answer only from the client project documents. "
A2_ASK = CATALOG["cases"]["A2"]["ask"]
A3_ASK = CATALOG["cases"]["A3"]["ask"]
C2_ASK = CATALOG["cases"]["C2"]["ask"]
DEFINITION = "What does Schedule 10 mean in a programme?"

# Fixture-only. Must not invent Works Guarantees or any other contents.
REGISTER_ROW = "Schedule 10: Not Used"
SCHEDULE_9_ROW = "Schedule 9: Health & Safety KPIs"

REG_NAME = (
    "AB-2023-101_QP2 Infra P1_Vol 1.0_Cond of Contract "
    "(complete)_Contract Data.pdf"
)
# Live citations were Vol 5 / Vol 4 / CPM — those uploads do not carry
# a PREFIX-YEAR-SEQ, so the particulars arrival-order lock cannot drop
# them. A year-seq on the lookalikes would hide the defect this fence
# is for.
VOL4_NAME = "Vol 4 - Schedules.pdf"
VOL5_NAME = "Vol 5 - Specifications.pdf"
CPM_NAME = "CPM 16-01-2024.pdf"
SHOW_NAME = "2012 EXW (Show Package).docx"

# Live shape: scanned / unprefixed schedule index. No CONTRACT DATA
# particulars header, so the particulars boost and unnamed election
# never fire on this chunk alone.
REG_TEXT = (
    "Volume 4 schedules\n"
    f"{SCHEDULE_9_ROW}\n"
    f"{REGISTER_ROW}\n"
)
VOL4_TEXT = (
    "Volume 4 Schedules form part of the Contract. Schedule 10 "
    "shall be read with the Conditions of Contract. The schedules "
    "listed in this volume apply to the Works."
)
VOL5_TEXT = (
    "Volume 5 Specification. Programming and scheduling of the Works "
    "shall follow the approved construction schedule. Schedule reviews "
    "are held weekly with the Engineer."
)
CPM_TEXT = (
    "CPM permit tracker 16 January 2024. Critical-path schedule "
    "updates. No contract-volume schedule register."
)
SHOW_TEXT = (
    "2012 EXW Show Package. Schedule 10 — Works Guarantee. The "
    "Contractor shall provide a Works Guarantee in the form annexed. "
    "Schedule 10 sets out any applicable Works Guarantees."
)

ACTIVE = "p_master"
REG_DOC = "cd118sched"
VOL4_DOC = "vol4"
VOL5_DOC = "vol5"
CPM_DOC = "cpm"
SHOW_DOC = "show"


def _chunk(cid, doc_id, score, text):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_g1_catalog_ask_is_frozen():
    assert G1_ASK == "What does Schedule 10 of the contract contain?"


def test_g1_ask_shape_and_register_gate():
    from app.core.rag.retriever import (
        chunk_states_schedule_not_used,
        chunk_states_schedule_register,
        extract_asked_schedule_labels,
        query_asks_for_numbered_contract_schedule,
    )

    assert query_asks_for_numbered_contract_schedule(G1_ASK)
    assert query_asks_for_numbered_contract_schedule(LIVE_PREFIX + G1_ASK)
    assert query_asks_for_numbered_contract_schedule(
        "What does Schedule 9 of the contract volumes cover?"
    )
    assert not query_asks_for_numbered_contract_schedule(
        "What does Schedule 10 contain?"
    )
    assert not query_asks_for_numbered_contract_schedule(DEFINITION)
    assert not query_asks_for_numbered_contract_schedule(A2_ASK)
    assert not query_asks_for_numbered_contract_schedule(A3_ASK)
    assert not query_asks_for_numbered_contract_schedule(C2_ASK)

    assert extract_asked_schedule_labels(G1_ASK) == ["schedule 10"]
    assert extract_asked_schedule_labels(LIVE_PREFIX + G1_ASK) == ["schedule 10"]

    labels = ["schedule 10"]
    assert chunk_states_schedule_register(REG_TEXT, labels)
    assert chunk_states_schedule_not_used(REG_TEXT)
    assert not chunk_states_schedule_register(VOL4_TEXT, labels)
    assert not chunk_states_schedule_register(VOL5_TEXT, labels)
    assert not chunk_states_schedule_register(CPM_TEXT, labels)
    assert not chunk_states_schedule_register(SHOW_TEXT, labels)
    assert not chunk_states_schedule_not_used(SHOW_TEXT)
    assert not chunk_states_schedule_not_used(VOL4_TEXT)


def _install_g1_corpus(monkeypatch, *, register_in_semantic: bool):
    from app.core.rag import retriever as ret

    vol4 = _chunk("vol4", VOL4_DOC, 0.91, VOL4_TEXT)
    vol5 = _chunk("vol5", VOL5_DOC, 0.88, VOL5_TEXT)
    cpm = _chunk("cpm", CPM_DOC, 0.84, CPM_TEXT)
    show = _chunk("show", SHOW_DOC, 0.80, SHOW_TEXT)
    reg = _chunk("reg", REG_DOC, 0.27, REG_TEXT)
    semantic = [vol4, vol5, cpm, show]
    if register_in_semantic:
        semantic.append(reg)

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return []

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        return []

    def fake_containing_all(self, project_id, needles, k=20):
        blob = REG_TEXT.lower()
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        if cleaned and all(n in blob for n in cleaned):
            return [_chunk("reg", REG_DOC, 0.0, REG_TEXT)]
        return []

    names = {
        REG_DOC: REG_NAME,
        VOL4_DOC: VOL4_NAME,
        VOL5_DOC: VOL5_NAME,
        CPM_DOC: CPM_NAME,
        SHOW_DOC: SHOW_NAME,
    }

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
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_g1_retrieves_not_used_when_volumes_lead_the_pool(monkeypatch):
    """The live failure: Vol 5 / Vol 4 / CPM occupy the top-k; register absent."""
    ret = _install_g1_corpus(monkeypatch, register_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(G1_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks, "register row never reached the pool"
    assert REGISTER_ROW in blob
    assert "Not Used" in chunks[0].text
    assert "Works Guarantee" not in blob
    assert all(
        ret.chunk_states_schedule_register(c.text, ["schedule 10"])
        for c in chunks
    )


def test_g1_live_prefix_still_elects_the_register(monkeypatch):
    ret = _install_g1_corpus(monkeypatch, register_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(LIVE_PREFIX + G1_ASK, ACTIVE, k=5)
    assert chunks
    assert REGISTER_ROW in chunks[0].text
    assert chunks[0].doc_id == REG_DOC


def test_g1_drops_volumes_when_the_register_is_already_in_the_pool(monkeypatch):
    ret = _install_g1_corpus(monkeypatch, register_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(G1_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert REGISTER_ROW in chunks[0].text
    assert "Works Guarantee" not in blob
    assert all(c.doc_id == REG_DOC for c in chunks)


def test_a2_a3_c2_are_not_stolen_onto_the_schedule_rescue(monkeypatch):
    ret = _install_g1_corpus(monkeypatch, register_in_semantic=False)
    for ask in (A2_ASK, A3_ASK, C2_ASK):
        chunks, _ = ret.retrieve_with_filter(ask, ACTIVE, k=5)
        assert all(c.doc_id != REG_DOC for c in chunks), ask


def test_show_package_prose_is_not_a_register_row():
    """Do not invent Schedule 10 contents from another package's prose."""
    from app.core.rag.retriever import chunk_states_schedule_register

    assert not chunk_states_schedule_register(SHOW_TEXT, ["schedule 10"])


def test_inject_states_not_used_on_a_single_class_set():
    """Live miss: mixed-class SOURCE CLASS never fired; the model hedged."""
    msg = format_chunks_as_system_message(
        [_chunk("reg", REG_DOC, 0.9, REG_TEXT)],
        4,
    )
    text = msg["content"]
    assert "SCHEDULE REGISTER" in text
    assert "Not Used" in text
    assert "generic acknowledgement" in text
    assert "SOURCE CLASS" not in text


def test_inject_does_not_invent_not_used_from_volume_prose():
    msg = format_chunks_as_system_message(
        [_chunk("vol4", VOL4_DOC, 0.9, VOL4_TEXT)],
        4,
    )
    assert "SCHEDULE REGISTER" not in msg["content"]


def test_mutation_register_predicate_is_what_lifts_the_row(monkeypatch):
    ret = _install_g1_corpus(monkeypatch, register_in_semantic=True)
    monkeypatch.setattr(
        ret, "chunk_states_schedule_register", lambda *_a, **_k: False,
    )
    chunks, _ = ret.retrieve_with_filter(G1_ASK, ACTIVE, k=5)
    assert chunks
    assert chunks[0].doc_id == VOL4_DOC
