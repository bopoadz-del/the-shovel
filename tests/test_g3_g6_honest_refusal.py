"""G3/G6: Contract Data honesty over form % and commencement-pack dates.

Live F-BAT-D on BASELINE 0d9fd23:

    G3 — "What is the value of the Parent Company Guarantee?"
    Ground truth: Not required (Contract Data 4.3.7 = No). Observed:
    invented "20% of paid-up Capital and Reserves" from the Schedule 8
    form.

    G6 — "What is the Commencement Date of the contract?"
    Ground truth: tied to LOA/NOA; field not populated in tender
    Contract Data. Observed: invented 10th January 2024 from a
    Site Commencement Pack Report.

Same shape as G1 (register row over Vol 4 prose) and G4 (Rate Only
over priced lookalikes). Do not invent a value or date; elect the
Contract Data row as written. A filled CD value/date still wins.

Fixture wording only — no live client names.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.runtime import (
    _CG_REFUSAL,
    _graft_honest_contract_refusal,
    _postprocess_answer,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk


CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
G3_ASK = CATALOG["cases"]["G3"]["ask"]
G6_ASK = CATALOG["cases"]["G6"]["ask"]
G3_ALT = "how much is the parent company guarantee?"
G6_ALT = "when does the contract commence?"
LIVE_PREFIX = "Answer only from the client project documents. "
LIVE_G3 = LIVE_PREFIX + G3_ASK
LIVE_G6 = LIVE_PREFIX + G6_ASK
A2_ASK = CATALOG["cases"]["A2"]["ask"]
A3_ASK = CATALOG["cases"]["A3"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]
G1_ASK = CATALOG["cases"]["G1"]["ask"]
PACK_ASK = (
    "What date does the Site Commencement Pack Report give "
    "for site commencement?"
)
DEFINITION_PCG = "What does Parent Company Guarantee mean?"
DEFINITION_CD = "What does Commencement Date mean in FIDIC?"

S1 = (Path(__file__).parent / "fixtures" / "ui_phys" / "S1_contract_data.md")
S1_TEXT = S1.read_text(encoding="utf-8")

PCG_NOT_REQUIRED = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    "4.3.7 Parent Company Guarantee\n"
    "No. A Parent Company Guarantee is not required."
)
PCG_FILLED = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    "4.3.7 Parent Company Guarantee\n"
    "10% of the Accepted Contract Amount."
)
PCG_FORM = (
    "Schedule 8 — Form of Parent Company Guarantee. The Guarantor shall "
    "provide a guarantee in the amount of 20% of paid-up Capital and "
    "Reserves. [INSERT NAME] [INSERT DATE]"
)
COMMENCEMENT_EMPTY = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    "Commencement Date\n"
    "The Commencement Date is tied to LOA/NOA issuance. The field is "
    "not populated in this tender document."
)
COMMENCEMENT_FILLED = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    "Commencement Date: 1 March 2025"
)
COMMENCEMENT_PACK = (
    "Site Commencement Pack Report. Site commencement on "
    "10th January 2024. Briefing held with the Engineer."
)
COMMENCEMENT_TBA = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    "Commencement Date: TBA — to be notified under Sub-Clause 8.1."
)
COMMENCEMENT_BLANK = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    "Commencement Date:\n"
)

CD_NAME = (
    "AB-2023-101_QP2 Infra P1_Vol 1.0_Cond of Contract "
    "(complete)_Contract Data.pdf"
)
FORM_NAME = "Schedule 8 form of Parent Company Guarantee.pdf"
PACK_NAME = "Site Commencement Pack Report.pdf"

ACTIVE = "p_master"
CD_DOC = "cd118g"
FORM_DOC = "sched8"
PACK_DOC = "cpack"
FILLED_DOC = "cdfilled"


def _chunk(cid, doc_id, score, text):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_g3_g6_catalog_asks_are_frozen():
    assert G3_ASK == "What is the value of the Parent Company Guarantee?"
    assert G6_ASK == "What is the Commencement Date of the contract?"


def test_s1_fixture_still_states_the_honest_rows():
    """No silent GT edit — the sanitized Contract Data fixture is the pin."""
    assert "4.3.7 Parent Company Guarantee" in S1_TEXT
    assert "not required" in S1_TEXT.lower()
    assert "not populated" in S1_TEXT.lower()
    assert "LOA/NOA" in S1_TEXT


def test_g3_g6_ask_shapes_and_honesty_gates():
    from app.core.rag.retriever import (
        chunk_states_commencement_contract_data,
        chunk_states_commencement_filled_date,
        chunk_states_commencement_not_populated,
        chunk_states_commencement_pack,
        chunk_states_pcg_contract_data,
        chunk_states_pcg_filled_value,
        chunk_states_pcg_form_template,
        chunk_states_pcg_not_required,
        query_asks_for_contract_commencement_date,
        query_asks_for_contract_particulars,
        query_asks_for_parent_company_guarantee,
        query_asks_for_site_commencement_pack,
        query_asks_for_time_for_completion,
    )

    assert query_asks_for_parent_company_guarantee(G3_ASK)
    assert query_asks_for_parent_company_guarantee(LIVE_G3)
    assert query_asks_for_parent_company_guarantee(G3_ALT)
    assert not query_asks_for_parent_company_guarantee(DEFINITION_PCG)
    assert not query_asks_for_parent_company_guarantee(A5_ASK)
    assert not query_asks_for_parent_company_guarantee(
        "What is the value of the performance bond?"
    )

    assert query_asks_for_contract_commencement_date(G6_ASK)
    assert query_asks_for_contract_commencement_date(LIVE_G6)
    assert query_asks_for_contract_commencement_date(G6_ALT)
    assert not query_asks_for_contract_commencement_date(DEFINITION_CD)
    assert not query_asks_for_contract_commencement_date(A3_ASK)
    assert query_asks_for_time_for_completion(A3_ASK)
    assert not query_asks_for_contract_commencement_date(PACK_ASK)
    assert query_asks_for_site_commencement_pack(PACK_ASK)

    # Containment: G3/G6 stay off the filled-particulars pipeline.
    assert not query_asks_for_contract_particulars(G3_ASK)
    assert not query_asks_for_contract_particulars(G6_ASK)

    assert chunk_states_pcg_not_required(PCG_NOT_REQUIRED)
    assert chunk_states_pcg_contract_data(PCG_NOT_REQUIRED)
    assert not chunk_states_pcg_filled_value(PCG_NOT_REQUIRED)
    assert not chunk_states_pcg_form_template(PCG_NOT_REQUIRED)
    assert chunk_states_pcg_form_template(PCG_FORM)
    assert not chunk_states_pcg_not_required(PCG_FORM)
    assert not chunk_states_pcg_contract_data(PCG_FORM)
    assert chunk_states_pcg_filled_value(PCG_FILLED)
    assert chunk_states_pcg_contract_data(PCG_FILLED)
    assert not chunk_states_pcg_not_required(PCG_FILLED)

    assert chunk_states_commencement_not_populated(COMMENCEMENT_EMPTY)
    assert chunk_states_commencement_contract_data(COMMENCEMENT_EMPTY)
    assert not chunk_states_commencement_filled_date(COMMENCEMENT_EMPTY)
    assert chunk_states_commencement_pack(COMMENCEMENT_PACK)
    assert not chunk_states_commencement_contract_data(COMMENCEMENT_PACK)
    assert chunk_states_commencement_filled_date(COMMENCEMENT_FILLED)
    assert chunk_states_commencement_contract_data(COMMENCEMENT_FILLED)
    assert not chunk_states_commencement_not_populated(COMMENCEMENT_FILLED)
    assert chunk_states_commencement_not_populated(COMMENCEMENT_TBA)
    assert chunk_states_commencement_not_populated(COMMENCEMENT_BLANK)
    assert not chunk_states_commencement_filled_date(COMMENCEMENT_TBA)
    assert not chunk_states_commencement_filled_date(COMMENCEMENT_BLANK)


def _install_g3_corpus(monkeypatch, *, cd_in_semantic: bool):
    from app.core.rag import retriever as ret

    form = _chunk("form", FORM_DOC, 0.92, PCG_FORM)
    cd = _chunk("cd", CD_DOC, 0.24, PCG_NOT_REQUIRED)
    semantic = [form]
    if cd_in_semantic:
        semantic.append(cd)

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return []

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        return []

    def fake_containing_all(self, project_id, needles, k=20):
        blob = PCG_NOT_REQUIRED.lower()
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        if cleaned and all(n in blob for n in cleaned):
            return [_chunk("cd", CD_DOC, 0.0, PCG_NOT_REQUIRED)]
        return []

    names = {CD_DOC: CD_NAME, FORM_DOC: FORM_NAME}
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
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 2,
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


def _install_g6_corpus(monkeypatch, *, cd_in_semantic: bool):
    from app.core.rag import retriever as ret

    pack = _chunk("pack", PACK_DOC, 0.93, COMMENCEMENT_PACK)
    cd = _chunk("cd", CD_DOC, 0.21, COMMENCEMENT_EMPTY)
    semantic = [pack]
    if cd_in_semantic:
        semantic.append(cd)

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return []

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        return []

    def fake_containing_all(self, project_id, needles, k=20):
        blob = COMMENCEMENT_EMPTY.lower()
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        if cleaned and all(n in blob for n in cleaned):
            return [_chunk("cd", CD_DOC, 0.0, COMMENCEMENT_EMPTY)]
        return []

    names = {CD_DOC: CD_NAME, PACK_DOC: PACK_NAME}
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
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 2,
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


def test_g3_retrieves_not_required_when_the_form_leads_the_pool(monkeypatch):
    ret = _install_g3_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(G3_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks, "Contract Data 4.3.7 never reached the pool"
    assert "not required" in chunks[0].text.lower()
    assert "20%" not in blob
    assert "paid-up" not in blob.lower()
    assert all(ret.chunk_states_pcg_contract_data(c.text) for c in chunks)


def test_g3_live_prefix_still_elects_not_required(monkeypatch):
    ret = _install_g3_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(LIVE_G3, ACTIVE, k=5)
    assert chunks
    assert "not required" in chunks[0].text.lower()
    assert chunks[0].doc_id == CD_DOC


def test_g3_drops_the_form_when_cd_is_already_in_the_pool(monkeypatch):
    ret = _install_g3_corpus(monkeypatch, cd_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(G3_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert "not required" in chunks[0].text.lower()
    assert "20%" not in blob
    assert all(c.doc_id == CD_DOC for c in chunks)


def test_g6_retrieves_not_populated_when_the_pack_leads_the_pool(monkeypatch):
    ret = _install_g6_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(G6_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks, "empty commencement row never reached the pool"
    assert "not populated" in chunks[0].text.lower()
    assert "10th January 2024" not in blob
    assert all(
        ret.chunk_states_commencement_contract_data(c.text) for c in chunks
    )


def test_g6_live_prefix_still_elects_not_populated(monkeypatch):
    ret = _install_g6_corpus(monkeypatch, cd_in_semantic=False)
    chunks, _ = ret.retrieve_with_filter(LIVE_G6, ACTIVE, k=5)
    assert chunks
    assert "not populated" in chunks[0].text.lower()
    assert chunks[0].doc_id == CD_DOC


def test_g6_drops_the_pack_when_cd_is_already_in_the_pool(monkeypatch):
    ret = _install_g6_corpus(monkeypatch, cd_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(G6_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert "not populated" in chunks[0].text.lower()
    assert "10th January 2024" not in blob
    assert all(c.doc_id == CD_DOC for c in chunks)


def test_a2_a3_g1_are_not_stolen_onto_the_g3_rescue(monkeypatch):
    ret = _install_g3_corpus(monkeypatch, cd_in_semantic=False)
    for ask in (A2_ASK, A3_ASK, G1_ASK):
        chunks, _ = ret.retrieve_with_filter(ask, ACTIVE, k=5)
        assert all(c.doc_id != CD_DOC for c in chunks), ask


def test_pack_ask_is_not_stolen_onto_the_g6_fence(monkeypatch):
    """An explicit commencement-pack ask may still see the pack date."""
    ret = _install_g6_corpus(monkeypatch, cd_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(PACK_ASK, ACTIVE, k=5)
    assert chunks
    assert any(c.doc_id == PACK_DOC for c in chunks)


def _install_filled_cd(monkeypatch, *, pcg: bool):
    """Contract Data that DOES specify a value/date must still win."""
    from app.core.rag import retriever as ret

    lookalike = (
        _chunk("form", FORM_DOC, 0.91, PCG_FORM) if pcg
        else _chunk("pack", PACK_DOC, 0.91, COMMENCEMENT_PACK)
    )
    filled = (
        _chunk("filled", FILLED_DOC, 0.20, PCG_FILLED) if pcg
        else _chunk("filled", FILLED_DOC, 0.20, COMMENCEMENT_FILLED)
    )
    body = PCG_FILLED if pcg else COMMENCEMENT_FILLED
    semantic = [lookalike]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return []

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        return []

    def fake_containing_all(self, project_id, needles, k=20):
        blob = body.lower()
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        if cleaned and all(n in blob for n in cleaned):
            return [_chunk("filled", FILLED_DOC, 0.0, body)]
        return []

    names = {
        FILLED_DOC: CD_NAME,
        FORM_DOC: FORM_NAME,
        PACK_DOC: PACK_NAME,
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
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 2,
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
    return ret


def test_g3_still_elects_a_filled_contract_data_value(monkeypatch):
    ret = _install_filled_cd(monkeypatch, pcg=True)
    chunks, _ = ret.retrieve_with_filter(G3_ASK, ACTIVE, k=5)
    assert chunks
    assert "10%" in chunks[0].text
    assert "20%" not in " ".join(c.text for c in chunks)


def test_g6_still_elects_a_filled_contract_data_date(monkeypatch):
    ret = _install_filled_cd(monkeypatch, pcg=False)
    chunks, _ = ret.retrieve_with_filter(G6_ASK, ACTIVE, k=5)
    assert chunks
    assert "1 March 2025" in chunks[0].text
    assert "10th January 2024" not in " ".join(c.text for c in chunks)


def test_inject_states_not_required_on_a_g3_ask():
    msg = format_chunks_as_system_message(
        [_chunk("cd", CD_DOC, 0.9, PCG_NOT_REQUIRED)],
        4,
        query=LIVE_G3,
    )
    text = msg["content"]
    assert "PARENT COMPANY GUARANTEE" in text
    assert "not required" in text
    # The hint cites the clause the excerpt prints and warns off any form.
    assert "not required (Contract Data 4.3.7)" in text
    assert "guarantee form" in text


def test_inject_does_not_fire_g3_on_the_form_alone():
    msg = format_chunks_as_system_message(
        [_chunk("form", FORM_DOC, 0.9, PCG_FORM)],
        4,
        query=LIVE_G3,
    )
    assert "PARENT COMPANY GUARANTEE" not in msg["content"]


def test_inject_states_not_populated_on_a_g6_ask():
    msg = format_chunks_as_system_message(
        [_chunk("cd", CD_DOC, 0.9, COMMENCEMENT_EMPTY)],
        4,
        query=LIVE_G6,
    )
    text = msg["content"]
    assert "COMMENCEMENT DATE" in text
    assert "not populated" in text
    assert "LOA" in text


def test_inject_forbids_invention_on_pack_only_g6():
    """Pack-only RAG must still tell the model not to invent a date."""
    msg = format_chunks_as_system_message(
        [_chunk("pack", PACK_DOC, 0.9, COMMENCEMENT_PACK)],
        4,
        query=LIVE_G6,
    )
    text = msg["content"]
    assert "COMMENCEMENT DATE" in text
    assert "Do not invent" in text
    assert "not stated in Contract Data" in text


def test_inject_does_not_fire_g3_on_an_a5_ask():
    msg = format_chunks_as_system_message(
        [_chunk("cd", CD_DOC, 0.9, PCG_NOT_REQUIRED)],
        4,
        query=A5_ASK,
    )
    assert "PARENT COMPANY GUARANTEE" not in msg["content"]


def _sys(*chunk_texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=doc{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(chunk_texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


INVENTED_PCG = (
    "The Parent Company Guarantee is 20% of paid-up Capital and Reserves."
)
INVENTED_DATE = "The Commencement Date of the contract is 10th January 2024."
INVENTED_DATE_ABBREV = "The Commencement Date of the contract is 10 Jan 2024."
GENERIC_GREETING = (
    "I'm ready to help. I will answer only from the client project documents."
)


def test_graft_replaces_invented_pcg_percent():
    rag = _sys(PCG_NOT_REQUIRED)
    msgs = [{"role": "user", "content": LIVE_G3}]
    out = _graft_honest_contract_refusal(INVENTED_PCG, rag, msgs)
    assert "not required" in out.lower()
    assert "20%" not in out
    assert "paid-up" not in out.lower()


def test_graft_replaces_invented_commencement_date():
    rag = _sys(COMMENCEMENT_EMPTY)
    msgs = [{"role": "user", "content": LIVE_G6}]
    out = _graft_honest_contract_refusal(INVENTED_DATE, rag, msgs)
    assert "not populated" in out.lower()
    assert "LOA" in out or "NOA" in out
    assert "10th January 2024" not in out


def test_graft_is_a_no_op_when_honesty_is_already_stated():
    rag = _sys(PCG_NOT_REQUIRED)
    msgs = [{"role": "user", "content": LIVE_G3}]
    already = "A Parent Company Guarantee is not required per Contract Data 4.3.7."
    assert _graft_honest_contract_refusal(already, rag, msgs) == already


def test_graft_does_not_invent_when_the_excerpt_has_no_cd_row():
    rag = _sys(PCG_FORM)
    msgs = [{"role": "user", "content": LIVE_G3}]
    assert _graft_honest_contract_refusal(INVENTED_PCG, rag, msgs) == INVENTED_PCG


def test_graft_preserves_a_filled_cd_pcg_value():
    rag = _sys(PCG_FILLED)
    msgs = [{"role": "user", "content": LIVE_G3}]
    already = "The Parent Company Guarantee is 10% of the Accepted Contract Amount."
    assert _graft_honest_contract_refusal(already, rag, msgs) == already
    out = _graft_honest_contract_refusal(INVENTED_PCG, rag, msgs)
    assert "10%" in out
    assert "20%" not in out


def test_graft_preserves_a_filled_cd_commencement_date():
    rag = _sys(COMMENCEMENT_FILLED)
    msgs = [{"role": "user", "content": LIVE_G6}]
    already = "The Commencement Date of the contract is 1 March 2025."
    assert _graft_honest_contract_refusal(already, rag, msgs) == already
    out = _graft_honest_contract_refusal(INVENTED_DATE, rag, msgs)
    assert "1 March 2025" in out
    assert "10th January 2024" not in out


def test_postprocess_g3_invention_states_not_required():
    rag = _sys(PCG_NOT_REQUIRED)
    msgs = [{"role": "user", "content": LIVE_G3}]
    out = _postprocess_answer(INVENTED_PCG, rag, msgs)
    assert "not required" in out.lower()
    assert "20%" not in out
    assert out != _CG_REFUSAL


def test_postprocess_g6_invention_states_not_populated():
    rag = _sys(COMMENCEMENT_EMPTY)
    msgs = [{"role": "user", "content": LIVE_G6}]
    out = _postprocess_answer(INVENTED_DATE, rag, msgs)
    assert "not populated" in out.lower()
    assert "10th January 2024" not in out
    assert out != _CG_REFUSAL


def test_postprocess_g3_greeting_states_not_required():
    rag = _sys(PCG_NOT_REQUIRED)
    msgs = [{"role": "user", "content": LIVE_G3}]
    out = _postprocess_answer(GENERIC_GREETING, rag, msgs)
    assert "not required" in out.lower()


def test_mutation_pcg_predicate_is_what_lifts_the_row(monkeypatch):
    ret = _install_g3_corpus(monkeypatch, cd_in_semantic=True)
    monkeypatch.setattr(
        ret, "chunk_states_pcg_contract_data", lambda *_a, **_k: False,
    )
    chunks, _ = ret.retrieve_with_filter(G3_ASK, ACTIVE, k=5)
    assert chunks
    assert chunks[0].doc_id == FORM_DOC


def _install_g6_pack_only_no_cd(monkeypatch):
    """Live leak after #563: pack retrieved, Contract Data rescue misses."""
    from app.core.rag import retriever as ret

    pack = _chunk("pack", PACK_DOC, 0.93, COMMENCEMENT_PACK)

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [pack][:k] if project_id == ACTIVE else []

    def fake_empty(self, *args, **kwargs):
        return []

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_empty,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs", fake_empty,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all", fake_empty,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 1,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: PACK_NAME, raising=False)
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda pid, phrase, limit=8: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_g6_drops_pack_even_when_cd_is_absent(monkeypatch):
    """Pack dates must not reach the model when Contract Data is missing."""
    ret = _install_g6_pack_only_no_cd(monkeypatch)
    chunks, _ = ret.retrieve_with_filter(G6_ASK, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert "10th January 2024" not in blob
    assert all(not ret.chunk_states_commencement_pack(c.text) for c in chunks)


def test_graft_refuses_invented_date_when_rag_is_pack_only():
    rag = _sys(COMMENCEMENT_PACK)
    msgs = [{"role": "user", "content": LIVE_G6}]
    out = _graft_honest_contract_refusal(INVENTED_DATE, rag, msgs)
    assert "10th January 2024" not in out
    assert "will not invent" in out.lower()
    out_abbrev = _graft_honest_contract_refusal(INVENTED_DATE_ABBREV, rag, msgs)
    assert "10 Jan 2024" not in out_abbrev
    assert "will not invent" in out_abbrev.lower()


def test_graft_refuses_invented_date_when_rag_is_empty():
    msgs = [{"role": "user", "content": LIVE_G6}]
    out = _graft_honest_contract_refusal(INVENTED_DATE, None, msgs)
    assert "10th January 2024" not in out
    assert "will not invent" in out.lower()


def test_graft_refuses_tba_and_blank_cd_inventions():
    for excerpt in (COMMENCEMENT_TBA, COMMENCEMENT_BLANK):
        rag = _sys(excerpt)
        msgs = [{"role": "user", "content": LIVE_G6}]
        out = _graft_honest_contract_refusal(INVENTED_DATE_ABBREV, rag, msgs)
        assert "10 Jan 2024" not in out, excerpt
        assert "not populated" in out.lower() or "will not invent" in out.lower()


def test_postprocess_g6_pack_only_strips_invented_date():
    rag = _sys(COMMENCEMENT_PACK)
    msgs = [{"role": "user", "content": LIVE_G6}]
    out = _postprocess_answer(INVENTED_DATE_ABBREV, rag, msgs)
    assert "10 Jan 2024" not in out
    assert "will not invent" in out.lower()
    assert out != _CG_REFUSAL


def test_g6_contract_commencement_phrasing_is_detected():
    from app.core.rag.retriever import (
        answer_invents_commencement_date,
        query_asks_for_contract_commencement_date,
    )

    assert query_asks_for_contract_commencement_date(
        "What is the contract commencement?"
    )
    assert answer_invents_commencement_date(INVENTED_DATE_ABBREV)
    assert answer_invents_commencement_date(
        "Site start was 2024-01-10 according to the pack."
    )
