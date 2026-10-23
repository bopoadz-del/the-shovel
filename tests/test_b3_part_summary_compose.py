"""F-BAT-D H3 / WAVE 2 B3: compose Part Summary total for page d/3/1.

Live Master Corpus on 0d9fd23 asked B3 (mobile H3 = run B3):

    What is the Part Summary total for page d/3/1 of the Demolition
    and Site Clearance bill?

Expected (live pack): SAR 34,645,529.00. Attempts 1–2 produced no
response; attempt 3 said the priced-BOQ Part Summary was not found.
B1/B2 (D110 / D290.1 on the same page) and B4/B5 PASS on that SHA, so
the demolition bill is indexed — the sparse Part Summary footer lost
to line items and compose never ran (B4/B5 require a CESMM code).

Sanitized fixture S2 prints 40,560.00 on that page. Live-shaped OCR
below pins 34,645,529.00. Do not sum line items. Kill-switch:
COMPOSE_PART_SUMMARY=0 restores the empty hang.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.runtime import (
    Agent,
    _CG_REFUSAL,
    _EMPTY_RESPONSE_FALLBACK,
    _compose_part_summary_instead_of_retry,
    _graft_part_summary_total,
    _postprocess_answer,
    _should_short_circuit_part_summary,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.retriever import (
    answer_states_part_summary,
    chunk_states_part_summary_total,
    compose_part_summary_total,
    compose_priced_boq_row,
    extract_asked_boq_page_refs,
    format_part_summary_line,
    part_summary_compose_enabled,
    query_asks_for_boq_item_amount,
    query_asks_for_part_summary_total,
)
from app.core.rag.vector_store import Chunk
from tests._fixture_values import STUB_TOKEN_02


LIVE_PREFIX = "Answer only from the client project documents. "
B3_ASK = (
    "What is the Part Summary total for page d/3/1 of the "
    "Demolition and Site Clearance bill?"
)
B3_ALT = "what does page d/3/1 of the demolition bill total?"
LIVE_B3 = LIVE_PREFIX + B3_ASK
B4_ASK = (
    "What is the quantity and amount for breaking out existing "
    "carriageway including road markings (D599.5)?"
)
LIVE_B4 = LIVE_PREFIX + B4_ASK

FIXTURE_AMT = "40560"
FIXTURE_PRETTY = "40,560.00"
LIVE_AMT = "34645529"
LIVE_PRETTY = "34,645,529.00"

S2 = (
    Path(__file__).resolve().parent / "fixtures" / "ui_phys" / "S2_demolition_boq.md"
).read_text(encoding="utf-8")

LINE_ITEMS_ONLY = (
    "Demolition and Site Clearance. Page d/3/1.\n"
    "| D110 | General site clearance | 12 | ha | 2,500.00 | 30,000.00 |\n"
    "| D290.1 | Removal of trees in existing sidewalks | 48 | Nr | "
    "220.00 | 10,560.00 |\n"
)
FIXTURE_ROW = "| | Part Summary total d/3/1 | | | | 40,560.00 |"
LIVE_OCR = (
    "DEMOLITION AND SITE CLEARANCE\n"
    "Page D / 3 / 1\n"
    "D110 General site clearance 158 ha 2,500.00 395,000.00\n"
    "PART SUMMARY total d/3/1 SAR 34,645,529.00\n"
)
LIVE_FOOTER_ONLY = (
    "PART SUMMARY\n"
    "Page D / 3 / 1\n"
    "SAR 34,645,529.00"
)
OTHER_PAGE = (
    "Page d/3/3\n"
    "Part Summary total d/3/3 | | | | 21,600.00 |\n"
)
TWO_PAGES = S2
NOT_FOUND = (
    "The priced BOQ / Part Summary total for page d/3/1 was not found."
)
SEARCH_PROMISE = (
    "Let me search the demolition BOQ more specifically for the "
    "Part Summary total on page d/3/1."
)
QUESTION_ECHO = LIVE_B3


def _sys(*texts: str) -> dict:
    blocks = [
        f"[doc_id=soup chunk={i} score=0.90] {t}"
        for i, t in enumerate(texts)
    ]
    return {"role": "system", "content": "\n\n".join(blocks)}


def _msgs(ask: str) -> list[dict]:
    return [{"role": "user", "content": ask}]


def _plain(value: float) -> str:
    return str(int(round(value)))


def _has_fixture_total(text: str) -> bool:
    return FIXTURE_AMT in (text or "").replace(",", "")


def _has_live_total(text: str) -> bool:
    return LIVE_AMT in (text or "").replace(",", "")


def test_b3_ask_is_part_summary_not_cesmm_item():
    assert query_asks_for_part_summary_total(B3_ASK)
    assert query_asks_for_part_summary_total(LIVE_B3)
    assert query_asks_for_part_summary_total(B3_ALT)
    assert extract_asked_boq_page_refs(LIVE_B3) == ["d/3/1"]
    assert not query_asks_for_boq_item_amount(LIVE_B3)
    assert compose_priced_boq_row(LIVE_B3, S2) is None
    assert not query_asks_for_part_summary_total(LIVE_B4)
    assert query_asks_for_boq_item_amount(LIVE_B4)


def test_compose_parses_s2_fixture_part_summary():
    parsed = compose_part_summary_total(LIVE_B3, S2)
    assert parsed, S2
    assert parsed["page"] == "d/3/1"
    assert _plain(parsed["amount"]) == FIXTURE_AMT
    line = format_part_summary_line(parsed)
    assert _has_fixture_total(line)
    assert "d/3/1" in line
    assert FIXTURE_PRETTY in line


def test_compose_parses_live_ocr_and_spaced_page_ref():
    for blob in (LIVE_OCR, LIVE_FOOTER_ONLY):
        parsed = compose_part_summary_total(LIVE_B3, blob)
        assert parsed, blob
        assert parsed["page"] == "d/3/1"
        assert _plain(parsed["amount"]) == LIVE_AMT
        assert (parsed.get("currency") or "").upper() == "SAR"
        line = format_part_summary_line(parsed)
        assert LIVE_PRETTY in line
        assert "SAR" in line


def test_compose_does_not_sum_line_items_when_summary_absent():
    assert compose_part_summary_total(LIVE_B3, LINE_ITEMS_ONLY) is None
    # Neighbor page total is not d/3/1.
    assert compose_part_summary_total(LIVE_B3, OTHER_PAGE) is None


def test_compose_elects_asked_page_when_two_summaries_share_excerpt():
    blob = TWO_PAGES + "\n" + OTHER_PAGE
    parsed = compose_part_summary_total(LIVE_B3, blob)
    assert parsed
    assert _plain(parsed["amount"]) == FIXTURE_AMT
    assert _plain(parsed["amount"]) != "21600"


def test_compose_does_not_elect_line_item_rate_from_same_page():
    parsed = compose_part_summary_total(LIVE_B3, S2)
    assert parsed
    assert _plain(parsed["amount"]) != "220"
    assert _plain(parsed["amount"]) != "30000"
    assert _plain(parsed["amount"]) != "10560"


def test_kill_switch_restores_the_empty_hang(monkeypatch):
    monkeypatch.setenv("COMPOSE_PART_SUMMARY", "0")
    assert part_summary_compose_enabled() is False
    assert compose_part_summary_total(LIVE_B3, S2) is None
    assert compose_part_summary_total(LIVE_B3, LIVE_OCR) is None
    out = _graft_part_summary_total("", _sys(S2), _msgs(LIVE_B3))
    assert out == ""
    assert not _has_fixture_total(out)


def test_graft_replaces_empty_echo_not_found_and_search_promise():
    rag = _sys(S2)
    for hung in (
        "", QUESTION_ECHO, SEARCH_PROMISE, NOT_FOUND,
        _EMPTY_RESPONSE_FALLBACK, _CG_REFUSAL,
        "I'm ready to help. Please let me know what you need.",
    ):
        out = _graft_part_summary_total(hung, rag, _msgs(LIVE_B3))
        assert _has_fixture_total(out), hung
        assert "d/3/1" in out
        assert "not found" not in out.lower() or _has_fixture_total(out)


def test_graft_keeps_an_already_stated_total():
    rag = _sys(S2)
    already = format_part_summary_line(compose_part_summary_total(LIVE_B3, S2))
    assert _graft_part_summary_total(already, rag, _msgs(LIVE_B3)) == already


def test_postprocess_empty_hang_states_fixture_total():
    rag = _sys(S2)
    out = _postprocess_answer("", rag, _msgs(LIVE_B3))
    assert _has_fixture_total(out)
    assert out != _CG_REFUSAL


def test_postprocess_not_found_states_live_total():
    rag = _sys(LIVE_OCR)
    out = _postprocess_answer(NOT_FOUND, rag, _msgs(LIVE_B3))
    assert _has_live_total(out)
    assert "not found" not in out.lower()


def test_retry_skip_composes_instead_of_another_llm_hop():
    rag = _sys(S2)
    filled = _compose_part_summary_instead_of_retry(
        SEARCH_PROMISE, rag, _msgs(LIVE_B3),
    )
    assert _has_fixture_total(filled)
    parsed = compose_part_summary_total(LIVE_B3, S2)
    assert answer_states_part_summary(filled, parsed)


def test_retry_skip_is_off_when_kill_switch(monkeypatch):
    monkeypatch.setenv("COMPOSE_PART_SUMMARY", "0")
    rag = _sys(S2)
    assert _compose_part_summary_instead_of_retry(
        SEARCH_PROMISE, rag, _msgs(LIVE_B3),
    ) == ""


def test_inject_fires_part_summary_header_on_b3_not_b4():
    chunks = [
        Chunk(
            chunk_id="s2", project_id="p_master", doc_id="s2",
            chunk_index=0, text=S2, score=0.9,
        )
    ]
    b3 = format_chunks_as_system_message(chunks, 4, query=LIVE_B3)["content"]
    b4 = format_chunks_as_system_message(chunks, 4, query=LIVE_B4)["content"]
    assert "PART SUMMARY TOTAL" in b3
    assert FIXTURE_AMT in b3.replace(",", "")
    assert "PART SUMMARY TOTAL" not in b4


def test_short_circuit_skips_provider_when_summary_is_in_rag():
    rag = _sys(S2)
    out = _should_short_circuit_part_summary(
        rag, _msgs(LIVE_B3), has_predispatch=False,
    )
    assert _has_fixture_total(out)
    assert _should_short_circuit_part_summary(
        rag, _msgs(LIVE_B3), has_predispatch=True,
    ) == ""
    assert _should_short_circuit_part_summary(
        rag, _msgs(LIVE_B4), has_predispatch=False,
    ) == ""


def test_short_circuit_kill_switch_restores_provider_hop(monkeypatch):
    monkeypatch.setenv("COMPOSE_PART_SUMMARY", "0")
    rag = _sys(S2)
    assert _should_short_circuit_part_summary(
        rag, _msgs(LIVE_B3), has_predispatch=False,
    ) == ""


def test_chunk_states_sparse_footer_and_s2_row():
    assert chunk_states_part_summary_total(FIXTURE_ROW, ["d/3/1"])
    assert chunk_states_part_summary_total(LIVE_FOOTER_ONLY, ["d/3/1"])
    assert not chunk_states_part_summary_total(LINE_ITEMS_ONLY, ["d/3/1"])
    assert not chunk_states_part_summary_total(OTHER_PAGE, ["d/3/1"])


@pytest.mark.asyncio
async def test_chat_short_circuits_b3_without_calling_llm(monkeypatch):
    """Live 0d9fd23 H3: provider hang / not-found must not empty the turn."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True,
    )
    calls = {"n": 0}

    async def fake_call_llm(*_a, **_k):
        calls["n"] += 1
        return {"status": "error", "error": "connection refused"}

    rag = _sys(LIVE_OCR)
    audit = {
        "project_id": "p_master",
        "chunks": [
            {
                "doc_id": "soup",
                "chunk_index": 0,
                "chunk_id": "p_master:soup:0",
                "score": 0.90,
            }
        ],
    }
    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant.",
        allowed_blocks=[],
    )
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "app.agents.runtime.rag_inject",
        lambda **_k: (rag, audit),
    )

    result = await agent.chat(LIVE_B3, project_id="p_master")
    assert calls["n"] == 0
    assert result["status"] == "success"
    assert _has_live_total(result["answer"])
    assert LIVE_PRETTY in result["answer"] or LIVE_AMT in result["answer"].replace(",", "")
    assert result["iterations"] == 0


def _chunk(cid, doc_id, score, text):
    return Chunk(
        chunk_id=cid,
        project_id="p_master",
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def _install_b3_corpus(monkeypatch, *, extra_summary: bool):
    from app.core.rag import retriever as ret

    items = _chunk("items", "demo", 0.93, LINE_ITEMS_ONLY)
    summary = _chunk("sum", "demo", 0.11, LIVE_OCR if extra_summary else FIXTURE_ROW)
    semantic = [items]
    if extra_summary:
        semantic.append(summary)

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers).lower()
        hits = []
        if "part summary" in blob or "d/3/1" in blob or "3/1" in blob:
            hits.append(summary)
        if "d110" in blob or "d290" in blob:
            hits.append(items)
        return hits[:k]

    def fake_containing_all(self, project_id, needles, k=20):
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        hits = []
        for chunk in (items, summary):
            hay = (chunk.text or "").lower()
            norm = hay.replace(" ", "")
            if cleaned and all(
                n in hay or n.replace(" ", "") in norm for n in cleaned
            ):
                hits.append(chunk)
        return hits[:k]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, project_id, doc_ids, k_per_doc=12: [items, summary],
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
    monkeypatch.setattr(
        ret, "_doc_name_for_id",
        lambda did: "AGII - Infra-1 - Demolition BOQ.pdf",
        raising=False,
    )
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
    monkeypatch.delenv("COMPOSE_PART_SUMMARY", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_retrieve_b3_elects_part_summary_over_line_items(monkeypatch):
    ret = _install_b3_corpus(monkeypatch, extra_summary=True)
    chunks, _ = ret.retrieve_with_filter(LIVE_B3, "p_master", k=5)
    assert chunks
    blob = " ".join(c.text for c in chunks)
    assert LIVE_AMT in blob.replace(",", "")
    top = chunks[0].text
    assert ret.chunk_states_part_summary_total(top, ["d/3/1"])
    assert LIVE_AMT in top.replace(",", "") or any(
        LIVE_AMT in (c.text or "").replace(",", "") for c in chunks
    )
    assert all(
        ret.chunk_states_part_summary_total(c.text, ["d/3/1"]) for c in chunks
    )


def test_named_part_summary_scope_keeps_summary_drops_foreign_year():
    from app.core.rag.retriever import _ContractScope

    ask = LIVE_PREFIX + B3_ASK
    named = (
        "AB-2023-101 - Demolition and Site Clearance BOQ.pdf",
        LIVE_OCR,
    )
    other = (
        "AB-2022-202 - Demolition and Site Clearance BOQ.pdf",
        "Page d/3/1\nPart Summary total d/3/1 | | | | 9,999.00 |\n",
    )
    items = (
        "AB-2023-101 - Demolition and Site Clearance BOQ.pdf",
        LINE_ITEMS_ONLY,
    )
    scope = _ContractScope(ask, [items, named, other])
    assert scope._part_summary_in_pool
    assert scope.allow(*named)
    assert not scope.allow(*items)
    assert not scope.allow(*other)
