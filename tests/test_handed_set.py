"""S-R5: the handed first-pass set is bounded by characters, diverse across
documents, and a multi-source document ask plans more than one retrieval.

Every chunk, document name and question here is synthetic and built by the
test itself. Each rule is pinned on two inputs.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from app.core.rag import handed_set as hs
from app.core.rag.vector_store import Chunk
from app.infra.monitoring import request_id_ctx
from tests._fixture_values import STUB_TOKEN_02

TRACE_EVENT = "rag_retrieval_trace"


def _c(doc, idx, score, text_len=400, name=None, layer="own", text=None):
    body = text if text is not None else (f"synthetic body {doc} {idx} " + "x" * text_len)[:text_len]
    return Chunk(
        chunk_id=f"p:{doc}:{idx}", project_id="p", doc_id=doc, chunk_index=idx,
        text=body, score=score, source_name=name if name is not None else f"{doc}.pdf",
        layer=layer,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("RAG_HANDED_CHAR_BUDGET",
                 "RAG_HANDED_POOL_FACTOR", "RAG_HANDED_DOC_SHARE",
                 "RAG_HANDED_DOC_MARGIN", "RAG_K", "RAG_RETRIEVAL_TRACE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("MAX_RAG_TOKENS", "100000")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


# --- (a) a character budget, not a count of two -----------------------------

@pytest.mark.parametrize("size,budget,expected", [
    (300, 1_000, 3),    # three 300-char excerpts fit 1,000
    (900, 1_000, 1),    # only one 900-char excerpt fits
])
def test_char_budget_decides_how_many_fit(size, budget, expected):
    chunks = [_c(f"d{i}", 0, 0.9 - i * 0.01, text_len=size) for i in range(6)]
    assert len(hs.fit_char_budget(chunks, budget)) == expected


@pytest.mark.parametrize("first_len", [5_000, 20_000])
def test_first_chunk_kept_even_over_budget_and_shorter_ones_fill_after(first_len):
    chunks = [_c("big", 0, 0.9, text_len=first_len),
              _c("mid", 0, 0.8, text_len=3_000),
              _c("small", 0, 0.7, text_len=200)]
    kept = hs.fit_char_budget(chunks, 4_000)
    assert kept[0].doc_id == "big"
    assert [c.doc_id for c in kept[1:]] == []  # nothing else fits beside it
    kept2 = hs.fit_char_budget(chunks[1:], 4_000)
    assert [c.doc_id for c in kept2] == ["mid", "small"]


@pytest.mark.parametrize("value", [12_345, 2_222])
def test_char_budget_reads_its_setting(value, monkeypatch):
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(value))
    assert hs.char_budget() == value
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", "not-a-number")
    assert hs.char_budget() == hs.DEFAULT_CHAR_BUDGET


@pytest.mark.parametrize("old_setting", ["1", "6000"])
def test_removed_degraded_setting_changes_nothing(old_setting, monkeypatch):
    """The uncap: the old degraded character budget is not read any more."""
    monkeypatch.setenv("RAG_DEGRADED_CHAR_BUDGET", old_setting)
    assert hs.char_budget() == hs.DEFAULT_CHAR_BUDGET
    assert not hasattr(hs, "DEFAULT_DEGRADED_CHAR_BUDGET")


@pytest.mark.parametrize("k,factor,pool", [(5, None, 10), (7, "3", 21)])
def test_pool_is_wider_than_k(k, factor, pool, monkeypatch):
    if factor:
        monkeypatch.setenv("RAG_HANDED_POOL_FACTOR", factor)
    assert hs.pool_k(k) == pool


# --- (b) diversity across documents, families counted once ------------------

@pytest.mark.parametrize("other_score,other_leads", [
    (0.80, True),   # another document scores within the margin -> it gets a slot
    (0.50, False),  # no competitor inside the margin -> the one document keeps them all
])
def test_one_document_is_capped_only_while_others_compete(other_score, other_leads):
    a = [_c("docA", i, 0.90 - i * 0.01) for i in range(5)]
    b = [_c("docB", 0, other_score)]
    out = hs.diversify(a + b, slots=5, share=0.4, margin=0.15)
    order = [c.doc_id for c in out]
    assert len(out) == 6  # deferred, never dropped
    # competing: docA takes its share (2), docB comes third, docA's rest after
    assert order.index("docB") == (2 if other_leads else 5)


@pytest.mark.parametrize("name_a,name_b", [
    ("Folder1/Method Statement Piling.pdf", "Folder2/Method Statement Piling.pdf"),
    ("1712345678901_Spec Section 03.docx", "Spec Section 03 (1).docx"),
])
def test_same_title_in_two_places_is_one_family(name_a, name_b):
    x = SimpleNamespace(source_name=name_a, doc_id="one")
    y = SimpleNamespace(source_name=name_b, doc_id="two")
    assert hs.family_key(x) == hs.family_key(y)
    z = SimpleNamespace(source_name="Different Title.pdf", doc_id="three")
    assert hs.family_key(x) != hs.family_key(z)


@pytest.mark.parametrize("copy_name", ["Report/Copy A.pdf", "Archive/Copy A.pdf"])
def test_duplicate_copy_does_not_take_a_second_slot(copy_name):
    same = "identical clause text of the copied document"
    orig = _c("orig", 0, 0.9, name="Live/Copy A.pdf", text=same)
    dup = _c("dup", 0, 0.89, name=copy_name, text=same)
    other = _c("other", 0, 0.85, name="Other.pdf")
    out = hs.diversify([orig, dup, other], slots=5, share=0.4, margin=0.15)
    assert [c.doc_id for c in out] == ["orig", "other"]


@pytest.mark.parametrize("n_docs", [3, 4])
def test_select_handed_reports_allotment(n_docs):
    chunks = [_c(f"d{i}", j, 0.9 - i * 0.02 - j * 0.001, text_len=300)
              for i in range(n_docs) for j in range(3)]
    chunks.append(_c("gk", 0, 0.88, text_len=300, layer="general_knowledge"))
    handed, allot = hs.select_handed(chunks, k=5)
    assert allot["families"] == n_docs + 1
    assert allot["layers"]["general_knowledge"] == 1
    assert allot["layers"]["own"] == len(handed) - 1
    assert allot["chars_handed"] == sum(len(c.text) for c in handed)
    assert allot["char_budget"] == hs.DEFAULT_CHAR_BUDGET


# --- (a)+(b) through rag_inject: the daily counter never shortens the set ---

def _inject(monkeypatch, pool, degraded, ask="what does the method statement say about piling?"):
    from app.core.rag import inject

    seen = {}

    def fake_retrieve(query, project_id, k=5, **kwargs):
        seen["k"] = k
        return list(pool)[:k], 0

    monkeypatch.setattr(inject, "retrieve_with_filter", fake_retrieve)
    monkeypatch.setattr(inject._budget, "snapshot", lambda day: {
        "day": day, "consumed": 9_000_000 if degraded else 9, "budget": 10,
        "remaining": 0 if degraded else 1, "degraded": degraded,
    })
    monkeypatch.setattr(inject._budget, "consume", lambda **kw: None)
    token = request_id_ctx.set("rid-sr5")
    try:
        msg, audit = inject.rag_inject(
            user_message=ask, project_id="p", conversation_id="ws-p-1",
            user_id="u", agent_name="project-assistant",
        )
    finally:
        request_id_ctx.reset(token)
    return msg, audit, seen


@pytest.mark.parametrize("degraded", [False, True])
def test_short_excerpts_from_several_documents_are_all_handed(degraded, monkeypatch):
    pool = [_c(f"d{i}", 0, 0.9 - i * 0.02, text_len=500) for i in range(5)]
    _msg, audit, seen = _inject(monkeypatch, pool, degraded)
    assert seen["k"] == 10  # pool of 2 x k=5, also when degraded
    assert audit["injected_k"] == 5
    assert len({c["doc_id"] for c in audit["chunks"]}) == 5


@pytest.mark.parametrize("size", [2_500, 1_000])
def test_spent_daily_budget_hands_the_same_set(size, monkeypatch):
    """Owner order (uncap): consumed far above the limit hands exactly what
    consumed below it hands."""
    pool = [_c(f"d{i}", 0, 0.9 - i * 0.02, text_len=size) for i in range(8)]
    _m, below, _s = _inject(monkeypatch, pool, False)
    _m, above, _s = _inject(monkeypatch, pool, True)
    assert below["budget_degraded"] is False and above["budget_degraded"] is True
    assert [c["chunk_id"] for c in above["chunks"]] == [c["chunk_id"] for c in below["chunks"]]
    assert above["injected_k"] == below["injected_k"] > 2


@pytest.mark.parametrize("degraded,consumed", [(True, 9_000_000), (False, 9)])
def test_trace_prints_requested_effective_k_and_allotment(degraded, consumed, monkeypatch):
    lg = logging.getLogger("app.rag.retrieval_trace")
    records = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = H(level=logging.DEBUG)
    monkeypatch.setattr(lg, "disabled", False)
    monkeypatch.setattr(lg, "level", logging.INFO)
    lg.addHandler(h)
    try:
        pool = [_c("a", i, 0.9 - i * 0.01, text_len=300) for i in range(4)]
        pool += [_c("b", 0, 0.86, text_len=300, layer="general_knowledge")]
        _inject(monkeypatch, pool, degraded)
    finally:
        lg.removeHandler(h)
    lines = [r.getMessage() for r in records if r.getMessage().startswith(TRACE_EVENT)]
    assert len(lines) == 1
    p = json.loads(lines[0][len(TRACE_EVENT):].strip())
    assert p["requested_k"] == 5 and p["effective_k"] == 5 and p["pool_k"] == 10
    assert p["budget_degraded"] is degraded
    assert p["budget_consumed"] == consumed
    assert p["allotment"]["char_budget"] == hs.DEFAULT_CHAR_BUDGET
    assert p["allotment"]["layers"] == {"own": 4, "general_knowledge": 1}


# --- (c) a multi-source document ask plans more than one retrieval ----------

_THREE = "Compare the bill of quantities item, the drawing and the method statement for the pile caps."
_ONE = "What does the method statement say about pile cap formwork?"


def _first_pass(doc_ids):
    return "\n".join(f"[doc_id={d} chunk=0 score=0.800 class=project_corpus] text" for d in doc_ids)


@pytest.mark.parametrize("question,kinds", [
    (_THREE, ("bill", "drawing", "method_statement")),
    ("Which specification clause and which RFI cover the waterproofing?", ("specification", "rfi")),
])
def test_named_source_kinds(question, kinds):
    assert hs.named_source_kinds(question) == kinds


@pytest.mark.parametrize("docs,wanted", [
    (["d1", "d1"], True),          # two chunks of one document, three kinds named
    (["d1", "d2", "d3"], False),   # three sources already in hand
])
def test_plan_when_first_pass_covers_fewer_sources_than_named(docs, wanted):
    assert hs.wants_source_coverage_retrieval(_THREE, _first_pass(docs)) is wanted


@pytest.mark.parametrize("docs", [["d1"], ["d1", "d1"]])
def test_single_kind_ask_never_triggers(docs):
    assert hs.wants_source_coverage_retrieval(_ONE, _first_pass(docs)) is False


@pytest.mark.parametrize("docs,ran", [(["d1", "d1"], True), (["d1", "d2", "d3"], False)])
def test_runtime_runs_the_loop_for_a_non_figure_multi_source_ask(docs, ran, monkeypatch):
    import asyncio

    from app.agents import runtime
    from app.core.rag import route

    calls = []

    async def fake_augment(**kwargs):
        calls.append(kwargs["question"])
        return kwargs["first_pass_evidence"] + "\n[doc_id=d9 chunk=1 score=0.7] more"

    monkeypatch.setattr(route, "augment_figure_retrieval", fake_augment)
    monkeypatch.setattr(runtime, "_first_pass_composes_figure", lambda *a: False)
    monkeypatch.setattr(runtime, "_agentic_figure_retrieval_excluded", lambda t: False)

    class _Store:
        def bm25_search(self, *a, **k):
            return []

    monkeypatch.setattr("app.core.rag.vector_store.get_store", lambda: _Store())
    # Every excerpt carries the question's words, so the first pass covers
    # the question's terms and only the named-source-kind rule decides.
    msg = {"role": "system", "content": _first_pass(docs).replace("] text", "] " + _THREE)}
    out = asyncio.run(runtime._agentic_figure_retrieval_augment(
        agent=SimpleNamespace(), user_message=_THREE, project_id="p",
        api_key="k", user_id="u", rag_sys_msg=msg,
    ))
    assert bool(calls) is ran
    assert ("doc_id=d9" in out["content"]) is ran


@pytest.mark.parametrize("excluded", [True, False])
def test_runtime_exclusions_still_apply(excluded, monkeypatch):
    from app.agents import runtime

    monkeypatch.setattr(runtime, "_agentic_figure_retrieval_excluded", lambda t: excluded)
    assert runtime._wants_source_coverage_plan(_THREE, _first_pass(["d1"])) is (not excluded)


# --- guards the mutation gate pins ------------------------------------------

@pytest.mark.parametrize("raw", [None, "not-a-score", True])
def test_non_numeric_score_ranks_as_zero_not_a_crash(raw):
    odd = SimpleNamespace(score=raw, doc_id="odd", source_name="odd.pdf", text="t odd")
    good = _c("good", 0, 0.5)
    out = hs.diversify([odd, good], slots=5, share=0.4, margin=0.15)
    assert [c.doc_id for c in out] == ["good", "odd"]


@pytest.mark.parametrize("slots", [1, 5])
def test_empty_pool_hands_nothing(slots):
    assert hs.diversify([], slots=slots, share=0.4, margin=0.15) == []
    handed, allot = hs.select_handed([], k=slots)
    assert handed == [] and allot["families"] == 0


@pytest.mark.parametrize("first_pass", ["", "plain text with no excerpt markers"])
def test_one_named_kind_never_plans_even_with_no_sources(first_pass):
    assert hs.wants_source_coverage_retrieval(_ONE, first_pass) is False
    assert hs.wants_source_coverage_retrieval(_THREE, first_pass) is True
