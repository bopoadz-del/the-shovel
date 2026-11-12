"""Token budget for the evidence-loop merge (production 400 defect).

A priced-BOQ ask whose first pass missed used to merge the whole matched
document (hundreds of chunks) into the prompt with no bound and 400 on the
model's context limit. These pin the general budget: a small document is
injected whole, a large synthetic sheet is capped to the highest-scoring rows
with a ``context_truncated`` provenance note, and the whole-turn merge has a
total cap. Two inputs per decision, no project/item/column names.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core import health_probes as hp
from app.core.rag import evidence_budget as eb
from app.core.rag.evidence_actions import semantic_action
from app.core.rag.evidence_loop import run_loop
from app.core.rag.evidence_plan import parse_evidence_plan


def _chunk(text, score=0.0, doc_id="doc"):
    return SimpleNamespace(text=text, score=score, doc_id=doc_id)


@pytest.fixture(autouse=True)
def _reset_provider_state():
    with hp._PROVIDER_LOCK:
        hp._PROVIDER_LAST.clear()
        hp._PROVIDER_COUNTS.update({"billing_402": 0, "rate_429": 0})
    hp._LAST_SERVED_AT = 0.0
    yield


# ── budget: small document injected whole ────────────────────────────────
@pytest.mark.parametrize("chunks", [
    [_chunk("Bulk excavation in all classes", 0.9), _chunk("Blinding 75 mm", 0.8)],
    [_chunk("D599.5 carriageway 340904 m2", 0.95), _chunk("D549.2 culvert", 0.85)],
])
def test_small_document_is_injected_whole(chunks):
    kept, dropped = eb.cap_chunks(chunks)
    assert dropped == 0
    assert {c.text for c in kept} == {c.text for c in chunks}
    text = semantic_action(
        lambda q, pid, k: ([_chunk(c.text, c.score, c.doc_id) for c in chunks], 0),
        "p",
    )("q")
    assert eb.CONTEXT_TRUNCATED_NOTE not in text


# ── budget: large synthetic sheet capped with the note ───────────────────
def _big_sheet(n):
    return [
        _chunk(f"row {i} description with quantity and rate {i}", score=float(i), doc_id="sheet")
        for i in range(n)
    ]


@pytest.mark.parametrize("rows,total,per_doc", [
    (200, 500, 400),
    (80, 200, 160),
])
def test_large_synthetic_sheet_is_capped_with_note(rows, total, per_doc):
    # Each row is ~40 chars ≈ 10 tokens, so these budgets keep a prefix only.
    chunks = _big_sheet(rows)
    kept, dropped = eb.cap_chunks(
        chunks, total_tokens=total, per_doc_tokens=per_doc,
    )
    assert dropped > 0
    assert len(kept) < len(chunks)
    # Highest-scoring rows win (whole-chunk, no mid-row truncation).
    assert kept[0].text == chunks[-1].text


@pytest.mark.parametrize("rows,total,per_doc", [
    (120, 500, 400),
    (60, 200, 160),
])
def test_action_budget_appends_truncation_note(rows, total, per_doc):
    chunks = _big_sheet(rows)
    from app.core.rag.evidence_actions import _budgeted_text
    text = _budgeted_text(
        chunks, total_tokens=total, per_doc_tokens=per_doc,
    )
    assert eb.CONTEXT_TRUNCATED_NOTE in text
    # The highest-scoring row still leads the kept sample.
    assert chunks[-1].text in text


# ── whole-turn merge total cap ───────────────────────────────────────────
@pytest.mark.parametrize("plan_text,fill", [
    ("bulk excavation, blinding", {"blinding": "x" * 4000}),
    ("formwork, painting", {"painting": "y" * 4000}),
])
def test_merge_total_cap_appends_note(monkeypatch, plan_text, fill):
    monkeypatch.setenv("AGENTIC_EVIDENCE_MAX_TOKENS", "2")
    plan = parse_evidence_plan(plan_text)

    def _retrieve(q, pid, k=5):
        return [_chunk("small first pass", 1.0)], 0

    def _bm25(pid, q, k=50):
        return [_chunk(fill.get(q, "short"), 0.5)]

    from app.core.rag.evidence_loop import build_actions
    actions = build_actions(retrieve_fn=_retrieve, bm25_fn=_bm25, project_id="p")
    run = run_loop(plan, actions, initial_evidence="first pass " * 50)
    assert eb.CONTEXT_TRUNCATED_NOTE in run.evidence


# ── per-function budget guards (kill the CI-M mutants) ───────────────────
@pytest.mark.parametrize("score,expected", [
    (1.5, 1.5),
    (3, 3.0),
    (True, 0.0),
    (None, 0.0),
    ("high", 0.0),
])
def test_chunk_score_ranks_only_numeric_scores(score, expected):
    assert eb._chunk_score(SimpleNamespace(score=score)) == expected


@pytest.mark.parametrize("doc_id", ["doc-a", "doc-b"])
def test_chunk_doc_id_reads_the_doc_id(doc_id):
    assert eb._chunk_doc_id(SimpleNamespace(doc_id=doc_id)) == doc_id
    assert eb._chunk_doc_id(SimpleNamespace()) == ""


@pytest.mark.parametrize("raw,default,expected", [
    ("", 7, 7),
    ("3", 7, 3),
    ("0", 7, 7),
    ("-2", 7, 7),
    ("abc", 7, 7),
])
def test_env_int_parses_and_guards(monkeypatch, raw, default, expected):
    monkeypatch.setenv("TST_ENV_INT", raw)
    assert eb._env_int("TST_ENV_INT", default) == expected


@pytest.mark.parametrize("text,cap,expected_body,expected_truncated", [
    ("x" * 40, 100, "x" * 40, False),
    ("y" * 200, 10, "y" * 40, True),
])
def test_cap_evidence_text_keeps_the_front(text, cap, expected_body, expected_truncated):
    out, truncated = eb.cap_evidence_text(text, max_tokens=cap)
    assert out == expected_body
    assert truncated == expected_truncated


@pytest.mark.parametrize("count,per_doc,expected", [
    (2, 10, 1),
    (3, 20, 2),
])
def test_per_doc_cap_binds_before_total(count, per_doc, expected):
    # Each chunk is 40 chars ≈ 10 tokens, all under one document id.
    chunks = [_chunk("x" * 40, score=float(count - i)) for i in range(count)]
    kept, _dropped = eb.cap_chunks(chunks, total_tokens=1000, per_doc_tokens=per_doc)
    assert len(kept) == expected


@pytest.mark.parametrize("strings,total", [
    (["A" * 200, "B" * 200], 10),
    (["C" * 300, "D" * 300], 20),
])
def test_plain_string_input_is_budgeted_as_one_text(strings, total):
    from app.core.rag.evidence_actions import _budgeted_text
    text = _budgeted_text(strings, total_tokens=total)
    body = text[: -len(eb.CONTEXT_TRUNCATED_NOTE)]
    assert eb.CONTEXT_TRUNCATED_NOTE in text
    assert body == strings[0][: total * 4]


# ── readiness: request-shaped 4xx must not poison serving ────────────────
@pytest.mark.parametrize("http_status", [400, 413])
def test_request_shaped_4xx_does_not_poison_serving(http_status):
    hp.record_provider_result("primary", "ok")
    hp.record_provider_result("primary", "error", http_status)
    snap = hp.provider_snapshot()
    assert snap["serving"] is True
    assert snap["providers"]["primary"]["status"] == "ok"


@pytest.mark.parametrize("http_status,status", [(402, "exhausted"), (429, "exhausted"), (500, "error")])
def test_provider_failures_still_mark_not_serving(http_status, status):
    hp.record_provider_result("primary", status, http_status)
    assert hp.provider_snapshot()["serving"] is False
