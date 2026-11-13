"""S-G1(a): the answer path reads a bill's BOQ verified-total attribute.

The ingest records ``documents.metadata.boq_total`` instead of writing a
warning chunk (P-G1). The bill-rows evidence action reads it for every bill
it hands to the model: an unverified bill leads with a "do not claim an
overall total" note that states what the parser observed and never guesses a
cause (unverified does not mean scanned). A bill with no attribute, or a
verified one, adds nothing.

Every test builds its own corpus and runs each decision on two inputs.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.core.rag.evidence_actions import BILL_TOTAL_NOTE, bill_rows_action
from app.core.rag.evidence_loop import build_actions
from app.core.rag.retriever import priced_row_count


def _priced(doc_id, desc, qty, rate):
    """A tab-separated priced row: qty x rate = amount."""
    return SimpleNamespace(
        doc_id=doc_id, chunk_id=uuid.uuid4().hex, score=1.0,
        text=f"{desc}\tm2\t{qty}\t{rate:.2f}\t{qty * rate:.2f}",
    )


def _bm25(pool):
    def fn(project_id, query, k=10, doc_ids=None):
        return [c for c in pool if doc_ids is None or c.doc_id in doc_ids]
    return fn


def _status(table):
    return lambda doc_id: table.get(doc_id)


def _note_head(name):
    return BILL_TOTAL_NOTE.split("{document}")[0] + name


@pytest.mark.parametrize("doc_id,name,reason", [
    ("doc-a", "Bill of Quantities A", "not_cross_checked"),
    ("doc-b", "Priced schedule B", "pages_skipped"),
])
def test_unverified_bill_leads_with_do_not_claim_total_note(doc_id, name, reason):
    pool = [_priced(doc_id, "Membrane", 12, 4.0)]
    status = {doc_id: {"verified": False, "reason": reason, "document": name}}
    text = bill_rows_action(
        _bm25(pool), "p1", row_count=priced_row_count, total_status=_status(status),
    )("Membrane")
    assert text.startswith(_note_head(name))
    assert reason.replace("_", " ") in text
    assert "Do not state an overall contract or BOQ total" in text
    assert "scan" not in text.lower()
    assert "Membrane" in text  # the rows still follow the note


@pytest.mark.parametrize("attr", [None, {"verified": True, "document": "Bill V"}])
def test_verified_or_unstaged_bill_adds_no_note(attr):
    pool = [_priced("doc-v", "Kerb", 30, 12.0)]
    text = bill_rows_action(
        _bm25(pool), "p1", row_count=priced_row_count,
        total_status=_status({"doc-v": attr}),
    )("Kerb")
    assert "bill total status" not in text
    assert "Kerb" in text


@pytest.mark.parametrize("unverified,verified", [("doc-1", "doc-2"), ("doc-2", "doc-1")])
def test_note_names_only_the_unverified_handed_bill(unverified, verified):
    pool = [_priced(unverified, "Sub-base", 44, 3.0), _priced(verified, "Sub-base", 20, 3.5)]
    status = {
        unverified: {"verified": False, "reason": "no_priced_rows", "document": f"name-{unverified}"},
        verified: {"verified": True, "document": f"name-{verified}"},
    }
    text = bill_rows_action(
        _bm25(pool), "p1", row_count=priced_row_count, total_status=_status(status),
    )("Sub-base")
    assert text.count("[bill total status:") == 1
    assert f"name-{unverified}" in text
    assert f"name-{verified}" not in text


@pytest.mark.parametrize("item", ["Membrane", "Kerb"])
def test_failing_status_read_hands_rows_without_a_note(item):
    def boom(doc_id):
        raise RuntimeError("db down")
    pool = [_priced("doc-x", item, 5, 2.0)]
    with_reader = bill_rows_action(
        _bm25(pool), "p1", row_count=priced_row_count, total_status=boom,
    )(item)
    without = bill_rows_action(_bm25(pool), "p1", row_count=priced_row_count)(item)
    assert with_reader == without
    assert item in with_reader


@pytest.mark.parametrize("verified,expect_note", [(False, True), (True, False)])
def test_build_actions_passes_the_status_reader_to_bill_rows(verified, expect_note):
    pool = [_priced("doc-q", "Gully", 3, 90.0)]
    actions = build_actions(
        retrieve_fn=lambda q, p, k=10: ([], 0),
        bm25_fn=_bm25(pool),
        project_id="p1",
        bill_total_status=_status({"doc-q": {"verified": verified, "reason": "not_cross_checked"}}),
    )
    texts = [a("Gully") for a in actions]
    assert any("[bill total status:" in t for t in texts) is expect_note


@pytest.mark.parametrize("meta,expected", [
    ({"boq_total": {"verified": False, "reason": "not_cross_checked", "priced_rows": 7}},
     {"verified": False, "reason": "not_cross_checked", "priced_rows": 7, "document": "Bill.pdf"}),
    ({"other": 1}, None),
])
def test_runtime_reader_returns_the_attribute_with_the_document_name(monkeypatch, meta, expected):
    from app.agents import runtime
    from app.core import projects

    monkeypatch.setattr(
        projects, "get_document",
        lambda doc_id: {"id": doc_id, "original_name": "Bill.pdf", "metadata": meta},
    )
    assert runtime._bill_total_status("d1") == expected


@pytest.mark.parametrize("doc_ids", [["d1"], ["d1", "d2"]])
def test_no_reader_adds_nothing_and_logs_nothing(caplog, doc_ids):
    from app.core.rag.evidence_actions import _total_status_text

    with caplog.at_level("WARNING", logger="app.core.rag.evidence_actions"):
        assert _total_status_text(None, doc_ids) == ""
    assert not [r for r in caplog.records if "total-status" in r.getMessage()]


@pytest.mark.parametrize("reasons", [["not_cross_checked"], ["pages_skipped", "no_priced_rows"]])
def test_one_note_line_per_unverified_bill(reasons):
    from app.core.rag.evidence_actions import _total_status_text

    table = {f"d{i}": {"verified": False, "reason": r} for i, r in enumerate(reasons)}
    text = _total_status_text(table.get, list(table))
    lines = text.splitlines()
    assert len(lines) == len(reasons)
    for line, reason in zip(lines, reasons):
        assert reason.replace("_", " ") in line
