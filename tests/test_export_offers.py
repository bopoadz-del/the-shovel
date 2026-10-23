"""`_build_exports_from_audit` decides which download buttons the user is shown.

Audit bar 3, 2026-08-12, agents subsystem. Replacing the whole body with
`return []` left all 150 agent tests green. One test file referenced the name;
nothing asserted an offer was ever produced.

An empty list is this function's own honest answer for a turn that produced
nothing exportable, which makes `return []` an unusually good disguise: the
gutted version is indistinguishable from correct behaviour on most turns, and
the failure it causes -- the export buttons quietly stop appearing -- looks to
the user like a product that never had the feature.

The opposite failure is worse and is the reason the function is metadata-only:
its own note says an offer that 422s on click is worse than no offer. So the
tests below pin BOTH directions -- the offer appears when the data backs it,
and does not appear when it does not.

Every positive test asserts a non-empty result with the exact endpoint, never
just "no exception". `[]` is the flattering default here; asserting a
truthy-shaped absence is how this function would be tested vacuously.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _build_exports_from_audit

PROJECT = "proj-123"

# Documents keyed by id. `_is_cost_boq_source` is metadata-only: digital grid
# extensions (.xlsx/.csv) plus a boq doc_type or a name hint. A .pdf BOQ is
# deliberately NOT exportable -- scanned BOQs OOM the box.
DOCS = {
    "doc-boq": {"original_name": "Tower B Priced BOQ.xlsx", "doc_type": "boq"},
    "doc-boq-pdf": {"original_name": "Tower B Priced BOQ.pdf", "doc_type": "boq"},
    "doc-rfp": {"original_name": "Employer Requirements RFP.pdf", "doc_type": "rfp"},
    "doc-plain": {"original_name": "site meeting minutes.docx", "doc_type": "minutes"},
}


@pytest.fixture(autouse=True)
def documents(monkeypatch):
    """Resolve doc_ids without a database.

    `_build_exports_from_audit` swallows any exception from `get_document` and
    skips that source, so an unstubbed lookup would produce `[]` and every
    positive test below would fail rather than pass vacuously. Stubbing keeps
    the failure honest in the other direction too: an unknown id returns {},
    which classifies as not-exportable.
    """
    from app.core import projects

    monkeypatch.setattr(projects, "get_document", lambda doc_id: DOCS.get(doc_id, {}))
    return DOCS


def _audit(*doc_ids: str, project_id: str | None = PROJECT) -> dict:
    """An audit record whose cited chunks resolve to the given documents.

    Scores descend so the top-3 fallback in `_build_sources_from_audit` keeps
    them in the order given -- these tests depend on which documents are
    CITED, so the sources they are built from must be deterministic.
    """
    return {
        "project_id": project_id,
        "chunks": [
            {"doc_id": d, "chunk_index": i, "chunk_id": f"{d}-{i}",
             "score": 0.9 - (i * 0.1)}
            for i, d in enumerate(doc_ids)
        ],
    }


def _wbs_call(**overrides) -> dict:
    result = {
        "status": "success",
        "brief": "18-month tower fit-out",
        "target_count": 200,
        "activities_total": 187,
        "project_type": "building",
        "start_date": "2026-09-01",
    }
    result.update(overrides)
    return {"name": "generate_wbs", "result": result}


# -- the cost BOQ offer ---------------------------------------------------

def test_a_cited_cost_boq_produces_an_export_offer():
    exports = _build_exports_from_audit(_audit("doc-boq"), "per the BOQ")

    assert exports, (
        "a priced .xlsx BOQ was cited and no export was offered -- the "
        "download button silently disappears for the turn it exists for"
    )
    offer = exports[0]
    assert offer["endpoint"] == f"/v1/projects/{PROJECT}/export/cost-boq", offer
    assert offer["method"] == "POST" and offer["format"] == "xlsx", offer
    assert offer["payload"]["document_id"] == "doc-boq", offer


def test_a_scanned_pdf_boq_is_not_offered():
    """The 422-on-click case, and the OOM one. A .pdf BOQ cannot be parsed
    into a cost grid on this box, so offering it would produce a button that
    fails -- worse than no button."""
    exports = _build_exports_from_audit(_audit("doc-boq-pdf"), "per the BOQ")
    assert not any(e["endpoint"].endswith("/cost-boq") for e in exports), exports


def test_a_document_that_is_not_a_boq_produces_no_offer():
    exports = _build_exports_from_audit(_audit("doc-plain"), "per the minutes")
    assert exports == [], exports


def test_one_document_cited_twice_is_offered_once():
    """Deduplicated, per the docstring. Two identical buttons is a defect the
    user sees directly."""
    audit = _audit("doc-boq")
    audit["chunks"].append({"doc_id": "doc-boq", "chunk_index": 7,
                            "chunk_id": "doc-boq-7", "score": 0.8})
    exports = _build_exports_from_audit(audit, "per the BOQ")
    boq_offers = [e for e in exports if e["endpoint"].endswith("/cost-boq")]
    assert len(boq_offers) == 1, boq_offers


# -- the schedule offer ---------------------------------------------------

def test_a_generate_wbs_call_produces_a_schedule_offer():
    exports = _build_exports_from_audit(
        _audit(), "here is the schedule", [_wbs_call()]
    )

    assert exports, "generate_wbs ran and no schedule export was offered"
    offer = exports[0]
    assert offer["endpoint"] == f"/v1/projects/{PROJECT}/export/schedule-from-brief", offer
    assert "187 activities" in offer["label"], offer["label"]
    # The activity list is stripped from the SSE payload, so the endpoint
    # re-derives it. That only works if the brief comes along.
    assert offer["payload"]["brief"] == "18-month tower fit-out", offer
    assert offer["payload"]["project_type"] == "building", offer
    assert offer["payload"]["start_date"] == "2026-09-01", offer


def test_a_cited_rfp_switches_the_schedule_to_the_document_driven_endpoint():
    """The distinction worth pinning: with a real RFP cited, the schedule is
    driven by ITS extracted lead times rather than a generic template. Same
    button to the user, different data behind it."""
    exports = _build_exports_from_audit(
        _audit("doc-rfp"), "per the RFP", [_wbs_call()]
    )

    schedule = [e for e in exports if "schedule" in e["endpoint"]]
    assert schedule, f"no schedule offer at all: {exports}"
    offer = schedule[0]
    assert offer["endpoint"] == (
        f"/v1/projects/{PROJECT}/export/schedule-from-document"
    ), offer
    assert offer["payload"]["document_ids"] == ["doc-rfp"], offer
    assert "from documents" in offer["label"], offer["label"]


def test_a_failed_generate_wbs_produces_no_schedule_offer():
    """Offering a download for a tool call that errored is the 422-on-click
    failure in its purest form."""
    exports = _build_exports_from_audit(
        _audit(), "I could not build it",
        [{"name": "generate_wbs", "result": {"status": "error", "error": "no brief"}}],
    )
    assert exports == [], exports


def test_an_unrelated_tool_call_produces_no_schedule_offer():
    exports = _build_exports_from_audit(
        _audit(), "found 3 documents",
        [{"name": "search_project_documents", "result": {"status": "success"}}],
    )
    assert exports == [], exports


def test_only_one_schedule_is_offered_when_wbs_ran_twice():
    """`break` after the first match, iterating reversed -- the LATEST WBS
    wins. Two schedule buttons offering different activity counts is
    indistinguishable to the user from a bug in the schedule itself."""
    exports = _build_exports_from_audit(
        _audit(), "revised",
        [_wbs_call(activities_total=90), _wbs_call(activities_total=187)],
    )
    schedule = [e for e in exports if "schedule" in e["endpoint"]]
    assert len(schedule) == 1, schedule
    assert "187 activities" in schedule[0]["label"], (
        f"the superseded WBS was offered instead of the latest: {schedule[0]['label']}"
    )


# -- the refusals ---------------------------------------------------------

def test_no_project_means_no_offers():
    """Every endpoint is project-scoped. Building one without a project id
    would emit `/v1/projects/None/export/...`."""
    assert _build_exports_from_audit(_audit("doc-boq", project_id=None),
                                     "per the BOQ", [_wbs_call()]) == []


def test_a_turn_with_no_sources_and_no_tools_offers_nothing():
    assert _build_exports_from_audit({"project_id": PROJECT}, "just chatting") == []
    assert _build_exports_from_audit(None, "") == []


def test_both_offers_can_appear_on_one_turn():
    """The combined case. Each is produced by a separate pass over the same
    sources, so a change that made them mutually exclusive would still pass
    every single-offer test above."""
    exports = _build_exports_from_audit(
        _audit("doc-boq", "doc-rfp"), "per the BOQ and the RFP", [_wbs_call()]
    )
    endpoints = {e["endpoint"] for e in exports}
    assert endpoints == {
        f"/v1/projects/{PROJECT}/export/cost-boq",
        f"/v1/projects/{PROJECT}/export/schedule-from-document",
    }, endpoints


def test_audit_export_uses_ui_alias_not_drive_archive_source(monkeypatch):
    """Same H1 scoping bug: audit project_id is the remapped RAG corpus."""
    from app.core import projects as projects_mod

    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_PROJECT_ID", "master_corpus")
    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_SOURCE_PROJECT_ID", "drive_archive")
    exports = _build_exports_from_audit(
        _audit("doc-boq", project_id="drive_archive"),
        "per the BOQ",
        [_wbs_call()],
    )
    endpoints = {e["endpoint"] for e in exports}
    assert all("drive_archive" not in ep for ep in endpoints), endpoints
    assert any("/v1/projects/master_corpus/" in ep for ep in endpoints), endpoints
