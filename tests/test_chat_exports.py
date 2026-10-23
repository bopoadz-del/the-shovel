"""SSE end-event `exports` descriptors — data-backed download offers.

The chat bubble already exposes an inline "Download" affordance under each
assistant answer (no separate UI buttons). This contract extends it: when the
answer cites a document the platform can turn into a real artifact, the end
event carries an `exports` descriptor so the bubble can offer that artifact
directly.

Pilot scope (intentional): only the **cost BOQ** is offerable from chat — its
export endpoint self-derives from a `document_id`. The gate is cheap metadata
only (extension + name); boq_processor is NEVER run here in the hot path (a
large scanned BOQ OOMs the box — memory `the-fork-boq-always-xlsx`), so the
offer is restricted to digital xlsx/csv BOQs that actually parse. Schedule/EVM
are deferred — chat turns don't yet produce their inline activities/periods.
"""
from __future__ import annotations

from app.agents.runtime import _build_exports_from_audit


def _audit_citing(doc_id: str, chunk_index: int, project_id: str = "proj1") -> dict:
    return {
        "project_id": project_id,
        "chunks": [{"doc_id": doc_id, "chunk_index": chunk_index, "score": 0.82}],
    }


def _stub_doc(monkeypatch, original_name: str, doc_type: str | None = None) -> None:
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"id": did, "original_name": original_name, "doc_type": doc_type},
    )


def test_cited_xlsx_boq_yields_cost_boq_export(monkeypatch):
    audit = _audit_citing("d1", 3)
    _stub_doc(monkeypatch, "the client project Priced BOQ.xlsx", doc_type="boq")
    text = "The priced bill totals SAR 1.66bn [source: the client project Priced BOQ.xlsx, chunk 3]."

    out = _build_exports_from_audit(audit, text)

    assert len(out) == 1
    exp = out[0]
    assert exp["format"] == "xlsx"
    assert exp["method"] == "POST"
    assert exp["endpoint"] == "/v1/projects/proj1/export/cost-boq"
    assert exp["payload"]["document_id"] == "d1"
    # A human-facing label so the bubble can render it without inventing one.
    assert "BOQ" in exp["label"]


def test_csv_boq_is_also_offerable(monkeypatch):
    audit = _audit_citing("d2", 1)
    _stub_doc(monkeypatch, "tender boq.csv")
    text = "See the tender pricing [source: tender boq.csv, chunk 1]."

    out = _build_exports_from_audit(audit, text)

    assert len(out) == 1
    assert out[0]["payload"]["document_id"] == "d2"


def test_scanned_pdf_boq_yields_no_export(monkeypatch):
    """A PDF named like a BOQ is NOT offered — scanned BOQs don't parse and
    OOM the box. The extension gate is the safety boundary."""
    audit = _audit_citing("d3", 2)
    _stub_doc(monkeypatch, "Priced BOQ scan.pdf", doc_type="boq")
    text = "Per the priced BOQ [source: Priced BOQ scan.pdf, chunk 2]."

    out = _build_exports_from_audit(audit, text)

    assert out == []


def test_non_boq_xlsx_yields_no_export(monkeypatch):
    """An xlsx that isn't a BOQ (by name or doc_type) is not a cost-BOQ source."""
    audit = _audit_citing("d4", 0)
    _stub_doc(monkeypatch, "Manpower Histogram.xlsx", doc_type="schedule")
    text = "Resource loading [source: Manpower Histogram.xlsx, chunk 0]."

    out = _build_exports_from_audit(audit, text)

    assert out == []


def test_no_cited_sources_yields_no_export(monkeypatch):
    """No retrieval / a declined answer → no download offer (nothing to back it)."""
    out = _build_exports_from_audit({"chunks": []}, "I could not find that.")
    assert out == []


def test_descriptor_deduplicated_per_document(monkeypatch):
    """The same BOQ cited via two chunks yields ONE export offer, not two."""
    audit = {
        "project_id": "proj1",
        "chunks": [
            {"doc_id": "d1", "chunk_index": 3, "score": 0.82},
            {"doc_id": "d1", "chunk_index": 7, "score": 0.80},
        ],
    }
    _stub_doc(monkeypatch, "Priced BOQ.xlsx", doc_type="boq")
    text = "Totals [source: Priced BOQ.xlsx, chunk 3] and detail [source: Priced BOQ.xlsx, chunk 7]."

    out = _build_exports_from_audit(audit, text)

    assert len(out) == 1


def _wbs_call(status="success", total=225):
    return {
        "name": "generate_wbs", "ok": status == "success",
        "result": {
            "status": status, "brief": "Anthropic hyperscale data center",
            "project_type": "data_center", "target_count": 200,
            "start_date": "2026-01-05", "activities_total": total,
        },
    }


def test_generate_wbs_call_yields_schedule_export():
    """A generate_wbs turn offers the cost-loaded schedule download — even when
    the turn cited no RAG documents (a schedule from a brief needn't cite)."""
    out = _build_exports_from_audit({"project_id": "proj1", "chunks": []}, "Here is the schedule.",
                                    [_wbs_call()])
    assert len(out) == 1
    exp = out[0]
    assert exp["endpoint"] == "/v1/projects/proj1/export/schedule-from-brief"
    assert exp["method"] == "POST" and exp["format"] == "xlsx"
    assert exp["payload"]["brief"] == "Anthropic hyperscale data center"
    assert exp["payload"]["project_type"] == "data_center"
    assert exp["payload"]["start_date"] == "2026-01-05"
    assert "225" in exp["label"]


def test_failed_wbs_call_yields_no_schedule_export():
    out = _build_exports_from_audit({"project_id": "proj1"}, "", [_wbs_call(status="error")])
    assert out == []


def test_non_wbs_tool_calls_yield_no_schedule_export():
    calls = [{"name": "search_project_documents", "ok": True, "result": {"status": "success"}}]
    out = _build_exports_from_audit({"project_id": "proj1"}, "", calls)
    assert out == []


def test_only_one_schedule_offer_for_multiple_wbs_calls():
    out = _build_exports_from_audit({"project_id": "proj1"}, "",
                                    [_wbs_call(total=200), _wbs_call(total=225)])
    scheds = [e for e in out if "schedule-from-brief" in e["endpoint"]]
    assert len(scheds) == 1
    assert "225" in scheds[0]["label"]  # latest WBS wins (reversed scan)


def test_wbs_with_cited_rfp_yields_document_driven_schedule(monkeypatch):
    """A generate_wbs turn that cited an RFP/BOD spec offers the DOCUMENT-driven
    schedule (real lead times) instead of the generic brief-driven one."""
    audit = _audit_citing("rfp1", 1)
    _stub_doc(monkeypatch, "Example Client RFP - Basis of Design.docx", doc_type="spec")
    out = _build_exports_from_audit(audit, "Schedule built from the RFP.", [_wbs_call()])
    sched = [e for e in out if "schedule" in e["endpoint"]]
    assert len(sched) == 1
    assert sched[0]["endpoint"].endswith("/export/schedule-from-document")
    assert sched[0]["payload"]["document_ids"] == ["rfp1"]
    assert "from documents" in sched[0]["label"].lower()


def test_wbs_with_cited_boq_stays_brief_driven(monkeypatch):
    """A cited BOQ is a cost source, not a spec — the schedule stays brief-driven
    (no document_engine run on a spreadsheet), and a cost-BOQ offer appears."""
    audit = _audit_citing("boq1", 1)
    _stub_doc(monkeypatch, "the client project Priced BOQ.xlsx", doc_type="boq")
    out = _build_exports_from_audit(audit, "totals [source: the client project Priced BOQ.xlsx, chunk 1]", [_wbs_call()])
    sched = [e for e in out if "schedule" in e["endpoint"]]
    assert sched and sched[0]["endpoint"].endswith("/export/schedule-from-brief")
    assert "document_ids" not in sched[0]["payload"]
    assert any(e["endpoint"].endswith("/export/cost-boq") for e in out)
