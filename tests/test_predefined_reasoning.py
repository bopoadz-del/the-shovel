"""The orchestrator block's predefined reasoning for the schedule workflow.
Acceptance: a QUESTION answers inline with no file; a PRODUCE request
materializes the workbook. Intent decides the form, not 'a tool ran'."""
from __future__ import annotations

import asyncio
import os

from app.schemas.project_session import ProjectSession
from app.core.predefined_reasoning import (
    run_workflow, is_deliverable_request,
)


def _ctx(message, **kw):
    base = {"message": message, "project_name": "DC",
            "params": {"project_type": "data_center", "target_count": 60, "start_date": "2026-01-05"}}
    base.update(kw)
    return base


def test_deliverable_detection():
    assert is_deliverable_request("produce the schedule")
    assert is_deliverable_request("export the schedule to excel")
    assert not is_deliverable_request("how long is procurement on the schedule?")
    assert not is_deliverable_request("what is the critical path")


def test_question_answers_inline_no_offer():
    s = ProjectSession.new("q1")
    out = asyncio.run(run_workflow("generate_wbs", _ctx("how long is procurement on the schedule?"), s))
    assert out["handled"] and out["status"] == "success", out
    assert out["deliverable"] is False
    assert out["exports"] == []                 # no download offer for a question
    assert "working days" in out["answer"]
    assert "download" not in out["answer"].lower()


def test_produce_offers_the_render_endpoint():
    s = ProjectSession.new("d1")
    out = asyncio.run(run_workflow(
        "generate_wbs", _ctx("produce the schedule", project_id="proj1"), s))
    assert out["handled"] and out["status"] == "success", out
    assert out["deliverable"] is True
    assert len(out["exports"]) == 1
    exp = out["exports"][0]
    assert exp["endpoint"].endswith("/export/schedule-from-brief")   # no docs -> brief
    assert "download" in out["answer"].lower()


def test_unknown_action_falls_through_to_agent():
    s = ProjectSession.new("u1")
    out = asyncio.run(run_workflow("summarize_document", _ctx("summarize this"), s))
    assert out == {"handled": False}


def test_document_driven_injects_procurement_and_offers_doc_endpoint(monkeypatch):
    async def fake_feed(ids):
        return ([{"equipment": "Generator", "lead_time_days": 200}],
                [{"name": "RFS", "target_date": "2027-12-01"}])
    monkeypatch.setattr("app.lib.schedule_feed.extract_schedule_feed", fake_feed)
    s = ProjectSession.new("dd1")
    out = asyncio.run(run_workflow(
        "generate_wbs",
        _ctx("produce the schedule from the RFP", project_id="proj1", document_ids=["d1"]), s))
    assert "extract_document" in out["plan_steps"]
    assert "long-lead procurement" in out["answer"] and "RFS" in out["answer"]
    assert out["exports"][0]["endpoint"].endswith("/export/schedule-from-document")
    assert out["exports"][0]["payload"]["document_ids"] == ["d1"]
