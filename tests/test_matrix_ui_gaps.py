"""Close live theshovel.ai 17-action UI gaps (M3 / M14 / M16 / M17).

No LLM. Pins the four product holes the physical UI found:
  M14 RFP full paragraphs were wiped by the cost-grounding refusal
  M3  commissioning defaulted to HVAC instead of waterproofing
  M3  empty model turn after a successful commissioning tool
  M17 XER without TASKRSRC now returns activity-count period buckets
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _CG_REFUSAL,
    _EMPTY_RESPONSE_FALLBACK,
    _cost_grounding_gate,
    _forced_specific_tool,
    _format_commissioning_tool,
    _infer_commissioning_systems,
    _is_document_deliverable_request,
    _message_wants_resource_histogram,
    _nudge_for_failed_tool,
    _postprocess_answer,
    _predispatch_resource_histogram,
    _recover_answer_from_tool_messages,
    _resolve_histogram_schedule_file,
)
from tests.conftest import requires_construction_kit



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

def test_rfp_document_is_not_a_rate_quote():
    prompt = (
        "Prepare an RFP for leftover torch-applied waterproofing subcontract. "
        "Write the invitation letter, scope of works, prequalification "
        "criteria, evaluation method, and key dates in full paragraphs."
    )
    assert _is_document_deliverable_request(prompt)
    answer = (
        "REQUEST FOR PROPOSALS (RFP-WP-LEFT-2024)\n"
        "Invitation: submit a proposal for torch-applied SBS.\n"
        "Tender bond 50,000 SAR if required by the form of tender.\n"
    )
    out = _cost_grounding_gate(
        answer, None, messages=[{"role": "user", "content": prompt}]
    )
    assert out == answer
    assert out != _CG_REFUSAL


def test_unit_rate_question_still_refused():
    prompt = "What is the unit rate per m3 for C30 in the leftover BOQ?"
    assert not _is_document_deliverable_request(prompt)
    answer = "Use 465 SAR/m3 for C30."
    out = _cost_grounding_gate(
        answer, None, messages=[{"role": "user", "content": prompt}]
    )
    assert out == _CG_REFUSAL


def test_infer_waterproofing_commissioning_systems():
    assert _infer_commissioning_systems(
        "Generate a commissioning checklist for leftover torch-applied "
        "waterproofing before backfill."
    ) == ["waterproofing"]
    assert _infer_commissioning_systems("when does commissioning start?") is None


def test_infer_reservoir_wet_test_not_waterproofing():
    assert _infer_commissioning_systems(
        "Generate a commissioning checklist for the PWPS-09 reservoir "
        "at 1e-North Center before the first wet test. C21-OPC reservoir "
        "blinding, planned 350 m3, poured 310 m3."
    ) == ["reservoir"]


def test_empty_fallback_recovers_commissioning_tool():
    payload = {
        "status": "success",
        "action": "commissioning_checklist_generated",
        "summary": {"systems_covered": 1, "total_tests": 2, "pending": 2},
        "checklists_by_system": {
            "waterproofing": [
                {
                    "test": "Holiday / spark test",
                    "standard": "ASTM D4787",
                    "witness_required": True,
                    "hold_point": True,
                    "acceptance_criteria": "No holidays",
                }
            ]
        },
    }
    messages = [
        {"role": "user", "content": "commissioning checklist for waterproofing"},
        {"role": "tool", "content": json.dumps(payload)},
    ]
    recovered = _recover_answer_from_tool_messages(
        _EMPTY_RESPONSE_FALLBACK, messages
    )
    assert "Holiday / spark test" in recovered
    assert "unable to generate" not in recovered.lower()
    posted = _postprocess_answer(
        _EMPTY_RESPONSE_FALLBACK, None, messages, fallback_used=False
    )
    assert "Holiday / spark test" in posted


def test_format_commissioning_lists_hold_points():
    text = _format_commissioning_tool({
        "summary": {"systems_covered": 1, "total_tests": 1, "pending": 1},
        "checklists_by_system": {
            "waterproofing": [
                {
                    "test": "Backfill release",
                    "standard": "ITP hold point",
                    "witness_required": True,
                    "hold_point": True,
                    "acceptance_criteria": "Engineer sign-off",
                }
            ]
        },
    })
    assert "Backfill release" in text
    assert "hold" in text.lower()


def test_image_fail_nudge_does_not_retry_photo():
    class _A:
        can_delegate = False
        name = "safety-officer"

    nudge = _nudge_for_failed_tool(
        {"name": "image", "ok": False, "result": {"error": "empty pixels"}},
        _A(),
    )
    assert "Do not retry the image tool" in nudge
    assert "clash" not in nudge.lower()


@requires_construction_kit
@pytest.mark.asyncio
async def test_commissioning_waterproofing_not_hvac():
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    out = await c.commissioning_checklist(
        {}, {"systems": ["waterproofing"]}
    )
    assert out["status"] == "success"
    assert "waterproofing" in out["checklists_by_system"]
    tests = out["checklists_by_system"]["waterproofing"]
    names = {t["test"] for t in tests}
    assert "Backfill release" in names
    assert "Holiday / spark test" in names
    assert any(t.get("hold_point") for t in tests)


def test_histogram_plus_xer_forces_resource_histogram_not_parser():
    prompt = "produce a manpower histogram from resource_loaded.xer"
    assert _message_wants_resource_histogram(prompt)
    assert not _message_wants_resource_histogram("what is a resource histogram?")
    assert _forced_specific_tool(
        [{"role": "user", "content": prompt}],
        {"resource_histogram", "primavera_parser"},
    ) == "resource_histogram"
    assert _forced_specific_tool(
        [{"role": "user", "content": "parse the xer for milestones"}],
        {"resource_histogram", "primavera_parser"},
    ) == "primavera_parser"
    assert _forced_specific_tool(
        [{"role": "user", "content": "what is a resource histogram?"}],
        {"resource_histogram", "primavera_parser"},
    ) is None


def test_resolve_histogram_prefers_named_xer(tmp_path, monkeypatch):
    import os

    loaded = os.path.abspath("tests/fixtures/resource_loaded.xer")
    other = tmp_path / "baseline_programme.xer"
    other.write_bytes(b"ERMHDR\n")
    docs = [
        {"original_name": "baseline_programme.xer", "file_path": str(other)},
        {"original_name": "resource_loaded.xer", "file_path": loaded},
    ]
    import app.core.projects as projects
    monkeypatch.setattr(projects, "list_documents", lambda pid, **k: docs)

    fp, name = _resolve_histogram_schedule_file(
        "proj", "histogram from resource_loaded.xer",
    )
    assert name == "resource_loaded.xer"
    assert fp == loaded

    fp2, name2 = _resolve_histogram_schedule_file("proj", "build the histogram")
    assert fp2 is None
    assert name2 == ""


@requires_construction_kit
@pytest.mark.asyncio
async def test_predispatch_histogram_uses_named_xer(tmp_path, monkeypatch):
    import os

    loaded = os.path.abspath("tests/fixtures/resource_loaded.xer")
    other = tmp_path / "baseline_programme.xer"
    other.write_bytes(b"ERMHDR\n")
    docs = [
        {"original_name": "baseline_programme.xer", "file_path": str(other)},
        {"original_name": "resource_loaded.xer", "file_path": loaded},
    ]
    import app.core.projects as projects
    monkeypatch.setattr(projects, "list_documents", lambda pid, **k: docs)

    class _A:
        allowed_blocks = ["construction"]
        name = "construction-pm"

    messages = [{
        "role": "user",
        "content": "manpower histogram from resource_loaded.xer",
    }]
    out = await _predispatch_resource_histogram(_A(), messages, "proj")
    assert out is not None
    assert out["name"] == "resource_histogram"
    assert out["ok"] is True
    assert out["result"]["total_manhours"] == 700.0
    assert out["result"]["by_trade_totals"] == {"LAB": 600.0, "CARP": 100.0}
    injected = messages[-1]["content"]
    assert "Do not call primavera_parser" in injected
    assert "clash" not in injected.lower()


@pytest.mark.asyncio
async def test_predispatch_histogram_kill_switch(monkeypatch):
    monkeypatch.setenv("AGENT_HISTOGRAM_PREDISPATCH", "0")

    class _A:
        allowed_blocks = ["construction"]

    out = await _predispatch_resource_histogram(
        _A(),
        [{"role": "user", "content": "manpower histogram from resource_loaded.xer"}],
        "proj",
    )
    assert out is None
