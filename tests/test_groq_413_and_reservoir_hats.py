"""Live Infra Pack empties after #408: Groq 413, reservoir holiday/spark,
WIR steal on job requisition, and missing deliverable drafts.

No live LLM. Pins the machinery that used to leave M1/M3/M6/M7/M9/M12/M14
empty on theshovel.ai once Kimi filtered and Groq TPM-8000 413'd.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _EMPTY_RESPONSE_FALLBACK,
    _approx_message_chars,
    _compact_messages_for_tpm,
    _conflicting_tools_after_predispatch,
    _format_as_built_note,
    _format_cash_flow,
    _format_claim_notice,
    _format_commissioning_tool,
    _format_job_requisition,
    _format_om_outline,
    _format_payment_certificate,
    _format_rfp_draft,
    _format_rfi,
    _format_safety_briefing,
    _format_wbs_result,
    _format_wir_form,
    _graft_operator_claim_facts,
    _infer_commissioning_systems,
    _inject_predispatch,
    _message_wants_as_built_note,
    _message_wants_cash_flow,
    _message_wants_commissioning,
    _message_wants_delay_claim,
    _message_wants_design_directive,
    _message_wants_first_run_wbs,
    _message_wants_follow_on_rfi,
    _message_wants_ipc_draft,
    _message_wants_job_requisition,
    _message_wants_locked_deliverable,
    _message_wants_om_manual,
    _message_wants_qc_punch,
    _message_wants_rfp_draft,
    _message_wants_safety_briefing,
    _message_wants_wir_form,
    _messages_user_and_history,
    _operator_user_text,
    _predispatch_construction_draft,
    _predispatch_remaining_deliverables,
    _recover_answer_from_tool_messages,
    _text_needs_tool_recovery,
)
from tests.conftest import requires_construction_kit

M1 = (
    "From Annexure 2 - Baseline Program XER.xer and Contract Data, build a WBS "
    "for the ten access-gated community milestones. Time for Completion of the "
    "whole of the Works is 852 days. Show Milestone 1 (Southern Community 1a "
    "Media, 397 days) and Milestone 3 (Boulevard Community 1c Boulevard "
    "South-West, 487 days) as separate branches with durations in calendar days."
)
M3 = (
    "Generate a commissioning checklist for the PWPS-09 reservoir at "
    "1e-North Center before the first wet test. Ground it in Cast In-Situ "
    "Concretes works for PWPS-09.pdf and the Week 52 pour record: C21-OPC "
    "reservoir blinding, planned 350 m3, poured 310 m3 on 7 and 9 January 2025."
)
M6 = (
    "Prepare a job requisition for street-lighting installation. Scope is the "
    "AM Rev Design NOC for Street Lighting (expires 17 July 2025) plus the "
    "Grand Mosque Phase 2 VO items: solar 6 to 8 m single-arm 1x120W 230V "
    "60Hz poles, traffic signage, and road markings. Include prequalification "
    "and a shortlist rubric."
)
M7 = (
    "Draft Interim Payment Certificate No. 4 from Contract Data. Accepted "
    "Contract Amount SAR 1,754,504,456.25. Advance payment 10%, retention 10% "
    "of each IPC, amortisation 10% of each IPC. This period certified work is "
    "SAR 42,800,000. Show gross, advance recovery, retention, and net payable "
    "in SAR."
)
M9 = (
    "Draft a delay claim notice under Contract Data clause 8.8.1 for late "
    "access to Milestone 1 (Southern Community 1a Media). Access was given 28 "
    "calendar days after the Engineer instructed date. Milestone 1 delay "
    "damages are 0.015% of the Contract Price per calendar day. Use Aconex as "
    "the approved communication method."
)
M12 = (
    "Prepare an as-built deviation note for PWPS-09 reservoir blinding: "
    "planned 350 m3 versus poured 310 m3 on 7 and 9 January 2025. State the "
    "40 m3 shortfall and percent."
)
M14 = (
    "Prepare an RFP for a stormwater manhole-rationalisation subcontract: "
    "remove the 24-manhole radius cluster in RFI002 and install a formed GRP "
    "radiused or closed concrete channel. Write the invitation letter, scope "
    "of works, prequalification, evaluation method, and key dates in full "
    "paragraphs. Reference SOPR and UMA Stormwater DDC 20212200076."
)
M2 = (
    "Produce a six-month cash-flow S-curve starting at commencement. Use the "
    "Accepted Contract Amount SAR 1,754,504,456.25 from Contract Data. Apply "
    "the 10% advance payment (14.2.1) and 10% retention on each IPC (14.3.2). "
    "State monthly drawdown, cumulative, and net after advance recovery."
)
M13 = (
    "Generate an operations and maintenance manual outline for PWPS-09 "
    "(potable water pump station and reservoir). Cover existing-services "
    "interfaces from Specification Vol 2 Existing Services."
)
M16 = (
    "Produce a live-haul-road and public-interface safety briefing for the "
    "Green Village diversion: required signage, speed control, and pedestrian "
    "crossing controls."
)
M10 = (
    "Draft a follow-on RFI to RFI002 (24 stormwater manholes on one road "
    "bend). Ask the Engineer to confirm whether a custom-radius GRP or closed "
    "concrete channel is accepted, and how that change affects the Week 53 "
    "collar pours of 11 manholes totalling 28 m3 of C-35 SRC."
)
M4 = (
    "Issue a design directive answering RFI901 Underpass Length Conflict. "
    "The road profile is 42 m and the underpass is 36 m."
)
M8 = (
    "Produce a QC punch list and hold-point sign-off for the Week 53 "
    "stormwater manhole collars. Reference the WIR template."
)


def test_text_needs_tool_recovery_on_413_and_empty():
    assert _text_needs_tool_recovery("")
    assert _text_needs_tool_recovery(_EMPTY_RESPONSE_FALLBACK)
    assert _text_needs_tool_recovery(
        'groq HTTP 413: {"error":{"message":"Request too large ... TPM": Limit 8000'
    )
    assert not _text_needs_tool_recovery("Claim notice under 8.8.1")


def test_compact_messages_cuts_tool_dumps():
    huge = json.dumps({"activities": ["x" * 80] * 200})
    messages = [
        {"role": "system", "content": "you are a test"},
        {"role": "user", "content": M1},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "primavera_parser", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "name": "primavera_parser",
         "content": huge},
    ]
    assert len(huge) > 1200
    out = _compact_messages_for_tpm(messages, budget=2000)
    tool = next(m for m in out if m.get("role") == "tool")
    assert len(str(tool.get("content") or "")) < len(huge)
    assert "truncated" in str(tool.get("content") or "").lower() or len(
        str(tool.get("content") or "")
    ) <= 1200


def test_recover_413_from_commissioning_tool():
    payload = {
        "status": "success",
        "action": "commissioning_checklist_generated",
        "summary": {"systems_covered": 1, "total_tests": 2, "pending": 2},
        "checklists_by_system": {
            "reservoir": [
                {
                    "test": "First wet test / watertightness",
                    "standard": "BS 8007",
                    "witness_required": True,
                    "hold_point": True,
                    "acceptance_criteria": "No visible leakage",
                }
            ]
        },
    }
    recovered = _recover_answer_from_tool_messages(
        "groq HTTP 413: TPM Limit 8000",
        [
            {"role": "user", "content": M3},
            {"role": "tool", "content": json.dumps(payload)},
        ],
    )
    assert "First wet test" in recovered
    assert "holiday" not in recovered.lower()
    assert "spark" not in recovered.lower()


def test_message_wants_job_req_and_first_run_wbs():
    assert _message_wants_job_requisition(M6)
    assert not _message_wants_job_requisition(M3)
    assert _message_wants_first_run_wbs(M1)
    assert not _message_wants_first_run_wbs("what is a WBS?")


def test_message_wants_remaining_deliverables():
    assert _message_wants_commissioning(M3)
    assert not _message_wants_commissioning(M6)
    assert _message_wants_ipc_draft(M7)
    assert not _message_wants_ipc_draft("issue a payment certificate")
    assert _message_wants_cash_flow(M2)
    assert not _message_wants_ipc_draft(M2)
    assert _message_wants_delay_claim(M9)
    assert _message_wants_as_built_note(M12)
    assert _message_wants_rfp_draft(M14)
    assert not _message_wants_rfp_draft(
        "Prepare a WIR for the RFP manhole collar pour"
    )


def test_first_run_wbs_skips_duration_rerun():
    rerun = "build a WBS. use 12 days for excavation and re-run"
    assert _message_wants_first_run_wbs(M1)
    assert not _message_wants_first_run_wbs(rerun)


def test_first_run_wbs_swallows_override_import_miss(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("shape miss")

    monkeypatch.setattr(
        "app.lib.wbs_duration_overrides.message_wants_wbs_duration_rerun",
        _boom,
    )
    assert _message_wants_first_run_wbs(M1) is True


def test_infer_commissioning_mixed_wet_test_stays_reservoir():
    assert _infer_commissioning_systems(M3) == ["reservoir"]
    assert _infer_commissioning_systems(
        "torch-applied SBS membrane holiday test before backfill"
    ) == ["waterproofing"]
    assert _infer_commissioning_systems(
        "PWPS-09 reservoir first wet test after membrane protection"
    ) == ["reservoir"]
    assert _infer_commissioning_systems(
        "reservoir membrane protection board only"
    ) == ["waterproofing"]
    assert _infer_commissioning_systems("ordinary concrete cubes") is None


def test_formatters_and_recovery_cover_each_deliverable():
    ipc = {
        "action": "payment_certificate",
        "certificate": {"period": "IPC 4", "contractor": "Al-Example"},
        "valuation": {
            "contract_value": 1_754_504_456.25,
            "gross_valuation": 42_800_000,
        },
        "deductions": {
            "retention_percent": 10,
            "retention_held": 4_280_000,
            "advance_recovery": 4_280_000,
        },
        "payment": {
            "net_due_this_period": 34_240_000,
            "cumulative_certified": 34_240_000,
        },
        "certificate_summary": "IPC 4 draft",
    }
    assert "42,800,000" in _format_payment_certificate(ipc) or "42800000" in (
        _format_payment_certificate(ipc).replace(",", "")
    )

    note = {
        "action": "as_built_deviation_report",
        "planned_m3": 350,
        "poured_m3": 310,
        "shortfall_m3": 40,
        "shortfall_percent": 11.43,
        "location": "PWPS-09",
    }
    assert "40" in _format_as_built_note(note)
    assert "11.43" in _format_as_built_note(note)
    assert "stated" in _format_as_built_note({"note": "stated shortfall"})

    claim = {
        "action": "claim_generated",
        "claim_narrative": {"full_narrative": "DELAY CLAIM NOTICE — Clause 8.8.1"},
        "quantum_formula": "0.015% × Contract Price × 28",
        "communication": "Aconex",
    }
    rendered_claim = _format_claim_notice(claim)
    assert "8.8.1" in rendered_claim
    assert "Aconex" in rendered_claim
    dumped = _format_claim_notice({"status": "success"})
    assert "success" in dumped

    wbs = {
        "action": "generate_wbs",
        "wbs_id": "w1",
        "project_type": "infrastructure",
        "actual_count": 20,
        "operator_milestones": {
            "time_for_completion_days": 852,
            "milestones": [
                {"id": 1, "name": "Media", "days": 397},
                {"id": 3, "name": "Boulevard SW", "days": 487},
            ],
        },
        "summary": {"activity_count": 20},
        "assumptions": ["Operator-stated milestone durations"],
    }
    rendered_wbs = _format_wbs_result(wbs)
    assert "852" in rendered_wbs and "397" in rendered_wbs
    fallback_wbs = _format_wbs_result({
        "target_milestones": [{"name": "Earthworks", "duration_days": 30}],
        "summary": {},
    })
    assert "Earthworks" in fallback_wbs

    jr = {
        "action": "job_requisition",
        "jr_number": "DRAFT-JR",
        "title": "Street lighting",
        "scope": "Street-lighting installation",
        "noc": "AM Rev Design NOC",
        "noc_expiry": "17 July 2025",
        "prequalification": ["HSE plan"],
        "shortlist_rubric": [
            {"criterion": "Technical", "weight": "40%"},
            "commercial",
        ],
        "note": "Draft JR",
    }
    rendered_jr = _format_job_requisition(jr)
    assert "17 July 2025" in rendered_jr
    assert "Technical" in rendered_jr

    rfp = {
        "action": "rfp_draft",
        "title": "RFP",
        "rfp_number": "DRAFT-RFP",
        "invitation": "INVITATION TO TENDER",
        "scope_of_works": "GRP radiused section",
        "prequalification": ["ITP"],
        "evaluation_method": "technical",
        "key_dates_note": "TBD",
        "references": ["RFI002", "SOPR"],
    }
    rendered_rfp = _format_rfp_draft(rfp)
    assert "INVITATION" in rendered_rfp
    assert "RFI002" in rendered_rfp

    wir = _format_wir_form({
        "wir_number": "DRAFT-WIR",
        "location": "H-1",
        "activity": "collars",
        "week": 53,
        "mix": "C-35",
        "volume_m3": 28,
        "manhole_range": "W-1–W-7",
        "supplier": "XBCC",
        "template": "ITP",
        "checklist": [{"item": "formwork"}, "cubes"],
        "hold_points": ["pre-pour"],
        "witness_points": ["cubes"],
        "signatories": [{"party": "Engineer", "status": "pending"}],
        "notice": "draft",
        "note": "operator facts",
    })
    assert "DRAFT-WIR" in wir and "Week 53" in wir

    comm = _format_commissioning_tool({
        "summary": {"systems_covered": 1, "total_tests": 1, "pending": 1},
        "checklists_by_system": {
            "reservoir": [
                {"test": "First wet test", "standard": "BS 8007",
                 "acceptance_criteria": "No leak", "witness_required": True,
                 "hold_point": True},
                "skip-me",
            ],
            "other": "not-a-list",
        },
    })
    assert "First wet test" in comm
    assert _format_commissioning_tool({"checklists_by_system": "x"}).startswith(
        "Commissioning checklist"
    )

    empty_recover = _recover_answer_from_tool_messages(
        "unable to generate a response",
        [{"role": "tool", "content": json.dumps(ipc)}],
    )
    assert "IPC 4" in empty_recover
    assert "Aconex" in _recover_answer_from_tool_messages(
        "HTTP 413 TPM", [{"role": "tool", "content": json.dumps(claim)}]
    )
    assert "DRAFT-JR" in _recover_answer_from_tool_messages(
        "", [{"role": "tool", "content": json.dumps(jr)}]
    )
    assert "INVITATION" in _recover_answer_from_tool_messages(
        _EMPTY_RESPONSE_FALLBACK, [{"role": "tool", "content": json.dumps(rfp)}]
    )
    assert "852" in _recover_answer_from_tool_messages(
        "groq HTTP 413", [{"role": "tool", "content": json.dumps(wbs)}]
    )
    assert "40" in _recover_answer_from_tool_messages(
        "TPM", [{"role": "tool", "content": json.dumps(note)}]
    )
    hist = _recover_answer_from_tool_messages(
        "HTTP 413",
        [{
            "role": "tool",
            "content": json.dumps({
                "action": "resource_histogram_generated",
                "histogram_kind": "activity_count",
                "note": "first 16 weeks",
                "periods": [{"label": "W1", "total": 12}, "skip"],
            }),
        }],
    )
    assert "W1" in hist
    truncated = _recover_answer_from_tool_messages(
        "HTTP 413",
        [{
            "role": "tool",
            "content": json.dumps({
                "truncated": True,
                "preview": json.dumps(ipc),
            }),
        }],
    )
    assert "IPC 4" in truncated
    msgs = []
    _inject_predispatch(msgs, "claims_builder", rendered_claim, "Present this.")
    assert "PLATFORM PRE-DISPATCH" in msgs[0]["content"]
    from_pre = _recover_answer_from_tool_messages(
        "HTTP 413", msgs,
    )
    assert "8.8.1" in from_pre


def test_compact_messages_truncates_roles_and_drops_middle():
    assert _approx_message_chars([{"role": "user", "content": "ab"}, "x"]) == 2
    messages = [
        {"role": "system", "content": "S" * 4000},
        *[{"role": "user", "content": f"mid-{i}"} for i in range(20)],
        {"role": "user", "content": "U" * 7000},
        "not-a-dict",
        {"role": "assistant", "content": ["not-a-string"]},
        {"role": "tool", "content": "T" * 2000, "tool_call_id": "1"},
        {"role": "user", "content": "tail"},
    ]
    out = _compact_messages_for_tpm(messages, budget=500)
    assert any(
        isinstance(m, dict) and m.get("role") == "system"
        and "[truncated for TPM]" in str(m.get("content"))
        for m in out
    )
    assert any(
        isinstance(m, dict) and m.get("role") == "user"
        and "[truncated for TPM]" in str(m.get("content"))
        for m in out
    )
    tool = next(
        m for m in out
        if isinstance(m, dict) and m.get("role") == "tool"
    )
    assert "truncated" in str(tool.get("content")).lower()
    assert "not-a-dict" in out
    assert any(
        isinstance(m, dict) and m.get("content") == "tail" for m in out
    )


@pytest.mark.asyncio
async def test_predispatch_skips_wir_killswitch_and_non_construction():
    class _A:
        allowed_blocks = ["search"]
        name = "contracts-manager"

    wir_msgs = [{"role": "user", "content": "Prepare a WIR for Week 53 collars"}]
    assert await _predispatch_remaining_deliverables(_A(), wir_msgs, "p") is None

    class _C:
        allowed_blocks = ["construction"]
        name = "contracts-manager"

    no_match = [{"role": "user", "content": "hello there"}]
    assert await _predispatch_remaining_deliverables(_C(), no_match, "p") is None

    off = await _predispatch_construction_draft(
        _C(), [{"role": "user", "content": M9}],
        env_key="AGENT_CLAIM_PREDISPATCH",
        want_fn=_message_wants_delay_claim,
        action="claims_builder",
        format_fn=_format_claim_notice,
        instruction="x",
    )
    # kill-switch is read from env; default on. Force off:
    import os
    old = os.environ.get("AGENT_CLAIM_PREDISPATCH")
    os.environ["AGENT_CLAIM_PREDISPATCH"] = "0"
    try:
        off = await _predispatch_construction_draft(
            _C(), [{"role": "user", "content": M9}],
            env_key="AGENT_CLAIM_PREDISPATCH",
            want_fn=_message_wants_delay_claim,
            action="claims_builder",
            format_fn=_format_claim_notice,
            instruction="x",
        )
        assert off is None
    finally:
        if old is None:
            os.environ.pop("AGENT_CLAIM_PREDISPATCH", None)
        else:
            os.environ["AGENT_CLAIM_PREDISPATCH"] = old

    assert await _predispatch_construction_draft(
        _A(), [{"role": "user", "content": M9}],
        env_key="AGENT_CLAIM_PREDISPATCH",
        want_fn=_message_wants_delay_claim,
        action="claims_builder",
        format_fn=_format_claim_notice,
        instruction="x",
    ) is None


def test_as_built_and_claim_helpers_parse_operator_facts():
    from app.containers.construction.documents import (
        _as_built_volume_facts_from_text,
    )
    from app.containers.construction.schedule import (
        _delay_claim_facts_from_text,
        _draft_delay_claim_notice,
        _operator_milestones_from_text,
    )

    vs = _as_built_volume_facts_from_text(
        "PWPS-09 blinding 350 m3 versus 310 m3 on 7 January 2025"
    )
    assert vs["planned_m3"] == 350
    assert vs["poured_m3"] == 310
    assert vs["shortfall_m3"] == 40
    assert vs["shortfall_percent"] == 11.43
    assert vs["location"].startswith("PWPS")
    assert _as_built_volume_facts_from_text("no volumes here") is None

    facts = _delay_claim_facts_from_text(M9)
    assert facts["delay_days"] == 28
    assert facts["rate_percent_per_day"] == 0.015
    assert facts["clause"] == "8.8.1"
    assert "Milestone 1" in facts["milestone"]
    assert facts["communication"] == "Aconex"
    assert _delay_claim_facts_from_text("no delay days") is None

    notice = _draft_delay_claim_notice(facts, "2026-08-23")
    assert notice["status"] == "success"
    assert notice["delay_days"] == 28
    assert "Aconex" in notice["claim_narrative"]["full_narrative"]
    assert "0.015" in (notice.get("quantum_formula") or "")

    op = _operator_milestones_from_text(M1)
    assert op["time_for_completion_days"] == 852
    ids = {m["id"]: m["days"] for m in op["milestones"]}
    assert ids[1] == 397
    assert ids[3] == 487


@pytest.mark.asyncio
async def test_reservoir_checklist_has_no_holiday_spark():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().commissioning_checklist(
        {}, {"systems": ["reservoir"]}
    )
    assert out["status"] == "success"
    tests = out["checklists_by_system"]["reservoir"]
    names = {t["test"] for t in tests}
    assert "First wet test / watertightness" in names
    assert "Holiday / spark test" not in names
    blob = json.dumps(out).lower()
    assert "holiday" not in blob
    assert "spark" not in blob
    assert "d4787" not in blob


@pytest.mark.asyncio
async def test_waterproofing_still_has_holiday_spark():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().commissioning_checklist(
        {}, {"systems": ["waterproofing"]}
    )
    names = {t["test"] for t in out["checklists_by_system"]["waterproofing"]}
    assert "Holiday / spark test" in names


@pytest.mark.asyncio
async def test_job_requisition_drafts_m6():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().job_requisition({"text": M6}, {})
    assert out["status"] == "success"
    assert out["action"] == "job_requisition"
    assert "Street-lighting" in out["scope"] or "street" in out["scope"].lower()
    assert out["prequalification"]
    assert out["shortlist_rubric"]
    rendered = _format_job_requisition(out)
    assert "17 July 2025" in rendered or "NOC" in rendered


@pytest.mark.asyncio
async def test_rfp_drafts_m14():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().rfp_draft({"text": M14}, {})
    assert out["status"] == "success"
    assert "GRP" in out["scope_of_works"] or "channel" in out["scope_of_works"].lower()
    rendered = _format_rfp_draft(out)
    assert "invitation" in rendered.lower()
    assert "RFI002" in rendered or "SOPR" in rendered


@pytest.mark.asyncio
async def test_as_built_volume_note_m12():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().as_built_deviation_report(
        {"text": M12}, {},
    )
    assert out["status"] == "success"
    assert out["planned_m3"] == 350
    assert out["poured_m3"] == 310
    assert out["shortfall_m3"] == 40
    assert out["shortfall_percent"] == 11.43
    assert "40" in _format_as_built_note(out)


@pytest.mark.asyncio
async def test_delay_claim_notice_m9():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().claims_builder({"text": M9}, {})
    assert out["status"] == "success"
    assert out["delay_days"] == 28
    assert out["rate_percent_per_day"] == 0.015
    assert out["clause"] == "8.8.1"
    assert "Aconex" in (out.get("communication") or "")
    rendered = _format_claim_notice(out)
    assert "28" in rendered
    assert "0.015" in rendered
    assert "Aconex" in rendered


@pytest.mark.asyncio
async def test_ipc_parses_accepted_contract_amount_m7():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().payment_certificate(
        {"message": M7}, {},
    )
    assert out["status"] == "success"
    val = out["valuation"]
    assert val["contract_value"] == 1_754_504_456.25
    assert val["gross_valuation"] == 42_800_000
    assert out["deductions"]["retention_percent"] == 10
    assert out["deductions"]["advance_recovery"] == 4_280_000
    assert out["payment"]["net_due_this_period"] == 34_240_000
    rendered = _format_payment_certificate(out)
    assert "42800000" in rendered.replace(",", "") or "42,800,000" in rendered


@pytest.mark.asyncio
async def test_wbs_keeps_operator_milestones_m1():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().generate_wbs(
        {}, {"brief": M1, "target_count": 20},
    )
    assert out["status"] == "success"
    assert out["project_type"] == "infrastructure"
    op = out.get("operator_milestones") or {}
    assert op.get("time_for_completion_days") == 852
    ids = {m["id"]: m["days"] for m in op.get("milestones") or []}
    assert ids[1] == 397
    assert ids[3] == 487
    rendered = _format_wbs_result(out)
    assert "852" in rendered
    assert "397" in rendered
    assert "487" in rendered


@requires_construction_kit
@pytest.mark.asyncio
async def test_predispatch_claim_and_ipc(monkeypatch):
    class _A:
        allowed_blocks = ["construction"]
        name = "contracts-manager"

    msgs = [{"role": "user", "content": M9}]
    out = await _predispatch_remaining_deliverables(_A(), msgs, "proj")
    assert out is not None
    assert out["name"] == "claims_builder"
    assert "8.8.1" in msgs[-1]["content"] or "28" in msgs[-1]["content"]

    msgs = [{"role": "user", "content": M7}]
    out = await _predispatch_remaining_deliverables(_A(), msgs, "proj")
    assert out is not None
    assert out["name"] == "payment_certificate"


@pytest.mark.asyncio
async def test_infer_then_container_reservoir_for_m3():
    assert _infer_commissioning_systems(M3) == ["reservoir"]
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().commissioning_checklist(
        {}, {"systems": _infer_commissioning_systems(M3)}
    )
    blob = json.dumps(out).lower()
    assert "holiday" not in blob
    assert "spark" not in blob


def test_messages_user_skips_platform_predispatch():
    user, _hist = _messages_user_and_history([
        {"role": "user", "content": M1},
        {"role": "user", "content": "PLATFORM PRE-DISPATCH: drawing_qto ran on a PDF"},
    ])
    assert "build a WBS" in user
    assert "PLATFORM PRE-DISPATCH" not in user
    assert _message_wants_first_run_wbs(user)
    assert _message_wants_locked_deliverable(user)


def test_wir_yields_to_job_req_and_remaining_intents():
    steal = M6 + " Also write an inspection request for the poles."
    assert _message_wants_job_requisition(steal)
    assert not _message_wants_wir_form(steal)
    assert _message_wants_cash_flow(M2)
    assert _message_wants_om_manual(M13)
    assert _message_wants_safety_briefing(M16)
    assert not _message_wants_safety_briefing(M1)
    assert _message_wants_follow_on_rfi(M10)
    assert not _message_wants_wir_form(M10)
    assert _message_wants_locked_deliverable(M10)
    assert _message_wants_design_directive(M4)
    assert not _message_wants_safety_briefing(M4)
    assert _message_wants_qc_punch(M8)
    assert not _message_wants_wir_form(M8)
    assert _message_wants_locked_deliverable(M8)
    assert not _message_wants_safety_briefing(M6)
    assert not _message_wants_commissioning(M13)
    assert _message_wants_locked_deliverable(M2)
    assert _message_wants_locked_deliverable(M13)
    assert _message_wants_locked_deliverable(M16)
    assert not _message_wants_locked_deliverable(
        "Complete a Work Inspection Request for Week 53 collars"
    )


def test_conflicting_tools_lock_steal_surfaces():
    assert "payment_certificate" in _conflicting_tools_after_predispatch(
        "generate_wbs"
    )
    assert "drawing_qto" in _conflicting_tools_after_predispatch(
        "cash_flow_forecast"
    )
    assert "wir_form" in _conflicting_tools_after_predispatch("rfp_draft")
    assert "wir_form" in _conflicting_tools_after_predispatch("rfi_generator")
    assert "safety_briefing" in _conflicting_tools_after_predispatch(
        "generate_wbs"
    )
    assert "safety_briefing" in _conflicting_tools_after_predispatch(
        "resource_histogram"
    )
    assert "commissioning_checklist" in _conflicting_tools_after_predispatch(
        "om_manual_generator"
    )
    assert _conflicting_tools_after_predispatch("unknown") == set()


def test_format_cash_om_safety_and_recover():
    cash = {
        "action": "cash_flow_forecast",
        "project_parameters": {"contract_value": 1_754_504_456.25},
        "monthly_forecast": [
            {"month": 1, "planned_progress_percent": 5,
             "monthly_value": 1, "cumulative_value": 1,
             "advance_recovery": 0, "retention_deduction": 0,
             "net_cash_in": 1},
        ],
    }
    rendered_cash = _format_cash_flow(cash)
    assert "1,754,504,456.25" in rendered_cash or "1754504456.25" in (
        rendered_cash.replace(",", "")
    )
    om = {
        "action": "om_manual_generated",
        "title": "Operations and Maintenance manual outline — PWPS-09",
        "sections": [{"section": "1. Purpose and scope", "content": "O&M"}],
        "note": "Outline from the operator brief",
    }
    assert "PWPS-09" in _format_om_outline(om)
    safety = {
        "action": "safety_briefing",
        "briefing": "Signage, speed control, and pedestrian crossing.",
    }
    assert "pedestrian" in _format_safety_briefing(safety).lower()
    recovered_cash = _recover_answer_from_tool_messages(
        "HTTP 413", [{"role": "tool", "content": json.dumps(cash)}]
    )
    assert "1,754,504,456.25" in recovered_cash or "1754504456.25" in (
        recovered_cash.replace(",", "")
    )
    assert "PWPS-09" in _recover_answer_from_tool_messages(
        "", [{"role": "tool", "content": json.dumps(om)}]
    )
    assert "Signage" in _recover_answer_from_tool_messages(
        _EMPTY_RESPONSE_FALLBACK, [{"role": "tool", "content": json.dumps(safety)}]
    )


@pytest.mark.asyncio
async def test_om_outline_and_safety_briefing_from_operator_text():
    from app.containers.construction import ConstructionContainer

    om = await ConstructionContainer().om_manual_generator({"text": M13}, {})
    assert om["status"] == "success"
    assert len(om.get("sections") or []) == 13
    assert "PWPS" in (om.get("title") or "") or om.get("location")
    blob = json.dumps(om).lower()
    assert "holiday" not in blob
    assert "spark" not in blob

    safety = await ConstructionContainer().safety_briefing({"text": M16}, {})
    assert safety["status"] == "success"
    blob = (safety.get("briefing") or "").lower()
    assert "signage" in blob
    assert "speed" in blob
    assert "pedestrian" in blob


@pytest.mark.asyncio
async def test_wir_refuses_rfp_when_model_rewrote_scope():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().wir_form(
        {"text": "Blinding concrete pour", "scope": "Week 53 collars"},
        {"user_message": M14},
    )
    assert out["status"] == "error"
    assert "non-WIR" in (out.get("error") or "") or "RFP" in (out.get("error") or "")


@pytest.mark.asyncio
async def test_cash_flow_parses_six_month_word_and_aca():
    from app.containers.construction.boq import _cashflow_figures_from_message
    from app.containers.construction import ConstructionContainer

    fig = _cashflow_figures_from_message(M2)
    assert fig.get("contract_value") == 1_754_504_456.25
    assert fig.get("duration_months") == 6

    out = await ConstructionContainer().cash_flow_forecast(
        {"message": M2, "text": M2, "user_message": M2},
        {"user_message": M2, "brief": M2},
    )
    assert out["status"] == "success"
    params = out.get("project_parameters") or {}
    assert params.get("contract_value") == 1_754_504_456.25
    months = out.get("monthly_forecast") or out.get("s_curve_data") or []
    assert len(months) == 6


def test_graft_keeps_aconex_and_clause_on_rewritten_claim():
    rewritten = (
        "**DELAY CLAIM NOTICE**\n*Clause 4.5 – Notice of Delay*\n"
        "Access was given 28 calendar days late. Rate 0.015%."
    )
    grafted = _graft_operator_claim_facts(rewritten, M9)
    assert "Aconex" in grafted
    assert "8.8.1" in grafted
    polluted = [
        {"role": "user", "content": M9},
        {"role": "user", "content": "haul-road safety briefing signage"},
        {"role": "user", "content": "PLATFORM PRE-DISPATCH: claims_builder ran"},
    ]
    blob = _operator_user_text(polluted)
    assert "Aconex" in blob
    assert "8.8.1" in blob
    last, _ = _messages_user_and_history(polluted)
    assert "Aconex" not in last
    assert "Aconex" in _graft_operator_claim_facts(rewritten, blob)


def test_format_rfi_tolerates_non_dict_rows():
    rendered = _format_rfi({
        "rfis": ["not-a-dict", {"rfi_number": "DRAFT-RFI", "question": "GRP?"}],
    })
    assert "DRAFT-RFI" in rendered
    assert "GRP?" in rendered


@pytest.mark.asyncio
async def test_wir_refuses_follow_on_rfi_when_model_said_inspection():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().wir_form(
        {"text": "inspection request for Week 53 collars", "scope": "pour"},
        {"user_message": M10},
    )
    assert out["status"] == "error"
    err = (out.get("error") or "").lower()
    assert "non-wir" in err or "rfi" in err

    numbered = await ConstructionContainer().wir_form(
        {"text": "inspection request for Week 53 collars", "scope": "pour"},
        {"user_message": "Raise an inspection request for the RFI002 manhole cluster"},
    )
    assert numbered["status"] == "error"


@pytest.mark.asyncio
async def test_rfi_generator_drafts_follow_on_from_operator_text():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().rfi_generator(
        {"text": M10, "user_message": M10},
        {"user_message": M10},
    )
    assert out["status"] == "success"
    assert out["rfis"]
    rendered = _format_rfi(out)
    assert "RFI002" in rendered or "follow-on" in rendered.lower()
    assert "GRP" in rendered or "channel" in rendered.lower()
    assert "28" in rendered


@requires_construction_kit
@pytest.mark.asyncio
async def test_remaining_uses_operator_text_not_polluted_last_bubble():
    class _A:
        allowed_blocks = ["construction"]
        name = "construction-pm"

    polluted = [
        {"role": "user", "content": M1},
        {
            "role": "user",
            "content": (
                "Access and Haul Road Site Modifications — live haul-road "
                "and road diversion safety briefing with signage and pedestrians"
            ),
        },
    ]
    out = await _predispatch_remaining_deliverables(
        _A(), polluted, "proj", operator_text=M1,
    )
    assert out is not None
    assert out["name"] == "generate_wbs"


@requires_construction_kit
@pytest.mark.asyncio
async def test_remaining_drafts_follow_on_rfi_not_wir():
    class _A:
        allowed_blocks = ["construction"]
        name = "construction-pm"

    msgs = [{"role": "user", "content": M10}]
    out = await _predispatch_remaining_deliverables(_A(), msgs, "proj")
    assert out is not None
    assert out["name"] == "rfi_generator"


@requires_construction_kit
@pytest.mark.asyncio
async def test_remaining_commissioning_uses_reservoir_systems():
    class _A:
        allowed_blocks = ["construction"]
        name = "construction-pm"

    msgs = [{"role": "user", "content": M3}]
    out = await _predispatch_remaining_deliverables(_A(), msgs, "proj")
    assert out is not None
    assert out["name"] == "commissioning_checklist"
    by_sys = (out.get("result") or {}).get("checklists_by_system") or {}
    assert "reservoir" in by_sys
    blob = json.dumps(out["result"]).lower()
    assert "wet" in blob or "watertight" in blob
    assert "holiday" not in blob
    assert "spark" not in blob


@requires_construction_kit
@pytest.mark.asyncio
async def test_remaining_job_req_beats_polluted_safety_bubble():
    class _A:
        allowed_blocks = ["construction"]
        name = "contracts-manager"

    polluted = [
        {"role": "user", "content": M6},
        {
            "role": "user",
            "content": "haul-road safety briefing signage speed pedestrian",
        },
    ]
    out = await _predispatch_remaining_deliverables(
        _A(), polluted, "proj", operator_text=M6,
    )
    assert out is not None
    assert out["name"] == "job_requisition"


@pytest.mark.asyncio
async def test_payment_certificate_operator_aca_beats_stored_fact():
    from app.containers.construction import ConstructionContainer

    out = await ConstructionContainer().payment_certificate(
        {"message": M7, "contract_value": 1463.0},
        {"contract_value": 1463.0, "user_message": M7},
    )
    assert out["status"] == "success"
    val = (out.get("valuation") or {}).get("contract_value")
    assert val == 1_754_504_456.25


# ── M16: the site cue must not require the word "road" ─────────────────────
# The M16 constant above carries THREE independent triggers — "safety
# briefing", "live-haul-road" and "public-interface" — so it matches however
# narrow the site cue becomes, and cannot detect a regression in it. A live
# safety-officer turn summarised as "Green Village diversion briefing from the
# three named files (signage, speed, pedestrian)" has none of those three, and
# safety_briefing never ran: only fetch_document did, and the hat answered
# "I'm ready to help, but I need a specific question".
#
# This phrasing is deliberately the HARD one: a diversion named after its
# LOCATION, no "road", no literal "safety briefing".
M16_LOCATION_ONLY = (
    "Prepare the Green Village diversion briefing from the three named files, "
    "covering signage, speed limits and pedestrian routes."
)


def test_m16_constant_is_too_easy_to_catch_a_narrow_site_cue():
    """Guard the guard: if this ever stops being true the assertion below is
    the only thing standing between a narrowed cue and a silent live miss."""
    triggers = ["safety briefing", "haul-road", "public-interface"]
    present = [t for t in triggers if t in M16.lower()]
    assert len(present) >= 2, (
        f"M16 was meant to carry redundant triggers; found {present}"
    )


def test_location_named_diversion_still_wants_a_safety_briefing():
    assert _message_wants_safety_briefing(M16_LOCATION_ONLY)
    assert _message_wants_locked_deliverable(M16_LOCATION_ONLY)


def test_briefing_needs_real_breadth_not_one_stray_topic():
    """The briefing-only route requires two of the three public-interface
    topics. One topic plus the bare word "briefing" is too thin to claim a
    turn away from every other deliverable."""
    assert not _message_wants_safety_briefing(
        "Briefing on the speed of the works this week."
    )
    assert _message_wants_safety_briefing(
        "Briefing for the crew on signage and pedestrian routes."
    )


@pytest.mark.parametrize("other", [M1, M2, M4, M6, M8, M10, M13])
def test_broadened_safety_cue_steals_nothing(other):
    """The whole point of #410/#411 was tools claiming turns that belong to
    another deliverable. Widening this predicate must not re-open that."""
    assert not _message_wants_safety_briefing(other)


def test_safety_still_yields_to_an_rfp_ask():
    rfp = (
        "Draft an RFP for the 24-manhole rationalisation — invitation to "
        "tender, scope of works, prequalification criteria."
    )
    assert _message_wants_rfp_draft(rfp)
    assert not _message_wants_safety_briefing(rfp)
