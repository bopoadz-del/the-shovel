"""Live M9 / M12 / M15: empty hats and incomplete WIR generation.

No live LLM. Pins the machinery that used to leave contracts-manager empty,
construction-pm content-filtered, and a WIR search-loop on theshovel.ai.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents.runtime import (
    _EMPTY_RESPONSE_FALLBACK,
    _final_text_needs_forced_retry,
    _format_wir_form,
    _is_document_deliverable_request,
    _llm_choice_is_empty_or_filtered,
    _looks_like_search_preamble,
    _message_wants_wir_form,
    _predispatch_wir_form,
    _recover_answer_from_tool_messages,
)
from tests.conftest import requires_construction_kit

M15 = (
    "Complete a Work Inspection Request on "
    "SW-QRS-990-0000-EXD-TEM-QL-000002(00) WIR_Work_Insp_Req.docx "
    "for Week 53 Boulevard North-West stormwater concrete collars: "
    "11 manholes MH-3-2 to MH-3-12, C-35 SRC, 28 m3, supplier XBCC. "
    "List hold points before pour and witness points for slump, cubes, and cover."
)


def _run(coro):
    return asyncio.run(coro)


@requires_construction_kit
@pytest.mark.asyncio
async def test_wir_form_fills_m15_pour_facts():
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().wir_form({"text": M15}, {})
    assert result["status"] == "success"
    assert result["action"] == "wir_form"
    assert result["issued"] is False
    assert result["wir_number"] == "DRAFT-WIR"
    assert result["week"] == "53"
    assert result["volume_m3"] == "28"
    assert "C-35" in result["mix"]
    assert "SRC" in result["mix"]
    assert result["manhole_count"] == 11
    assert "MH-3-2" in (result.get("manhole_range") or "")
    assert result["supplier"] == "XBCC"
    assert "WIR_Work_Insp_Req.docx" in (result.get("template") or "")
    holds = " ".join(result["hold_points"]).lower()
    assert "hold" in holds or "notice" in holds
    witness = " ".join(result["witness_points"]).lower()
    assert "slump" in witness
    assert "cube" in witness
    assert "cover" in witness
    rendered = _format_wir_form(result)
    assert "28 m³" in rendered or "28 m3" in rendered
    assert "Week 53" in rendered


@requires_construction_kit
@pytest.mark.asyncio
async def test_inspection_request_routes_to_wir_form_not_photo_qa():
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().route(
        "inspection_request",
        {"text": "prepare an inspection request for the blinding concrete pour at zone B"},
        {"action": "inspection_request"},
    )
    assert result["status"] == "success"
    assert result.get("action") == "wir_form"
    assert result.get("error") != "No inspection image provided"
    assert result["hold_points"]
    assert "zone b" in (result.get("location") or "").lower()


@requires_construction_kit
@pytest.mark.asyncio
async def test_empty_inspection_request_still_drafts_skeleton():
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().route(
        "inspection_request", {}, {"action": "inspection_request"}
    )
    assert result["status"] == "success"
    assert result.get("action") == "wir_form"
    assert result["wir_number"] == "DRAFT-WIR"
    assert result["issued"] is False


def test_message_wants_wir_form_m15_not_definition():
    assert _message_wants_wir_form(M15)
    assert _message_wants_wir_form(
        "prepare an inspection request for the blinding pour"
    )
    assert not _message_wants_wir_form("what is a WIR?")
    assert not _message_wants_wir_form("cash-flow S-curve for six months")


M10 = (
    "Draft a follow-on RFI to RFI002 (24 stormwater manholes on one road bend). "
    "Ask the Engineer to confirm whether a custom-radius GRP or closed concrete "
    "channel is accepted, and how that change affects the Week 53 collar pours "
    "of 11 manholes totalling 28 m3 of C-35 SRC."
)
M14 = (
    "Prepare an RFP for a stormwater manhole-rationalisation subcontract: "
    "remove the 24-manhole radius cluster in RFI002 and install a formed GRP "
    "radiused or closed concrete channel. Write the invitation letter, scope "
    "of works, prequalification, evaluation method, and key dates in full "
    "paragraphs. Reference SOPR and UMA Stormwater DDC 20212200076."
)


@requires_construction_kit
@pytest.mark.asyncio
async def test_wir_form_refuses_rfi_and_rfp():
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    rfi = await c.wir_form({"text": M10}, {})
    assert rfi["status"] == "error"
    assert rfi.get("action") == "wir_form"
    assert "RFI" in (rfi.get("error") or "")
    rfp = await c.wir_form({"text": M14}, {})
    assert rfp["status"] == "error"
    assert "RFP" in (rfp.get("error") or "")
    still = await c.wir_form({"text": M15}, {})
    assert still["status"] == "success"
    assert still["week"] == "53"
    job = await c.wir_form({
        "text": (
            "Prepare a job requisition for street-lighting installation. "
            "Scope is the AM Rev Design NOC for Street Lighting."
        )
    }, {})
    assert job["status"] == "error"
    assert "job requisition" in (job.get("error") or "").lower() or "RFI" in (job.get("error") or "") or "RFP" in (job.get("error") or "")


def test_work_inspection_request_is_a_document_deliverable():
    assert _is_document_deliverable_request(M15)
    assert _is_document_deliverable_request(
        "Draft a delay claim notice under clause 8.8.1"
    )
    assert _is_document_deliverable_request(
        "Prepare an as-built deviation note for 350 vs 310"
    )


def test_search_preamble_forces_retry_on_wir():
    assert _looks_like_search_preamble("Let me search the WIR template.")
    assert _looks_like_search_preamble("I'll pull the spec and get back to you.")
    assert _final_text_needs_forced_retry(
        "Let me search the project documents for the WIR.",
        user_message=M15,
    )
    # REVERSED 2026-08-31. This used to assert that a plain factual question
    # may keep a dangling promise as its answer. Live WAVE-2 re-smoke on
    # e6c250e proved that wrong: D1 ("Let me search more precisely for the
    # Engineer's Representative...") and E1 ("I'm searching the executed
    # contract volume...") both shipped the promise to the user as the final
    # answer, because neither question was a deliverable or generative
    # request. A promise to search is not an answer to ANY question, so the
    # retry is now forced regardless of request type.
    assert _final_text_needs_forced_retry(
        "Let me search the project documents for the WIR.",
        user_message="what is the notice period?",
    )


def test_empty_or_content_filtered_choice():
    assert _llm_choice_is_empty_or_filtered({
        "finish_reason": "content_filter",
        "message": {"role": "assistant", "content": ""},
    })
    assert _llm_choice_is_empty_or_filtered({
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": ""},
    })
    assert not _llm_choice_is_empty_or_filtered({
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
    })
    assert not _llm_choice_is_empty_or_filtered({
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "Claim notice under 8.8.1"},
    })


@requires_construction_kit
def test_orchestrator_routes_wir_to_inspection_request():
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    result = _run(SmartOrchestratorBlock().process({"user_message": M15}))
    matched = [m["action"] for m in (result.get("matched_actions") or [])]
    assert "inspection_request" in matched


@requires_construction_kit
@pytest.mark.asyncio
async def test_predispatch_wir_form_injects_draft(monkeypatch):
    class _A:
        allowed_blocks = ["construction"]
        name = "construction-pm"

    messages = [{"role": "user", "content": M15}]
    out = await _predispatch_wir_form(_A(), messages, "proj")
    assert out is not None
    assert out["name"] == "wir_form"
    assert out["ok"] is True
    assert out["result"]["volume_m3"] == "28"
    injected = messages[-1]["content"]
    assert "PLATFORM PRE-DISPATCH: wir_form" in injected
    assert "Do not search again" in injected
    assert "28" in injected


@requires_construction_kit
@pytest.mark.asyncio
async def test_predispatch_wir_kill_switch(monkeypatch):
    monkeypatch.setenv("AGENT_WIR_PREDISPATCH", "0")

    class _A:
        allowed_blocks = ["construction"]

    out = await _predispatch_wir_form(
        _A(), [{"role": "user", "content": M15}], "proj",
    )
    assert out is None


def test_empty_fallback_recovers_predispatched_wir():
    draft = _format_wir_form({
        "wir_number": "DRAFT-WIR",
        "location": "Boulevard North-West",
        "activity": "Stormwater manhole collar concrete pour",
        "scope": "Week 53 — 11 manholes — C-35 SRC — 28 m³",
        "volume_m3": "28",
        "week": "53",
        "mix": "C-35 SRC",
        "hold_points": ["24-hour notice"],
        "witness_points": ["Slump test at discharge (witness)"],
        "checklist": [{"item": "Mix ready"}],
        "signatories": [{"party": "Contractor QC", "status": "pending"}],
        "notice": "No pour until sign-off.",
    })
    recovered = _recover_answer_from_tool_messages(
        _EMPTY_RESPONSE_FALLBACK,
        [{
            "role": "user",
            "content": (
                "PLATFORM PRE-DISPATCH: wir_form has ALREADY been run from "
                "the operator facts in this turn. Authoritative draft:\n"
                f"{draft}\n"
                "Present this WIR in full. Do not search again for the "
                "template. Do not claim the inspection was issued."
            ),
        }],
    )
    assert recovered != _EMPTY_RESPONSE_FALLBACK
    assert "28 m³" in recovered
    assert "Slump" in recovered
