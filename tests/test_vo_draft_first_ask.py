"""Live Phase 2: variation_order_manager must draft on the FIRST ask.

Ask1 used to refuse / return hollow Status:Success / route to
change_order_impact (or the empty-fixture unindexed gate). Ask2 sometimes
drafted VO-D-002 with ADD/OMIT lines MATCH. Owner requirement: the first
variation-order ask that already carries synthetic ADD/OMIT + rates must
return a drafted VO body (description, lines, net).

These tests do NOT require CEREBRUM_DOMAIN_KITS=construction. Virgin CI
skipped the kit-gated copies and dropped diff coverage below 50%.
"""
from __future__ import annotations

import json

import pytest

from app.containers.construction.boq import (
    _parse_vo_priced_lines,
    _variation_has_draft_facts,
)
from app.core.site_vocab import message_wants_vo_draft


# Self-contained first ask — no project corpus, no prior turn, priced lines
# in the operator message. ADD 120 × 85 = 10_200; OMIT 40 × 45 = 1_800;
# net = 8_400.
FIRST_VO_ASK = (
    "Draft a variation order for the following priced lines:\n"
    "ADD 120 m2 ceramic tiling @ 85 SAR/m2\n"
    "OMIT 40 m2 carpet @ 45 SAR/m2"
)

ADD_AMOUNT = 10_200.0
OMIT_AMOUNT = 1_800.0
NET_AMOUNT = 8_400.0


def _matched_actions(message: str) -> list[str]:
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    return [
        m["action"]
        for m in SmartOrchestratorBlock()._match_actions(message, None)
    ]


def _assert_drafted_vo_body(result: dict) -> None:
    """A first-ask VO must be a real draft, not hollow success / empty."""
    assert isinstance(result, dict), result
    assert result.get("status") == "success", result
    assert result.get("action") not in {
        None,
        "change_order_analysis",
        "change_order_impact",
        "intelligent_workflow",
    }, result
    assert result.get("error") in (None, ""), result

    description = (
        result.get("description")
        or result.get("vo_description")
        or ""
    )
    assert str(description).strip(), f"missing description: {result}"
    assert "draft a variation order" not in str(description).strip().lower() or (
        "tiling" in str(description).lower() or "carpet" in str(description).lower()
    )

    lines = result.get("lines") or result.get("vo_lines") or []
    assert isinstance(lines, list) and len(lines) >= 2, result
    blob = json.dumps(lines).upper()
    assert "ADD" in blob, lines
    assert "OMIT" in blob, lines
    assert "120" in json.dumps(lines)
    assert "40" in json.dumps(lines)
    assert "85" in json.dumps(lines)
    assert "45" in json.dumps(lines)

    net = result.get("net")
    if net is None:
        net = (result.get("pricing") or {}).get("total_value")
    assert net is not None, result
    assert float(net) == pytest.approx(NET_AMOUNT, rel=1e-4)

    body = (
        result.get("document_content")
        or result.get("vo_document")
        or result.get("draft")
        or ""
    )
    assert str(body).strip(), f"hollow success / empty draft: {result}"
    vo_no = str(result.get("vo_number") or "")
    assert vo_no, result
    assert vo_no.upper().startswith("VO"), vo_no


def test_message_wants_vo_draft_not_impact():
    assert message_wants_vo_draft(FIRST_VO_ASK)
    assert message_wants_vo_draft("issue a VO for extra blockwork")
    assert not message_wants_vo_draft(
        "assess the cost and time impact of variation order VO-12"
    )
    assert not message_wants_vo_draft("update the variation log")


def test_parse_vo_priced_lines_add_omit_and_aliases():
    assert _parse_vo_priced_lines("") == []
    assert _parse_vo_priced_lines(None) == []
    parsed = _parse_vo_priced_lines(FIRST_VO_ASK)
    assert [p["kind"] for p in parsed] == ["ADD", "OMIT"]
    assert parsed[0]["amount"] == pytest.approx(ADD_AMOUNT)
    assert parsed[1]["amount"] == pytest.approx(-OMIT_AMOUNT)

    alias = _parse_vo_priced_lines(
        "OMISSION 40 m2 carpet @ 45 SAR/m2\n"
        "DELETE 10 m2 paint @ 20\n"
        "DEDUCT 5 m2 grout @ 10"
    )
    assert {r["kind"] for r in alias} == {"OMIT"}
    assert all(r["amount"] < 0 for r in alias)

    flipped = _parse_vo_priced_lines("ADD ceramic tiling 120 m2 @ 85 SAR/m2")
    assert len(flipped) == 1
    assert flipped[0]["quantity"] == 120.0
    assert flipped[0]["rate"] == 85.0

    dup = _parse_vo_priced_lines(
        "ADD 120 m2 ceramic tiling @ 85\nADD 120 m2 ceramic tiling @ 85"
    )
    assert len(dup) == 1

    skipped = _parse_vo_priced_lines("ADD 0 m2 ceramic tiling @ 85")
    assert skipped == []


def test_variation_has_draft_facts_from_priced_lines():
    assert _variation_has_draft_facts({}, FIRST_VO_ASK) is True
    assert _variation_has_draft_facts(None, FIRST_VO_ASK) is True
    assert _variation_has_draft_facts({}, "Draft a variation order") is False
    assert _variation_has_draft_facts(
        {"description": "extra blockwork", "direct_cost": "not-a-number"},
        "",
    ) is False


def test_first_vo_draft_ask_prefers_variation_order_manager():
    """change_order_impact must not steal a VO *draft* intent on ask1."""
    matched = _matched_actions(FIRST_VO_ASK)
    assert matched, FIRST_VO_ASK
    assert matched[0] == "variation_order_manager", matched
    assert "change_order_impact" not in matched


def test_impact_ask_still_reaches_change_order_impact():
    """Impact analysis stays on change_order_impact (not a draft intent)."""
    matched = _matched_actions(
        "assess the cost and time impact of variation order VO-12 "
        "adding 300m of storm drain"
    )
    assert "change_order_impact" in matched, matched


@pytest.mark.asyncio
async def test_first_vo_ask_with_add_omit_rates_drafts_body():
    """First ask (no history, no vo_data) drafts description / lines / net."""
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().variation_order_manager(
        {"message": FIRST_VO_ASK},
        {},
    )
    _assert_drafted_vo_body(result)
    assert str(result.get("vo_number", "")).upper().startswith("VO-D"), result
    assert result["vo_type"] == "mixed"


@pytest.mark.asyncio
async def test_first_vo_ask_via_route_is_not_change_order_only():
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    routed = await container.route(
        "variation_order_manager",
        {"message": FIRST_VO_ASK, "text": FIRST_VO_ASK},
        {},
    )
    _assert_drafted_vo_body(routed)

    stolen = await container.change_order_impact(
        {"message": FIRST_VO_ASK, "text": FIRST_VO_ASK},
        {},
    )
    stolen_lines = stolen.get("lines") or stolen.get("vo_lines") or []
    stolen_net = stolen.get("net")
    assert not (
        stolen.get("status") == "success"
        and stolen.get("action") == "change_order_analysis"
        and not stolen_lines
        and stolen_net is None
    ) or routed.get("action") != "change_order_analysis"


@pytest.mark.asyncio
async def test_omit_only_and_add_only_drafts():
    from app.containers.construction import ConstructionContainer

    omit = await ConstructionContainer().variation_order_manager(
        {"text": "Draft a variation order: OMIT 40 m2 carpet @ 45 SAR/m2"},
        {},
    )
    assert omit["status"] == "success"
    assert omit["vo_type"] == "omission"
    assert omit["net"] == pytest.approx(-1800.0)
    assert omit["vo_number"].startswith("VO-D-")

    add = await ConstructionContainer().variation_order_manager(
        {
            "variation_data": {"description": "Draft a variation order"},
            "text": "ADD 120 m2 ceramic tiling @ 85 SAR/m2",
        },
        {},
    )
    assert add["status"] == "success"
    assert add["vo_type"] == "addition"
    assert add["net"] == pytest.approx(ADD_AMOUNT)
    assert "tiling" in add["description"].lower()


@pytest.mark.asyncio
async def test_priced_lines_plus_contract_value_still_drafts():
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().variation_order_manager(
        {"message": FIRST_VO_ASK, "contract_value": 1_000_000},
        {},
    )
    _assert_drafted_vo_body(result)
    assert result["cumulative_impact"]["percent_of_contract"] is not None


@pytest.mark.asyncio
async def test_intelligent_workflow_draft_vs_impact_chain():
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    draft_chain = c._build_intelligent_chain(FIRST_VO_ASK, None)
    actions = [s["action"] for s in draft_chain]
    assert "variation_order_manager" in actions
    assert "change_order_impact" not in actions

    impact_chain = c._build_intelligent_chain(
        "assess the cost and time impact of variation order VO-12",
        None,
    )
    impact_actions = [s["action"] for s in impact_chain]
    assert "change_order_impact" in impact_actions
    assert "variation_order_manager" in impact_actions


@pytest.mark.asyncio
async def test_first_vo_ask_predispatch_injects_draft(monkeypatch):
    from app.agents.runtime import (
        _conflicting_tools_after_predispatch,
        _format_variation_order,
        _message_wants_locked_deliverable,
        _message_wants_vo_draft,
        _predispatch_remaining_deliverables,
        _recover_answer_from_tool_messages,
    )
    from app.containers.construction import ConstructionContainer

    assert _message_wants_vo_draft(FIRST_VO_ASK)
    assert _message_wants_locked_deliverable(FIRST_VO_ASK)
    assert "change_order_impact" in _conflicting_tools_after_predispatch(
        "variation_order_manager"
    )

    monkeypatch.setattr(
        "app.dependencies.get_block_instance",
        lambda name: ConstructionContainer() if name == "construction" else None,
    )

    class _A:
        allowed_blocks = ["construction"]
        name = "contracts-manager"

    msgs = [{"role": "user", "content": FIRST_VO_ASK}]
    out = await _predispatch_remaining_deliverables(_A(), msgs, "proj")
    assert out is not None
    assert out["name"] == "variation_order_manager"
    assert out.get("ok") is True
    _assert_drafted_vo_body(out["result"])
    injected = msgs[-1]["content"]
    assert "PLATFORM PRE-DISPATCH" in injected
    assert "ADD" in injected.upper()
    assert "OMIT" in injected.upper()
    assert "8400" in injected.replace(",", "").replace(" ", "") or "8,400" in injected

    rendered = _format_variation_order(out["result"])
    assert "VO-D" in rendered
    assert "Net" in rendered
    loose = _format_variation_order({
        "action": "variation_order_processed",
        "vo_number": "VO-D-009",
        "description": "scope",
        "lines": ["ADD leftover"],
        "pricing": {"total_value": 1},
    })
    assert "ADD leftover" in loose

    recovered = _recover_answer_from_tool_messages(
        "HTTP 413",
        [{"role": "tool", "content": json.dumps(out["result"])}],
    )
    assert "VO-D" in recovered


def test_empty_fixture_vo_draft_is_not_an_unindexed_refusal():
    """Empty FIXTURE projects must not early-return the unindexed refusal."""
    from app.agents.runtime import (
        _project_has_non_rag_context,
        _should_short_circuit_rag_miss,
    )

    assert _project_has_non_rag_context("empty-fixture", FIRST_VO_ASK) is True
    assert _should_short_circuit_rag_miss(
        {
            "identifier_miss": True,
            "threshold_fired": True,
            "extracted_identifiers": ["85/m2", "45/m2", "VO-D-001"],
        },
        None,
        FIRST_VO_ASK,
    ) is False


def test_generate_vo_document_lists_lines():
    from app.containers.construction import ConstructionContainer

    body = ConstructionContainer()._generate_vo_document(
        "VO-D-001",
        "tiling vs carpet",
        {"total": 8400.0},
        "mixed",
        lines=[
            {"kind": "ADD", "quantity": 120, "unit": "m2",
             "description": "tiling", "rate": 85, "amount": 10200},
            "bare-line",
        ],
    )
    assert "VO-D-001" in body
    assert "ADD 120" in body
    assert "bare-line" in body
    assert "Net: 8400" in body
