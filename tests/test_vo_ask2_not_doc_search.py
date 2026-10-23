"""Live tip d9d5971: VO ask2 must draft via variation_order_manager, not doc search.

#683 / SO7 hard-excluded validation_pipeline + delegate_to_agent (and the
earlier formula/template/change_order_impact set). Ask1 PASSes with a real
VO draft. Ask2 still ran tools=['search_project_documents','fetch_document']
and answered status/search chatter (“BOQ on file has no drainage…
Computing the VO totals: 2 of 17 project documents indexed”) — scored
insufficient VO draft.

Root on current main: LIVE ask2 priced lines do not parse (ADD:/OMIT: plus
fittings/each), so AGENT_VO_PREDISPATCH misses; fetch_document then locks
synthesis and variation_order_manager never runs.

Hard-exclude search/fetch (and list) on VO-draft intent; steal them after
VO predispatch; parse the live ADD:/OMIT: shape so predispatch actually
invokes variation_order_manager with ADD/OMIT + qty + rate + net AED.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _conflicting_tools_after_predispatch,
    _forced_specific_tool,
    _message_wants_vo_draft,
    _vo_draft_hard_excludes,
    _vo_draft_ready_for_synthesis,
)
from app.containers.construction.boq import (
    _parse_vo_priced_lines,
    _variation_has_draft_facts,
)
from app.core.site_vocab import message_wants_vo_draft

LIVE_ASK2 = (
    "variation_order_manager: DRAFT VO-D-002 content (must include drafted clauses/lines, "
    "not merely 'Status: Success'). ADD: additional drainage 45 m @ AED 180/m; "
    "OMIT: omit feature lighting 12 fittings @ AED 2500 each. Show AED totals."
)

DOC_STEAL = (
    "search_project_documents",
    "fetch_document",
    "list_project_documents",
)

ASK2_AVAIL = {
    "variation_order_manager",
    "search_project_documents",
    "fetch_document",
    "list_project_documents",
    "construction_calc",
    "validation_pipeline",
    "delegate_to_agent",
}

# ADD 45 m × 180 = 8_100; OMIT 12 fittings × 2500 = 30_000; net = −21_900
ADD_AMOUNT = 8_100.0
OMIT_AMOUNT = 30_000.0
NET_AMOUNT = -21_900.0


def test_live_ask2_wants_vo_draft():
    assert message_wants_vo_draft(LIVE_ASK2)
    assert _message_wants_vo_draft(LIVE_ASK2)


def test_live_ask2_hard_excludes_doc_search_and_fetch():
    ex = _vo_draft_hard_excludes(LIVE_ASK2)
    for name in DOC_STEAL:
        assert name in ex, name


def test_live_ask2_predispatch_steals_doc_search_and_fetch():
    steal = _conflicting_tools_after_predispatch("variation_order_manager")
    for name in DOC_STEAL:
        assert name in steal, name


def test_live_ask2_fetch_does_not_lock_synthesis():
    """fetch_document must not force_synthesis on a VO-draft turn."""
    assert _vo_draft_ready_for_synthesis(LIVE_ASK2, "fetch_document") is False
    assert _vo_draft_ready_for_synthesis(LIVE_ASK2, "search_project_documents") is False
    assert _vo_draft_ready_for_synthesis(LIVE_ASK2, "list_project_documents") is False
    assert _vo_draft_ready_for_synthesis(LIVE_ASK2, "variation_order_manager") is True
    assert _vo_draft_ready_for_synthesis("what is the concrete spec?", "fetch_document") is True


def test_live_ask2_forces_variation_order_manager_not_doc_search():
    assert (
        _forced_specific_tool([{"role": "user", "content": LIVE_ASK2}], ASK2_AVAIL)
        == "variation_order_manager"
    )


def test_live_ask2_priced_lines_parse_add_omit_net():
    """Colon-tagged ADD:/OMIT: + fittings/each must be draft facts, not empty."""
    parsed = _parse_vo_priced_lines(LIVE_ASK2)
    kinds = [p["kind"] for p in parsed]
    assert kinds == ["ADD", "OMIT"], parsed
    add = next(p for p in parsed if p["kind"] == "ADD")
    omit = next(p for p in parsed if p["kind"] == "OMIT")
    assert add["quantity"] == pytest.approx(45.0)
    assert add["rate"] == pytest.approx(180.0)
    assert add["amount"] == pytest.approx(ADD_AMOUNT)
    assert "drain" in str(add.get("description") or "").lower()
    assert omit["quantity"] == pytest.approx(12.0)
    assert omit["rate"] == pytest.approx(2500.0)
    assert omit["amount"] == pytest.approx(-OMIT_AMOUNT)
    blob = str(omit.get("description") or "").lower()
    assert "light" in blob or "fitting" in blob
    assert _variation_has_draft_facts({}, LIVE_ASK2) is True


@pytest.mark.asyncio
async def test_live_ask2_predispatch_invokes_variation_order_manager(monkeypatch):
    from app.agents.runtime import _predispatch_remaining_deliverables
    from app.containers.construction import ConstructionContainer

    monkeypatch.setattr(
        "app.dependencies.get_block_instance",
        lambda name: ConstructionContainer() if name == "construction" else None,
    )

    class _A:
        allowed_blocks = ["construction"]
        name = "contracts-manager"

    msgs = [{"role": "user", "content": LIVE_ASK2}]
    out = await _predispatch_remaining_deliverables(_A(), msgs, "proj")
    assert out is not None
    assert out["name"] == "variation_order_manager"
    assert out.get("ok") is True
    result = out["result"]
    assert result.get("status") == "success"
    lines = result.get("lines") or []
    kinds = {str(l.get("kind") or "").upper() for l in lines}
    assert "ADD" in kinds and "OMIT" in kinds, lines
    assert result.get("net") == pytest.approx(NET_AMOUNT)
    injected = msgs[-1]["content"]
    assert "PLATFORM PRE-DISPATCH" in injected
    assert "variation_order_manager" in injected
    low = injected.lower()
    assert "drain" in low
    assert "8100" in injected.replace(",", "").replace(" ", "") or "8,100" in injected
