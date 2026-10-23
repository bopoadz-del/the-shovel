"""rfi_generator ask1 must draft — not drawing_qto / cde_post_rfi / RAG-miss.

Live tip 50c37f ask1: routed action=drawing_qto (Drawing S-201 / A-105),
then answered "could not confirm this reference … FIXTURE-d-…" with
tools=[]. #659 exempted unindexed-project refusal + calc predispatch;
RAG-miss short-circuit and drawing_qto keyword steal remained.
Clash + the word RFI then stole the draft onto cde_post_rfi via
keyword 'clash rfi'.
"""
from __future__ import annotations

from app.agents.runtime import (
    _message_wants_rfi_draft,
    _should_short_circuit_rag_miss,
)
from app.blocks.smart_orchestrator import SmartOrchestratorBlock
from app.core.action_router import message_wants_rfi_draft
from app.core.site_vocab import message_wants_clash_cde_rfi


LIVE_RFI_ASK1 = (
    "Use rfi_generator. Draft ONE RFI from this synthetic clash: Drawing S-201 footing F3 "
    "centerline conflicts with Drawing A-105 door D-12 swing — 150mm overlap. "
    "Ask engineer to confirm footing relocation or door swing revision."
)

LIVE_RFI_ASK2 = (
    "rfi_generator: create RFI for synthetic clash between MEP duct on M-301 at grid B/3 "
    "and beam soffit on S-220 (duct IL +3.10 vs beam soffit +3.05). Request clarification."
)


def test_live_rfi_asks_detected():
    assert _message_wants_rfi_draft(LIVE_RFI_ASK1)
    assert message_wants_rfi_draft(LIVE_RFI_ASK1)
    assert _message_wants_rfi_draft(LIVE_RFI_ASK2)


def test_rfi_ask1_not_rag_miss_short_circuit():
    audit = {
        "identifier_miss": True,
        "extracted_identifiers": ["S-201", "A-105", "D-12", "150"],
    }
    assert not _should_short_circuit_rag_miss(audit, None, LIVE_RFI_ASK1)


def test_orchestrator_promotes_rfi_over_drawing_qto():
    block = SmartOrchestratorBlock()
    for ask in (LIVE_RFI_ASK1, LIVE_RFI_ASK2):
        out = block._match_actions(ask, None)  # noqa: SLF001
        assert out
        assert out[0]["action"] == "rfi_generator", (ask, out[:4])
        stolen = {"drawing_qto", "cde_post_rfi"}
        assert not any(r["action"] in stolen for r in out), (ask, out[:6])
        assert not message_wants_clash_cde_rfi(ask)


def test_explicit_aconex_post_still_routes_cde():
    """Clash + 'post an RFI to Aconex' is CDE, not a local draft."""
    msg = "Run clash detection and post an RFI to Aconex for the duct/beam clash."
    assert message_wants_clash_cde_rfi(msg)
    assert not message_wants_rfi_draft(msg)
    block = SmartOrchestratorBlock()
    out = block._match_actions(msg, None)  # noqa: SLF001
    actions = [r["action"] for r in out]
    assert "cde_post_rfi" in actions
    assert "rfi_generator" not in actions
