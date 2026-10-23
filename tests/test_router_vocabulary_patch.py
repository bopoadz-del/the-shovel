"""TASK 1 vocabulary patch: the pilot-critical phrasings that missed at
confidence 0.0 in the 2026-07-06 feature-matrix sweep must route.

Source of truth: the routing-miss evidence log (DECISIONS.md) - verbatim
engineer phrasings from FEATURE_MATRIX.md's 20 class-1 misses. The patch is
ADDITIVE keyword vocabulary only; thresholds and matcher logic unchanged.
Two scoring facts these cases depend on (see _match_actions):

* a single-word keyword scores 0.2, below the 0.3 non-generative gate -
  which is why multi-word phrases were added rather than bare words;
* keyword matching is word-boundary exact, so plurals never matched their
  singular keyword ("drawings" vs "drawing").

If one of these starts failing, the router vocabulary regressed for a
feature the pilot demos.
"""
from __future__ import annotations

import asyncio

import pytest

from app.blocks.smart_orchestrator import SmartOrchestratorBlock


@pytest.fixture(scope="module")
def block():
    return SmartOrchestratorBlock()


def _matched(block, message: str) -> list:
    r = asyncio.run(block.process({"message": message}))
    return [m["action"] for m in (r.get("matched_actions") or [])] or list(
        r.get("action_queue") or []
    )


# (verbatim sweep phrasing, action that must appear in the match set)
SWEEP_MISS_PHRASINGS = [
    ("what does the the client project project execution plan cover?", "process_document"),
    ("analyze the concrete specification requirements - what grades and standards apply?", "spec_analyze"),
    ("pull out the material specs for the road works", "spec_analyze"),
    ("list the documents in this project and what type each one is", "process_document"),
    ("which drawings do we have for the stormwater network?", "process_document"),
    ("extract the key milestones from the project programme with their dates", "parse_primavera_schedule"),
    ("give me a milestone report - what are the major completion dates?", "parse_primavera_schedule"),
    ("what does the cumulative spend curve look like month by month?", "cash_flow_forecast"),
    ("what materials do we need to buy for the substructure works?", "procurement_list_generator"),
    ("prepare an RFP for the landscaping subcontract package", "rfp_management"),
    ("do a quantity takeoff from the infrastructure drawings - pipe lengths and manhole counts", "drawing_qto"),
    ("what T&C steps do we need before energising the electrical rooms?", "commissioning_checklist"),
    ("take off the concrete quantities for the ground floor slabs", "extract_quantities"),
]


@pytest.mark.parametrize("message,action", SWEEP_MISS_PHRASINGS,
                         ids=[a for _, a in SWEEP_MISS_PHRASINGS])
def test_sweep_miss_phrasing_routes(block, message, action):
    matched = _matched(block, message)
    assert action in matched, (
        f"vocabulary regression: {action!r} not in {matched!r} for {message!r}"
    )


# The zero-sum risk the additive patch must not realise: phrasings that
# routed correctly BEFORE the patch keep routing to the same action.
PREVIOUSLY_PASSING = [
    ("generate a commissioning checklist for the MV substation", "commissioning_checklist"),
    ("generate a level 2 construction schedule for a 30-storey residential tower, 30 months", "generate_wbs"),
    ("build a cash flow forecast with an S-curve for a SAR 60M, 18-month project", "cash_flow_forecast"),
    ("produce a manpower histogram for the structure works over 12 months", "resource_histogram"),
    ("process the bill of quantities and give me the total value", "boq_process"),
    ("generate a procurement list for the MEP package", "procurement_list_generator"),
    ("for a 2.5m thick mass concrete raft, what thermal control limits apply and how long to equilibrium?", "construction_advisor"),
]


@pytest.mark.parametrize("message,action", PREVIOUSLY_PASSING,
                         ids=[a for _, a in PREVIOUSLY_PASSING])
def test_previously_passing_phrasing_still_routes(block, message, action):
    matched = _matched(block, message)
    assert action in matched, (
        f"regression: patch stole {action!r} - got {matched!r} for {message!r}"
    )
