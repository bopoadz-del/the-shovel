"""Tests for the pure evidence-plan + coverage-check pieces (item 2)."""
from __future__ import annotations

import pytest

from app.core.rag.evidence_plan import (
    ROUND_LIMIT,
    EvidencePlan,
    missing_plan_items,
    parse_evidence_plan,
    round_allowed,
)


@pytest.mark.parametrize("text,items", [
    # bulleted plan
    ("- bulk excavation rows\n- blinding rows\n- lift wall rows",
     ("bulk excavation rows", "blinding rows", "lift wall rows")),
    # numbered plan
    ("1. concrete quantities\n2. formwork quantities",
     ("concrete quantities", "formwork quantities")),
    # single comma-separated line
    ("Bill No.4 items for excavation, blinding, lift walls",
     ("Bill No.4 items for excavation", "blinding", "lift walls")),
])
def test_parse_evidence_plan_splits_items(text, items):
    assert parse_evidence_plan(text).items == items


@pytest.mark.parametrize("text,items", [
    # a preamble line ending in ":" before a numbered list
    ("Here is where to look, in order:\n1. drainage bill rows\n2. manhole schedule",
     ("drainage bill rows", "manhole schedule")),
    # a bulleted list with prose before and after it, no colon anywhere
    ("Sources worth checking first\n- piling bill rows\n- pile schedule drawing\n"
     + "That should cover it",
     ("piling bill rows", "pile schedule drawing")),
])
def test_only_list_entries_are_items(text, items):
    assert parse_evidence_plan(text).items == items


@pytest.mark.parametrize("text,items", [
    # plain (unmarked) plan: a colon-ended intro line is dropped, lines kept
    ("Look in these places:\nexternal works bill\nsite layout drawing",
     ("external works bill", "site layout drawing")),
    # intro line then a single comma-separated plain line is still split
    ("Needed:\nroad kerbs, footpath paving",
     ("road kerbs", "footpath paving")),
])
def test_plain_plan_drops_colon_intro_line(text, items):
    assert parse_evidence_plan(text).items == items


@pytest.mark.parametrize("text,items", [
    # a colon inside an entry is content, not an intro
    ("- Priced bill: unit rate column for gullies\n- Spec: bedding class",
     ("Priced bill: unit rate column for gullies", "Spec: bedding class")),
    ("1. Schedule: door types\n2. Bill: ironmongery rows",
     ("Schedule: door types", "Bill: ironmongery rows")),
])
def test_mid_line_colon_entry_is_kept(text, items):
    assert parse_evidence_plan(text).items == items


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_parse_evidence_plan_empty(text):
    assert parse_evidence_plan(text).items == ()


@pytest.mark.parametrize("plan,evidence,missing", [
    # every item's term appears -> nothing missing
    (EvidencePlan("p", ("bulk excavation", "blinding")),
     "bulk excavation 75 mm blinding to slab",
     ()),
    # one item absent -> it is missing
    (EvidencePlan("p", ("bulk excavation", "staircase landings")),
     "bulk excavation 75 mm blinding to slab",
     ("staircase landings",)),
])
def test_missing_plan_items_is_lexical(plan, evidence, missing):
    assert missing_plan_items(plan, evidence) == missing


@pytest.mark.parametrize("plan_text,evidence,missing", [
    ("bulk excavation, lift walls, staircases",
     "the priced BOQ has raft concrete and shear walls",
     {"bulk excavation", "lift walls", "staircases"}),
    ("blinding, formwork",
     "the priced BOQ has painting and ceilings",
     {"blinding", "formwork"}),
])
def test_a_plan_item_with_zero_evidence_is_missing(plan_text, evidence, missing):
    plan = parse_evidence_plan(plan_text)
    assert set(missing_plan_items(plan, evidence)) == missing


@pytest.mark.parametrize("item", ["the", "and of the"])
def test_an_empty_term_item_is_not_a_gap(item):
    # An item whose only words are stopwords carries no signal: not a gap.
    plan = EvidencePlan("p", (item,))
    assert missing_plan_items(plan, "any evidence") == ()


def test_round_bound_never_allows_round_four():
    assert round_allowed(1) and round_allowed(2) and round_allowed(ROUND_LIMIT)
    assert not round_allowed(ROUND_LIMIT + 1)
    assert not round_allowed(0)
    assert not round_allowed(-1)
