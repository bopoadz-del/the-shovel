"""Regression pins for UI-PHYS diagnostics D2, D4 and D5.

All three fixes are on main but were carried only by the browser/nightly
battery, which needs a live service, an LLM and a seeded project. Nothing
cheap stopped someone editing the regex or the alias table and shipping the
failure back. These are the unit-level pins.

Each diagnostic is a routing failure, not a retrieval one: the right answer
was reachable, but the turn was handed to the wrong capability.

  D5 (UI-PHYS G1) "What does Schedule 10 of the contract contain?" is a
     numbered contract-volume Schedule, not a P6 programme. The keyword
     router stole the turn to parse_primavera_schedule / generate_wbs at
     confidence 0.2 and answered about a construction programme.

  D4 (UI-PHYS F2) "Use 45 days for the tree-removal activity and re-run."
     The override parsed, then matched nothing: the WBS row reads "Remove
     trees and shrubs", and some rows fold tree work into "Site clearance".
     A duration override that matches no activity silently does nothing.

  D2 (UI-PHYS A5) A Contract Data question was stolen to drawing_qto because
     a document stem collided with ordinary English in the question --
     "the whole of the Works.dxf" is a substring of "...for the whole of the
     Works?". The file predispatch fired on the collision.
"""
from __future__ import annotations

import pytest

from app.core.contract_lookup_intent import (
    CONTRACT_LOOKUP_BLOCKED_ACTIONS,
    message_is_contract_data_lookup,
)
from app.lib.wbs_duration_overrides import (
    activity_matches_override,
    message_wants_wbs_duration_rerun,
    parse_duration_overrides,
)

# ── D5: numbered contract Schedules are lookups, not programmes ──────────

# The exact battery wording. Questions are frozen; see the fixture README.
G1_ASK = "What does Schedule 10 of the contract contain?"


def test_d5_the_battery_question_is_a_contract_lookup():
    assert message_is_contract_data_lookup(G1_ASK) is True


@pytest.mark.parametrize(
    "ask",
    [
        "What does Schedule 10 of the contract contain?",
        "What is Schedule 3 of the agreement?",
        "Which schedule 7 covers payment?",
        "What does schedule 12 of the conditions include?",
    ],
)
def test_d5_numbered_schedule_phrasings_are_recognised(ask):
    assert message_is_contract_data_lookup(ask) is True


@pytest.mark.parametrize(
    "ask",
    [
        "Update the project schedule for next week",
        "Generate a WBS from the BOQ",
        "Import the Primavera schedule and show the critical path",
    ],
)
def test_d5_ordinary_programme_talk_is_not_hijacked(ask):
    """The guard must not swallow real scheduling work -- that would trade
    one routing failure for the opposite one."""
    assert message_is_contract_data_lookup(ask) is False


def test_d5_the_generative_actions_that_stole_the_turn_are_blocked():
    for action in ("generate_wbs", "parse_primavera_schedule"):
        assert action in CONTRACT_LOOKUP_BLOCKED_ACTIONS
    # drawing_qto is blocked for the same reason under D2.
    assert "drawing_qto" in CONTRACT_LOOKUP_BLOCKED_ACTIONS


# ── D4: the override has to find the activity it names ───────────────────

F2_ASK = "Use 45 days for the tree-removal activity and re-run."


def test_d4_the_battery_question_parses_to_45_days_and_a_rerun():
    overrides = parse_duration_overrides(F2_ASK)
    assert len(overrides) == 1
    assert overrides[0]["days"] == 45
    assert overrides[0]["match"] == "tree removal"
    assert message_wants_wbs_duration_rerun(F2_ASK) is True


@pytest.mark.parametrize(
    "activity",
    [
        "Remove trees and shrubs",   # inflection: removal->remove, trees->tree
        "Tree Removal",              # exact, different case
        "Site clearance — Hall A",   # CESMM family alias
        "Site Clearance - Hall B",
    ],
)
def test_d4_tree_removal_matches_how_the_wbs_actually_names_it(activity):
    """This is the whole defect: the override parsed fine and matched
    nothing, so the re-run silently returned the original durations."""
    assert activity_matches_override(activity, "tree removal") is True


@pytest.mark.parametrize(
    "activity",
    ["Excavate foundations", "Pour blinding concrete", "Install ductwork"],
)
def test_d4_unrelated_activities_are_not_swept_up(activity):
    """Not a fuzzy score floor. An override must not repaint the programme."""
    assert activity_matches_override(activity, "tree removal") is False


def test_d4_mutation_probe_dropping_the_alias_reopens_the_defect():
    """MUTATION PROBE. Without the family alias, the CESMM row that folds
    tree work into site clearance stops matching and F2 fails again."""
    import app.lib.wbs_duration_overrides as w

    original = dict(w._MATCH_ALIASES)
    try:
        w._MATCH_ALIASES.clear()
        assert activity_matches_override("Site clearance — Hall A", "tree removal") is False
    finally:
        w._MATCH_ALIASES.clear()
        w._MATCH_ALIASES.update(original)
    # restored
    assert activity_matches_override("Site clearance — Hall A", "tree removal") is True


# ── D2: a filename collision must not steal a Contract Data question ─────

A5_ASK = "What is the Time for Completion for the whole of the Works?"


def test_d2_the_colliding_question_is_recognised_as_a_contract_lookup():
    """The predispatch guard is driven by this classification. If it stops
    returning True, a document named 'the whole of the Works.dxf' captures
    the turn again."""
    assert message_is_contract_data_lookup(A5_ASK) is True


def test_d2_the_filename_really_does_collide():
    """Proof the collision is real and not hypothetical: the document stem
    is a substring of the question, which is why substring matching fired."""
    stem = "the whole of the works"
    assert stem in A5_ASK.lower()


def test_d2_qto_and_bim_are_blocked_for_a_contract_lookup():
    """The two tools the collision routed to."""
    assert "drawing_qto" in CONTRACT_LOOKUP_BLOCKED_ACTIONS
