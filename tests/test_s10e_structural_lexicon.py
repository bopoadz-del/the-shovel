"""S-10e: structural-design vocabulary is recognised as a CLASS (a lexicon),
not per-phrase keyword literals.

The design hat used 13 hand-kept keyword triggers (beam, column, ...), so bare
class members it had not listed -- slab, foundation, pile cap, clear span, load
path -- did not route. A TriggerType.LEXICON trigger naming the "structural"
class (app/agents/signal_lexicons.py) now fires on ANY member of the class, so
the whole vocabulary routes to the design hat. Removing the class check drops
every class-only ask back to 0 (the literals never covered them).
"""
from __future__ import annotations

import pytest

from app.agents import load_agents
from app.agents.catalog import get_hat
from app.agents.models import Discipline
from app.agents.signal_lexicons import lexicon_matches


def _design():
    load_agents()
    return get_hat(Discipline.DESIGN)


# Each ask uses ONLY class members that are NOT among the 13 literal triggers,
# so the score it earns can come only from the structural lexicon.
@pytest.mark.parametrize("ask", [
    "size the slab and its foundation",
    "check the pile cap and the clear span under this load path",
])
def test_a_class_only_structural_ask_routes_to_design(ask):
    m = _design()
    assert m is not None
    assert m.score_message(ask) >= 0.8, (ask, m.score_message(ask))


@pytest.mark.parametrize("ask", [
    "what is the contract completion date",
    "summarise the delay damages clause and the retention bond",
])
def test_a_non_design_ask_does_not_route_to_design(ask):
    m = _design()
    assert m.score_message(ask) == 0.0, (ask, m.score_message(ask))


# lexicon_matches: whole-word / whole-phrase, two inputs each way.
@pytest.mark.parametrize("message, hit", [
    ("the load path governs", True),
    ("inspect the slabs on level 3", True),      # plural member
    ("a columnar chromatogram", False),          # substring, not the word
    ("the quarterly budget review", False),      # no class term
])
def test_lexicon_matches_is_whole_word(message, hit):
    assert lexicon_matches("structural", message) is hit


def test_unknown_lexicon_is_never_a_match():
    assert lexicon_matches("no-such-class", "beam column slab") is False
    assert lexicon_matches("", "beam") is False
