"""ROUTING MUST NOT DEPEND ON SENTENCE SHAPE (owner's numbered item 3).

F-PHRASE-1, ``FLEET_OPS/artifacts/gate_battery_13b2bf7_2026-08-31.md``,
quoted verbatim:

    the routing is phrasing-sensitive, and the sheet's phrasing is the one
    that fails. Two IDs, same session:
    - E4 "Concrete volume for a raft 30x20x1.5 m including your documented
      waste factor." -> refusal. A conversational paraphrase of the same
      question ... fired construction_calc and returned 900 -> 945 m3,
      119 trucks, an hour earlier on 87c7996.
    ...
    A prior "PASS" obtained on a paraphrase is not evidence about the
    battery question.

THE CAUSE, measured rather than guessed. ``_looks_like_self_contained_
calculation`` needs a calc verb plus two measurement-bearing numbers.
``_DIMENSION_RE`` anchors each number on a word boundary, and ``x`` is a word
character, so in ``30x20x1.5 m`` only the LAST number can ever match: one
dimension, below the threshold, no calculation. The paraphrase writes
``30 m long, 20 m wide and 1.5 m thick`` -- three matches -- and fires. Same
question, same numbers, different notation.

WHAT THIS MATRIX IS. Every battery question, in BOTH forms, asserted to
reach the SAME routing verdict. It is deterministic: no model, no network,
no corpus. It cannot prove the answer is right -- only the re-battery can --
but it makes "works when you phrase it my way" a test failure.

WHAT IT IS NOT. C2's failure is in the same finding but not the same
mechanism: both of its forms route identically here, and the wrong document
came back from retrieval RANKING. That is a different class and is not
claimed to be fixed by this file.
"""

import json
from pathlib import Path

import pytest

from app.agents.runtime import (
    _count_dimensions,
    _looks_like_self_contained_calculation as routes_to_calc,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ui_phys"
CATALOG = json.loads((FIXTURES / "questions.json").read_text(encoding="utf-8"))
FORMS = json.loads((FIXTURES / "routing_forms.json").read_text(encoding="utf-8"))

ASKED = {cid: c["ask"] for cid, c in CATALOG["cases"].items() if c.get("ask", "").strip()}

#: The only battery question that is a self-contained calculation. Pinned so
#: the matrix cannot be satisfied by a regression that routes EVERYTHING to
#: retrieval -- both forms would then still "agree", and agreeing on the
#: wrong answer is the failure this whole item is about.
CALC_CASES = {"E4"}


# -- coverage --------------------------------------------------------------


def test_every_battery_question_has_both_forms():
    """The owner's wording: BOTH forms, every question. A matrix that
    silently skips a question is how the next E4 gets through.

    Mutation killed: dropping a case from routing_forms.json.
    """
    missing = sorted(set(ASKED) - set(FORMS["forms"]))
    assert missing == [], f"no second form for: {missing}"
    extra = sorted(set(FORMS["forms"]) - set(ASKED))
    assert extra == [], f"second forms for unknown ids: {extra}"


def test_the_second_form_is_actually_a_different_form():
    """Mutation killed: filling the fixture by copying each ask, which makes
    every row agree with itself and proves nothing."""
    for cid, ask in ASKED.items():
        alt = FORMS["forms"][cid]["alt"]
        assert alt.strip(), cid
        assert alt.strip().lower() != ask.strip().lower(), cid


def test_the_two_measured_paraphrases_are_the_recorded_ones():
    """E4 and C2 are not written for this file -- they are the paraphrases
    that were run on 87c7996 and recorded in the evidence pack. Marked so a
    reader can tell measured evidence from a fixture I authored."""
    assert FORMS["measured_from_evidence_pack"] == ["C2", "E4"]
    assert FORMS["forms"]["E4"]["origin"] == "measured"
    assert FORMS["forms"]["C2"]["origin"] == "measured"
    assert FORMS["forms"]["E4"]["alt"] == (
        "I'm pouring a raft 30 m long, 20 m wide and 1.5 m thick, how much "
        "concrete do I need including waste, and how many truck loads?"
    )
    assert FORMS["forms"]["C2"]["alt"] == (
        "Which specification section sets out the procedure for variations "
        "and adjustments?"
    )


# -- the matrix ------------------------------------------------------------


@pytest.mark.parametrize("cid", sorted(ASKED))
def test_both_forms_of_every_question_route_the_same_way(cid):
    ask, alt = ASKED[cid], FORMS["forms"][cid]["alt"]
    sheet, para = routes_to_calc(ask), routes_to_calc(alt)
    assert sheet == para, (
        f"{cid} routes differently by phrasing: sheet form -> "
        f"{'calc' if sheet else 'retrieval'}, second form -> "
        f"{'calc' if para else 'retrieval'}\n  sheet: {ask}\n  alt:   {alt}"
    )


@pytest.mark.parametrize("cid", sorted(ASKED))
def test_each_question_routes_where_it_should(cid):
    """Agreement is not enough. Mutation killed: a change that sends every
    question to retrieval -- both forms agree, E4 is broken again, and the
    matrix above stays green."""
    want = cid in CALC_CASES
    assert routes_to_calc(ASKED[cid]) is want, cid
    assert routes_to_calc(FORMS["forms"][cid]["alt"]) is want, cid


def test_e4_in_the_instrument_s_own_notation_reaches_the_calculator():
    """The regression itself, named. Before the compact-dimension fix this
    returned False and the platform answered E4 with a refusal while holding
    every number it needed."""
    assert routes_to_calc(
        "Concrete volume for a raft 30x20x1.5 m including your documented "
        "waste factor."
    )


# -- the counter -----------------------------------------------------------


@pytest.mark.parametrize("text,want", [
    ("30x20x1.5 m", 3),
    ("30×20×1.5 m", 3),
    ("30 x 20 x 1.5 m", 3),
    ("25m x 18m x 1.2m", 3),
    ("1200x600", 2),
    ("30 m long, 20 m wide and 1.5 m thick", 3),
    ("a 40 m culvert", 1),
    ("no numbers here at all", 0),
])
def test_a_dimension_chain_counts_every_number_in_it(text, want):
    """Mutation killed: counting a chain as one dimension (the original bug)
    or double-counting its trailing number, which would make a single
    "40x2 m" look like three."""
    assert _count_dimensions(text) == want, text


def test_a_document_lookup_with_a_product_size_is_still_a_lookup():
    """The fence on the fix. Compact dimensions appear in ordinary product
    descriptions; without a calc verb they are not a calculation.

    Mutation killed: dropping the calc-verb requirement now that chains
    count, which would send BOQ lookups to the calculator.
    """
    for q in (
        "What is the rate for 600x600 floor tiles?",
        "Which drawing shows the 1200x600 precast panels?",
        "What is the concrete volume in the BOQ?",
    ):
        assert not routes_to_calc(q), q


def test_the_kill_switch_still_disables_the_whole_path(monkeypatch):
    """FORCE_CALC_ON_DIMENSIONS=0 was the operator's escape hatch before this
    change and must remain one after it."""
    monkeypatch.setenv("FORCE_CALC_ON_DIMENSIONS", "0")
    assert not routes_to_calc(
        "Concrete volume for a raft 30x20x1.5 m including your documented "
        "waste factor."
    )
