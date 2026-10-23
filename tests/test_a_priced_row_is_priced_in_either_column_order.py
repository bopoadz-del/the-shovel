"""Rate and quantity on the row means there is an amount. In either order.

Live d8d9573, master_corpus, 0/5:

    "What is the amount for removal of existing chain link fence (D549.2)?"

    -> "the chain link fence item (D549.2) is **not priced** in the sources
        shown -- it appears as a rate-only or excluded item"

The row was retrieved at rank 1, and it reads:

    G | Breakout and remove existing chain link fence  D549.2  m  3,504
        80.00  280,320.00
    H | Breakout and remove existing storm water culverts  D529.3  m
        1,370.00  Rate Only

Owner's ruling on this exact item: "rate only is when you have no quantity;
if you have rate and quantity, you must have an amount."

A deterministic composer exists to state such a row without asking the model
to read it (WAVE 2 B5). It parsed nothing, so the model read the row itself
and attached the NEIGHBOUR's "Rate Only" to the fence. The parser expected

    <qty> <unit> <rate> <amount>        3,504 m 80.00 280,320.00

and a CESMM bill prints its columns  Ref | Unit | Qty | Rate | Amount:

    <unit> <qty> <rate> <amount>        m 3,504 80.00 280,320.00

Accepting the second order is safe for the same reason the first is: a triple
only counts when qty x rate equals the printed amount.
"""
from __future__ import annotations

import pytest

from app.core.rag.retriever import (
    chunk_states_priced_item,
    chunk_states_rate_only_item,
    compose_priced_boq_row,
)

ASK = "What is the amount for removal of existing chain link fence (D549.2)?"
CODES = ["d549.2"]

# Row shape exactly as indexed; item figures kept (they are arithmetic, not
# identity), party names never present.
UNIT_FIRST = (
    "119 335.00 39,865.00} F |Breakout and remove existing metal beam guard rail "
    "D549.1 m240.00 Rate Only G |Breakout and remove existing chain link fence "
    "D549.2 m 3,504 80.00 280,320.00 H_ |Breakout and remove existing storm water "
    "culverts D529.3 m 1,370.00 Rate Only} fl stamp noise"
)
EXCLUDED_ELSEWHERE = (
    "r 74 4,000 296,000.00 G_ |Breakout and remove existing metal beam guard rail "
    "D 549.1 sum 1 Excluded H_ {Breakout and remove existing chain link fence "
    "D 549.2 | sum 1 Excluded J |Breakout and remove existing storm water culverts "
    "D599.5 | sum 1 Excluded"
)
SCOPE_LIST = (
    "D 549.1 - Breakout and remove existing metal beam guard rail D 549.2 - "
    "Breakout and remove existing chain link fence D 599.5 - Breakout and remove"
)


def test_the_live_row_is_priced():
    assert chunk_states_priced_item(UNIT_FIRST, CODES)


def test_the_neighbours_rate_only_does_not_rub_off():
    """D549.1 above it and D529.3 below it ARE Rate Only. D549.2 is not."""
    assert not chunk_states_rate_only_item(UNIT_FIRST, CODES)
    assert chunk_states_rate_only_item(UNIT_FIRST, ["d549.1"])
    assert chunk_states_rate_only_item(UNIT_FIRST, ["d529.3"])


def test_compose_states_quantity_rate_and_amount():
    row = compose_priced_boq_row(ASK, UNIT_FIRST)
    assert row, "the composer parsed nothing from a fully priced row"
    assert (row["qty"], row["rate"], row["amount"]) == (3504.0, 80.0, 280320.0)
    assert row["unit"].lower() == "m"
    assert row["code"].lower() == "d549.2"


@pytest.mark.parametrize("order", ["priced_first", "excluded_first"])
def test_the_priced_row_wins_over_an_excluded_sibling_in_another_bill(order):
    """Live pool: the same code is "sum 1 Excluded" in a different bill
    section. Whichever arrives first, the row that is priced is the answer."""
    parts = [UNIT_FIRST, SCOPE_LIST, EXCLUDED_ELSEWHERE]
    if order == "excluded_first":
        parts.reverse()
    row = compose_priced_boq_row(ASK, "\n\n".join(parts))
    assert row and row["amount"] == 280320.0


@pytest.mark.parametrize(
    "row",
    [
        "D549.2 m 3,504 80.00 280,320.00",          # unit qty rate amount
        "D549.2 3,504 m 80.00 280,320.00",          # qty unit rate amount (old)
        "D549.2 m3,504 80.00 280,320.00",           # OCR glued the unit
        "D549.2 | m | 3,504 | 80.00 | 280,320.00",  # table pipes
        "D549.2 m2 1,000 12.50 12,500.00",          # another unit
        "D549.2 Nr 48 220.00 10,560.00",
    ],
)
def test_every_column_order_and_ocr_variant_parses(row):
    assert chunk_states_priced_item("Breakout fence " + row, CODES), row


@pytest.mark.parametrize(
    "row",
    [
        # The arithmetic is the guard: these are not priced rows.
        "D549.2 m 3,504 80.00 999,999.00",    # amount does not equal qty x rate
        "D549.2 m 1,370.00 Rate Only",        # a rate and no quantity
        "D549.2 m 3,504 Rate Only",
        "D549.2 sum 1 Excluded",
        "D549.2 - Breakout and remove existing chain link fence",
    ],
)
def test_a_row_that_is_not_priced_is_never_made_priced(row):
    assert not chunk_states_priced_item("Breakout fence " + row, CODES), row
    assert compose_priced_boq_row(ASK, "Breakout fence " + row) is None


# ── one item code, two bills, two different items ─────────────────────────
#
# Live 5312551, and caused by the fix above. Until the parser could read
# unit-first rows it could not read THIS row either, which hid the flaw:
#
#   priced BOQ:      ...existing concrete wall/barrier  D 529.3  m  26,997  500  13,498,500.00
#   demolition bill: ...existing storm water culverts   D529.3   m  1,370.00  Rate Only
#
# Asked "What is the total amount for removal of storm water culverts
# (D529.3)?", the platform answered "D529.3: quantity 26,997 m @ 500.00 =
# 13,498,500" -- the other bill's wall, stated as the culverts' total. The
# correct answer is that the item is Rate Only and has no amount.
#
# The code alone does not identify the item. The question's own description
# does, and a priced row whose description shares nothing with it is a
# different item that happens to carry the same number.

G4 = "What is the total amount for removal of storm water culverts (D529.3)?"
OTHER_BILLS_WALL = (
    "and concrete structures D529.2 | sum 1 Excluded D_ {Breakout and remove "
    "existing concrete wall/barrier D 529.3 m 26,997 500 13,498,500.00 E "
    "|Breakout and remove existing walkway D 599.1 sum 1"
)
THE_CULVERTS = (
    "G |Breakout and remove existing chain link fence D549.2 m 3,504 80.00 "
    "280,320.00 H_ |Breakout and remove existing storm water culverts D529.3 m "
    "1,370.00 Rate Only} stamp"
)


def test_a_priced_row_for_a_different_item_is_not_the_answer():
    assert compose_priced_boq_row(G4, OTHER_BILLS_WALL) is None


def test_with_both_bills_in_the_excerpt_the_culverts_stay_rate_only():
    both = OTHER_BILLS_WALL + "\n\n" + THE_CULVERTS
    assert compose_priced_boq_row(G4, both) is None
    assert chunk_states_rate_only_item(both, ["d529.3"])
    assert not chunk_states_priced_item(both, ["d529.3"], query=G4)


def test_the_same_row_answers_the_question_that_describes_it():
    ask = "What is the amount for breaking out the existing concrete wall/barrier (D529.3)?"
    row = compose_priced_boq_row(ask, OTHER_BILLS_WALL + "\n\n" + THE_CULVERTS)
    assert row and row["amount"] == 13498500.0


def test_a_question_that_gives_only_the_code_still_gets_the_priced_row():
    """No description to check against: the code is all there is."""
    row = compose_priced_boq_row("What is the amount for D529.3?", OTHER_BILLS_WALL)
    assert row and row["amount"] == 13498500.0


def test_plural_and_wording_differences_do_not_break_the_match():
    """The bill says "culverts", "fence"; people say "culvert", "fencing"."""
    ask = "What is the amount for the chain-link fencing removal (D549.2)?"
    assert compose_priced_boq_row(ask, UNIT_FIRST)["amount"] == 280320.0


# ── a check is not a lookup ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "question",
    [
        "Verify: does 158 ha at SAR 186,328/ha equal the stated D110 amount?",
        "Verify: does 34,844 m at SAR 142.00/m equal the stated D529.2 amount?",
        "Check whether the D549.2 amount is consistent with its rate and quantity.",
        "Is the D549.2 amount larger than the D599.6 amount?",
    ],
)
def test_a_verify_question_is_not_answered_by_restating_the_row(question):
    """Live 5312551: both "Verify:" questions got the bare row back -- the
    figures, and no verdict. The expected answer says yes or no."""
    from app.core.rag.retriever import query_is_a_check_not_a_lookup

    assert query_is_a_check_not_a_lookup(question), question


def test_a_plain_amount_question_is_a_lookup():
    from app.core.rag.retriever import query_is_a_check_not_a_lookup

    assert not query_is_a_check_not_a_lookup(ASK)
    assert not query_is_a_check_not_a_lookup(G4)


def test_the_runtime_shortcut_steps_aside_for_a_check_and_not_for_a_lookup():
    from app.agents.runtime import _should_short_circuit_priced_boq

    rag = {"role": "system", "content": "Project excerpts:\n" + UNIT_FIRST}
    lookup = [{"role": "user", "content": ASK}]
    check = [{"role": "user", "content":
              "Verify: does 3,504 m at SAR 80.00/m equal the stated D549.2 amount?"}]

    assert "280,320" in _should_short_circuit_priced_boq(rag, lookup, has_predispatch=False)
    assert _should_short_circuit_priced_boq(rag, check, has_predispatch=False) == ""


# ── the description is the words right before the code ────────────────────

def test_words_that_are_in_no_row_are_not_a_description_of_another_item():
    """"according to the bill" matches nothing. That is no description, not a
    description of something else -- the code is all there is to go on."""
    ask = "What is the amount of D549.2 according to the tender bill?"
    assert compose_priced_boq_row(ask, UNIT_FIRST)["amount"] == 280320.0


def test_a_verify_question_still_composes_so_the_guard_above_is_what_stops_it():
    """"verify", "equal" are in no row either; the composer must not refuse
    by accident, or the explicit check/lookup guard is never exercised."""
    ask = "Verify: does 3,504 m at SAR 80.00/m equal the stated D549.2 amount?"
    assert compose_priced_boq_row(ask, UNIT_FIRST)["amount"] == 280320.0


def test_fence_and_fencing_are_the_same_word():
    from app.core.rag.retriever import _same_word

    assert _same_word("fencing", "fence") and _same_word("culvert", "culverts")
    assert not _same_word("wall", "walkway") and not _same_word("rail", "railing's")
    # ...and it matters: "fencing" is the ONLY describing word here, and the
    # excerpt also holds another item under the same code.
    other = "Supply of welded mesh panels D549.2 m 900 10.00 9,000.00"
    ask = "What is the amount for fencing (D549.2)?"
    assert compose_priced_boq_row(ask, other + "\n\n" + UNIT_FIRST)["amount"] == 280320.0


def test_a_code_is_not_paired_with_its_neighbours_description():
    """The row ABOVE D549.2 is the guard rail. Its words must not be borrowed:
    asked for a guard rail under D549.2 while a real guard-rail row sits
    elsewhere under that code, the fence row is not the answer."""
    elsewhere = "Supply metal beam guard rail D549.2 m 10 5.00 50.00"
    ask = "What is the amount for the metal beam guard rail (D549.2)?"
    row = compose_priced_boq_row(ask, UNIT_FIRST + "\n\n" + elsewhere)
    assert row and row["amount"] == 50.0


def test_one_chunk_is_enough_to_tell_it_is_the_wrong_item():
    """No pool needed: the wall row, alone, is still not the culverts."""
    assert not chunk_states_priced_item(OTHER_BILLS_WALL, ["d529.3"], query=G4)
    assert chunk_states_priced_item(OTHER_BILLS_WALL, ["d529.3"])  # no question, no check


def test_an_earlier_clause_sets_the_scene_it_does_not_describe_the_item():
    ask = "For the boundary wall package handover: what is the amount for D549.2?"
    assert compose_priced_boq_row(ask, UNIT_FIRST)["amount"] == 280320.0


# ── two bills that disagree about the SAME item ───────────────────────────
#
# Unseen Set 3, live 5312551: "How many street lighting poles are to be
# removed and what is the amount (D999.1)?" The top two excerpts were
#
#   priced BOQ:       ...street lighting poles...  D 999.1  Nr  915  3,800     3,477,000.00
#   demolition bill:  ...street lighting poles...  D999.1   Nr  897  1,275.00  1,143,675.00
#
# Same item, same code, two documents, two answers -- and the platform stated
# the first one flatly. A deterministic one-line answer is only honest when
# there is one answer. When the sources disagree the turn goes to the model,
# with both excerpts in front of it.

POLES_PRICED_BOQ = (
    "C_ |Breakout and remove existing street lighting poles and luminaires "
    "including feeder pillars D 999.1 Nr 915 3,800 3,477,000.00 D_ {Breakout"
)
POLES_DEMOLITION = (
    "C Breakout and remove existing street lighting poles and luminaires "
    "including feeder pillars D999.1 Nr 897 1,275.00 1,143,675.00 D Breakout"
)
POLES_ASK = "How many street lighting poles are to be removed and what is the amount (D999.1)?"


def test_two_sources_that_disagree_are_not_settled_by_picking_the_first():
    both = POLES_PRICED_BOQ + "\n\n" + POLES_DEMOLITION
    assert compose_priced_boq_row(POLES_ASK, both) is None
    assert compose_priced_boq_row(POLES_ASK, POLES_DEMOLITION + "\n\n" + POLES_PRICED_BOQ) is None


def test_the_same_row_quoted_twice_is_not_a_disagreement():
    """Two copies of one bill (signed and unsigned, or an OCR of it) agree."""
    twice = POLES_DEMOLITION + "\n\n" + POLES_DEMOLITION.replace("C Breakout", "C |Breakout")
    assert compose_priced_boq_row(POLES_ASK, twice)["amount"] == 1143675.0


def test_one_source_is_still_answered_directly():
    assert compose_priced_boq_row(POLES_ASK, POLES_DEMOLITION)["amount"] == 1143675.0


# ── the graft must not delete a verdict ───────────────────────────────────
#
# Unseen Set 3 E4, live 5011b62 -- with the shortcut guard above already
# deployed: "Verify: does 34,844 m at SAR 142.00/m equal the stated D529.2
# amount?" still came back as the bare row. A second path: the post-answer
# graft REPLACES any answer under 500 characters that does not restate the
# row in its own format. A verdict is short. "Yes -- it matches exactly" is
# the whole answer to a check, and the graft threw it away.

VERIFY = "Verify: does 3,504 m at SAR 80.00/m equal the stated D549.2 amount?"


def _graft(answer, ask):
    from app.agents.runtime import _graft_priced_boq_item

    rag = {"role": "system", "content": "Project excerpts:\n" + UNIT_FIRST}
    return _graft_priced_boq_item(answer, rag, [{"role": "user", "content": ask}])


@pytest.mark.parametrize(
    "verdict",
    [
        "Yes. 3,504 x 80.00 = 280,320.00, which matches the stated amount exactly.",
        "Yes, it matches.",
        "No -- the bill states Rate Only in one copy and Excluded in another, so the "
        "figures cannot be confirmed from a single source.",
    ],
)
def test_a_verdict_survives_the_graft(verdict):
    out = _graft(verdict, VERIFY)
    assert verdict in out, out


def test_a_check_with_no_answer_at_all_still_gets_the_row():
    assert "280,320" in _graft("", VERIFY)


def test_a_lookup_is_still_grafted_as_before():
    """The B5 behaviour this graft exists for: a short non-answer to a plain
    lookup is replaced by the row."""
    out = _graft("It appears as a rate-only or excluded item.", ASK)
    assert "280,320" in out and "rate-only or excluded" not in out
