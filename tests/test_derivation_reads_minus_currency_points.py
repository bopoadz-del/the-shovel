"""The derivation check reads the working an answer actually prints.

Four gaps in ``derivation_mismatches`` (app/agents/runtime.py), each with
its own inputs. Every input here is invented; none is taken from a document,
a probe or a live answer.

1. A sum that takes deductions away ("a − b − c = X") was never read: only
   "+" joined addends, so a net figure that is not its own working passed.
2. A currency code or sign between "=" and the figure ("= QAR 4,100") ended
   the reading, so the result was never compared.
3. A figure that ends a sentence ("= 31,815.") was read as the digits before
   its first comma (31): the end-of-figure rule refused the full stop and the
   reader backtracked, then accused correct working.
4. A percent sign on a lone result was always hundredths, so "= 57.0 %" after
   a sum of points was accused of being 0.57.

Each decision has two inputs: one where the printed result is the working's
(no note) and one where it is not (a note).
"""

from __future__ import annotations

import pytest

from app.agents import runtime as rt


def _found(text: str):
    return rt.derivation_mismatches(text)


# ── 1. subtraction is read ──────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["-", "−"])
def test_a_net_that_is_its_own_working_is_not_accused(op):
    assert _found(f"Net: 7,400,000 {op} 370,000 {op} 740,000 = 6,290,000") == []
    assert _found(f"Balance 5,250 {op} 1,125 = 4,125 m") == []


@pytest.mark.parametrize("op", ["-", "−"])
def test_a_net_that_is_not_its_own_working_is_named(op):
    found = _found(f"Net: 7,400,000 {op} 370,000 {op} 740,000 {op} 2,100,000 = 6,290,000")
    assert len(found) == 1
    expr, stated, computed = found[0]
    assert expr == f"7,400,000 {op} 370,000 {op} 740,000 {op} 2,100,000"
    assert stated == pytest.approx(6_290_000)
    assert computed == pytest.approx(4_190_000)
    found = _found(f"Balance 5,250 {op} 1,125 = 4,425 m")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(4425), pytest.approx(4125))]


# Each case: (printed working, the stated result read, the computed result)
# or None when the working is the result's own and nothing is named.
_FOLDS = [
    ("620 + 3 x 40 - 90 = 650", None),
    ("620 + 3 x 40 - 90 = 830", (830, 650)),
    ("1,000 - 4 x 25 + 60 = 960", None),
    ("1,000 - 4 x 25 + 60 = 840", (840, 960)),
    # the side after a difference may itself be a sum; it is read whole
    ("900 - 150 = 600 + 150 = 750", None),
    ("900 - 150 = 600 + 150 = 760", (760, 750)),
]


@pytest.mark.parametrize(("text", "named"), _FOLDS)
def test_signs_fold_left_with_products_inside(text, named):
    found = [(f[1], f[2]) for f in _found(text)]
    assert found == ([] if named is None else [(pytest.approx(named[0]), pytest.approx(named[1]))])


@pytest.mark.parametrize("text", [
    "Grades range 30 - 45 by element.",
    "Pages 12 – 18 = the full schedule",   # an en dash is a range, not a minus
    "Reference DX-4410-07 applies.",
])
def test_ranges_and_references_are_not_working(text):
    assert _found(text) == []


@pytest.mark.parametrize(("text", "accused"), [
    # Read as a minus an en dash would compare 18 - 12 = 6 with 9.
    ("18 – 12 = 9 bays", False),
    ("40 – 25 = 3 zones", False),
    ("18 - 12 = 9 bays", True),
    ("40 − 25 = 3 zones", True),
])
def test_an_en_dash_is_never_read_as_a_minus(text, accused):
    assert bool(_found(text)) is accused


# ── 2. a currency code or sign before the figure ────────────────────────────

@pytest.mark.parametrize("cur", ["QAR ", "EUR", "$", "US$ ", "€"])
def test_a_result_after_a_currency_is_compared(cur):
    assert _found(f"3 x 2,400 x 1.25 = {cur}9,000") == []
    found = _found(f"3 x 2,400 x 1.25 = {cur}6,000")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(6000), pytest.approx(9000))]


@pytest.mark.parametrize(("text", "named"), [
    ("QAR 2,300 + QAR 1,150 = QAR 3,450", None),
    ("QAR 2,300 + QAR 1,150 = QAR 3,750", (3750, 3450)),
    ("$ 410 - $ 60 = $ 350", None),
    ("$ 410 - $ 60 = $ 380", (380, 350)),
])
def test_an_addend_after_a_currency_is_read(text, named):
    found = [(f[1], f[2]) for f in _found(text)]
    assert found == ([] if named is None else [(pytest.approx(named[0]), pytest.approx(named[1]))])


@pytest.mark.parametrize(("text", "accused"), [
    # lower-case words after "=" are prose, not a currency: nothing is read
    ("3 x 2,400 x 1.25 = about 6,000", False),
    ("8 x 125 = roughly 900", False),
    ("3 x 2,400 x 1.25 = 6,000", True),
    ("8 x 125 = 900", True),
])
def test_lower_case_words_after_equals_still_end_the_reading(text, accused):
    assert bool(_found(text)) is accused


# ── 3. a figure that ends a sentence ────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The frame takes 46.50 x 684 = 31,806.",
    "Total 7.25 x 1,290 = 9,352.50.",
    "Area 12 x 15,000 = 180,000. Next line.",
])
def test_a_correct_figure_before_a_full_stop_is_not_accused(text):
    assert _found(text) == []


@pytest.mark.parametrize(("text", "stated", "computed"), [
    ("The frame takes 46.50 x 684 = 33,806.", 33_806, 31_806),
    ("Total 7.25 x 1,290 = 9,952.50, all in.", 9_952.5, 9_352.5),
])
def test_a_wrong_figure_before_a_full_stop_is_named_whole(text, stated, computed):
    found = [(f[1], f[2]) for f in _found(text)]
    assert found == [(pytest.approx(stated), pytest.approx(computed))]


@pytest.mark.parametrize(("text", "accused"), [
    # "1.2.3" is not one figure; nothing is read rather than a fragment
    ("4 x 3 = 1.2.3", False),
    ("6 x 5 = 3,0.1", False),
    ("4 x 3 = 1.2", True),
    ("6 x 5 = 3.0", True),
])
def test_a_figure_followed_by_another_digit_group_is_still_refused(text, accused):
    assert bool(_found(text)) is accused


# ── 4. a percent sign on a lone result ──────────────────────────────────────

@pytest.mark.parametrize(("text", "named"), [
    # points: a sum of weighted scores stated as a percent
    ("(9/10 x 25) + (5/10 x 35) + (4/10 x 40) = 22.5 + 17.5 + 16 = 56.0 %", None),
    ("22.5 + 17.5 + 16 = 56 %", None),
    ("22.5 + 17.5 + 16 = 61.0 %", 56.0),
    ("30 + 12.5 = 45 %", 42.5),
    # hundredths: a ratio stated as a percent
    ("27/36 = 75%", None),
    ("27/36 = 80%", 0.75),
])
def test_a_lone_percent_result_agrees_under_either_reading(text, named):
    found = _found(text)
    if named is None:
        assert found == []
    else:
        assert len(found) == 1 and found[0][2] == pytest.approx(named)


@pytest.mark.parametrize(("text", "accused"), [
    # a percent factor on the left fixes hundredths: "= 12 %" is not 12 points
    ("4% x 300 = 12", False),
    ("5% x 260 = 13", False),
    ("4% x 300 = 12 %", True),
    ("5% x 260 = 13%", True),
])
def test_a_percent_factor_on_the_left_fixes_hundredths(text, accused):
    assert bool(_found(text)) is accused


# ── the note on the answer ──────────────────────────────────────────────────

@pytest.mark.parametrize(("text", "noted"), [
    ("Net due: 7,400,000 − 370,000 − 740,000 − 2,100,000 = QAR 6,290,000", True),
    ("Crew cost: 2 x 1,950 x 1.4 = EUR 3,900.", True),
    ("Net due: 7,400,000 − 370,000 − 740,000 = QAR 6,290,000.", False),
    ("Crew cost: 2 x 1,950 x 1.4 = EUR 5,460.", False),
])
def test_the_note_follows_the_arithmetic_and_keeps_the_answer(text, noted):
    out = rt._annotate_derivation_mismatches(text)
    assert out.startswith(text)
    assert ("working and the result disagree" in out) is noted
    if not noted:
        assert out == text


# ── a start that is not a figure is not a chain ─────────────────────────────

@pytest.mark.parametrize(("text", "start"), [
    ("Net: about 12 x 3 = 36", 5),
    ("= QAR 9 x 2 = 18", 2),
])
def test_a_chain_start_that_is_not_a_figure_reads_nothing(text, start):
    out: list = []
    assert rt._derivation_chain_mismatches(text, start, out) is None
    assert out == []
