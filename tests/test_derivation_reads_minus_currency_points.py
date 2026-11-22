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


def test_mixed_signs_fold_left_with_products_inside():
    assert _found("620 + 3 x 40 - 90 = 650") == []
    found = _found("620 + 3 x 40 - 90 = 830")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(830), pytest.approx(650))]


def test_a_difference_carries_into_the_next_side_as_a_sum():
    # The side after a difference may itself be a sum; it is read whole.
    assert _found("900 - 150 = 600 + 150 = 750") == []
    found = _found("900 - 150 = 600 + 150 = 760")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(760), pytest.approx(750))]


@pytest.mark.parametrize("text", [
    "Grades range 30 - 45 by element.",
    "Pages 12 – 18 = the full schedule",   # an en dash is a range, not a minus
    "Reference DX-4410-07 applies.",
])
def test_ranges_and_references_are_not_working(text):
    assert _found(text) == []


def test_an_en_dash_is_never_read_as_a_minus():
    # Read as a minus this would be 18 - 12 = 6 (agrees) or accuse "= 9".
    assert _found("18 – 12 = 9 bays") == []
    assert _found("18 - 12 = 9 bays") != []


# ── 2. a currency code or sign before the figure ────────────────────────────

@pytest.mark.parametrize("cur", ["QAR ", "EUR", "$", "US$ ", "€"])
def test_a_result_after_a_currency_is_compared(cur):
    assert _found(f"3 x 2,400 x 1.25 = {cur}9,000") == []
    found = _found(f"3 x 2,400 x 1.25 = {cur}6,000")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(6000), pytest.approx(9000))]


def test_an_addend_after_a_currency_is_read():
    assert _found("QAR 2,300 + QAR 1,150 = QAR 3,450") == []
    found = _found("QAR 2,300 + QAR 1,150 = QAR 3,750")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(3750), pytest.approx(3450))]


def test_lower_case_words_after_equals_still_end_the_reading():
    # "= about 9,000" is prose, not a currency; nothing is compared.
    assert _found("3 x 2,400 x 1.25 = about 6,000") == []
    assert _found("3 x 2,400 x 1.25 = 6,000") != []


# ── 3. a figure that ends a sentence ────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The frame takes 46.50 x 684 = 31,806.",
    "Total 7.25 x 1,290 = 9,352.50.",
    "Area 12 x 15,000 = 180,000. Next line.",
])
def test_a_correct_figure_before_a_full_stop_is_not_accused(text):
    assert _found(text) == []


def test_a_wrong_figure_before_a_full_stop_is_named_whole():
    found = _found("The frame takes 46.50 x 684 = 33,806.")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(33_806), pytest.approx(31_806))]
    found = _found("Total 7.25 x 1,290 = 9,952.50, all in.")
    assert [(f[1], f[2]) for f in found] == [(pytest.approx(9_952.5), pytest.approx(9_352.5))]


def test_a_figure_followed_by_another_digit_group_is_still_refused():
    # "1.2.3" is not one figure; nothing is read rather than a fragment.
    assert _found("4 x 3 = 1.2.3") == []
    assert _found("4 x 3 = 1.2") != []


# ── 4. a percent sign on a lone result ──────────────────────────────────────

def test_a_sum_of_points_stated_as_a_percent_is_not_accused():
    assert _found("(9/10 x 25) + (5/10 x 35) + (4/10 x 40) = 22.5 + 17.5 + 16 = 56.0 %") == []
    assert _found("22.5 + 17.5 + 16 = 56 %") == []


def test_a_sum_of_points_with_a_wrong_percent_is_named():
    found = _found("22.5 + 17.5 + 16 = 61.0 %")
    assert len(found) == 1
    assert found[0][2] == pytest.approx(56.0)


def test_a_ratio_stated_as_a_percent_still_reads_hundredths():
    assert _found("27/36 = 75%") == []
    assert _found("27/36 = 80%") != []


def test_a_percent_factor_on_the_left_fixes_hundredths():
    # The left side is in hundredths, so "= 12 %" cannot mean 12 points.
    assert _found("4% x 300 = 12") == []
    assert _found("4% x 300 = 12 %") != []


# ── the note on the answer ──────────────────────────────────────────────────

def test_the_note_names_a_wrong_net_and_keeps_the_answer():
    text = "Net due: 7,400,000 − 370,000 − 740,000 − 2,100,000 = QAR 6,290,000"
    out = rt._annotate_derivation_mismatches(text)
    assert out.startswith(text)
    assert "working and the result disagree" in out
    assert "4,190,000" in out.replace("4.19e+06", "4,190,000")


def test_a_correct_net_with_a_currency_leaves_the_answer_byte_for_byte():
    text = "Net due: 7,400,000 − 370,000 − 740,000 = QAR 6,290,000."
    assert rt._annotate_derivation_mismatches(text) == text
