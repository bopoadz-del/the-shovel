"""An answer's own arithmetic must agree with its own result.

Two live defects of the same shape, found a day apart and fixed one at a time:

    T20  "L/20 = 4800/20 = 200 mm"          -- 4800/20 is 240
    T12  "Ec = 4700 x 5.9161 = 28,062 MPa"  -- that product is 27,806

Both answers stated the right rule, the right inputs and a wrong number. Every
visible part was correct, so a reader who checks the working is *reassured* by
it — which makes this the most dangerous defect shape the platform produces.
Each needed its own fix (a unit conversion in the binder, a unit on a modulus
payload); this catches the class.

The mismatch is ANNOTATED, never silently corrected: rewriting the figure
would hide that the answer and its own working disagree, and that fact is
what the reader needs.
"""
import pytest

from app.agents import runtime as rt


def _mismatches(text):
    return rt.derivation_mismatches(text)


# ── the two live defects ───────────────────────────────────────────────────

def test_the_slab_defect_is_caught():
    found = _mismatches("Per Table 7.3.1.1 the minimum is L/20. For a 4.8 m span: 4800/20 = 200 mm.")
    assert len(found) == 1
    expr, stated, computed = found[0]
    assert expr == "4800/20"
    assert stated == pytest.approx(200.0)
    assert computed == pytest.approx(240.0)


def test_the_modulus_defect_is_caught():
    found = _mismatches("Ec = 4700 x sqrt(35) = 4700 x 5.9161 = 28,062 MPa")
    assert len(found) == 1
    assert found[0][1] == pytest.approx(28_062.0)
    assert found[0][2] == pytest.approx(27_805.67, abs=0.5)


# ── correct working passes, including the way answers round ────────────────

@pytest.mark.parametrize("text", [
    "4800 / 20 = 240 mm",
    "3.2 x 3.2 x 0.75 x 18 = 138.24 m3",
    "138.24 x 1.05 = 145.15 m3",          # 145.152 shown rounded
    "145.152 x 390 = 56,609.28",
    "1,754,504,456.25 x 0.00015 = 263,175.67",
    "0.613 x 1600 = 980.8 Pa",
])
def test_correct_working_is_not_flagged(text):
    assert _mismatches(text) == []


def test_a_rounded_result_inside_tolerance_agrees():
    # 0.5% or 0.01, whichever is larger -- the gate's own tolerance.
    assert _mismatches("7.68 x 18 = 138.2") == []
    assert _mismatches("7.68 x 18 = 130.0") != []


# ── it does not invent work for itself ─────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The contract is AB-2023-101 and the period is 365 days.",
    "Rates range 10 - 20 per m2.",
    "Clause 8.8.1 applies.",
    "",
])
def test_prose_without_an_equation_is_untouched(text):
    assert _mismatches(text) == []


def test_a_mixed_operator_chain_is_skipped():
    # "2 x 3 / 4" needs precedence rules the answer may not have followed;
    # guessing them would produce false accusations.
    assert _mismatches("2 x 3 / 4 = 99") == []


def test_division_by_zero_is_skipped_not_crashed():
    assert _mismatches("100 / 0 = 5") == []


# ── the annotation ─────────────────────────────────────────────────────────

def test_the_note_names_the_expression_and_both_numbers():
    out = rt._annotate_derivation_mismatches("The minimum is 4800/20 = 200 mm.")
    assert "4800/20" in out
    assert "240" in out
    assert "200" in out
    assert out.startswith("The minimum is"), "the original answer is preserved above the note"


def test_a_clean_answer_is_returned_unchanged():
    clean = "The minimum thickness is 4800/20 = 240 mm."
    assert rt._annotate_derivation_mismatches(clean) == clean


def test_the_figures_are_never_rewritten():
    out = rt._annotate_derivation_mismatches("Ec = 4700 x 5.9161 = 28,062 MPa")
    assert "28,062 MPa" in out, "the answer's own number must still be visible"


def test_the_note_is_bounded():
    many = " ".join(f"{i} x 2 = {i * 3}" for i in range(2, 12))
    out = rt._annotate_derivation_mismatches(many)
    assert out.count("\n- ") <= rt._DERIVATION_MAX_NOTES + 1
    assert "more." in out


def test_the_kill_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv("DERIVATION_CHECK", "0")
    text = "4800/20 = 200 mm"
    assert rt._annotate_derivation_mismatches(text) == text


def test_a_broken_check_never_breaks_the_turn(monkeypatch):
    monkeypatch.setattr(rt, "derivation_mismatches", lambda _t: 1 / 0)
    text = "4800/20 = 200 mm"
    assert rt._annotate_derivation_mismatches(text) == text
