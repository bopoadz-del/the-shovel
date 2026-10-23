"""The fy modification chain is not a disagreement with its own addend.

Live on 23a04d0: ACI one-way slab answers from slab_thickness_min showed
correct working

    0.4 + 420/700 = 0.4 + 0.600 = 1.000

and the post-answer checker still appended "420/700 is 0.6, not 0.4"
(or "not 1"). The quotient was paired with the neighbouring addend, or
with the total of the whole sum. Each equality step has to be checked
against its own left side.

Wrong arithmetic in the same shape must still be flagged.
"""
import pytest

from app.agents import runtime as rt


_SPAN_WORKING = (
    "h_min = L/20 × (0.4 + 420/700) = 4800/20 × 1.000 = 240 mm"
)

_SLAB_PARAGRAPH = (
    "Minimum one-way slab thickness for a simply supported span of 4.8 m "
    "with fy = 420 MPa, per ACI 318-19 Table 7.3.1.1. "
    "The modification factor is 0.4 + 420/700 = 0.4 + 0.600 = 1.000. "
    f"{_SPAN_WORKING}."
)


def _annotated(text):
    return rt._annotate_derivation_mismatches(text)


# ── (a) correct working must not be accused ───────────────────────────────

@pytest.mark.parametrize("text", [
    "0.4 + 420/700 = 0.4 + 0.600 = 1.000",
    "0.4 + fy/700 = 0.4 + 420/700 = 1.00",
    "factor = 0.4 + 420/700 = 1.0",
    "420/700 = 0.6",
    _SLAB_PARAGRAPH,
])
def test_correct_fy_modification_is_not_accused(text):
    if text is _SLAB_PARAGRAPH:
        assert _SPAN_WORKING in text
    assert rt.derivation_mismatches(text) == []
    assert "working and the result disagree" not in _annotated(text)
    assert _annotated(text) == text


# ── (b) genuinely wrong arithmetic is still flagged ───────────────────────

@pytest.mark.parametrize("text", [
    "420/700 = 0.4",
    "0.4 + 420/700 = 0.4 + 0.500 = 0.900",
    "4800/20 = 200",
    "0.4 + 420/700 = 1.2",
])
def test_wrong_arithmetic_is_still_flagged(text):
    assert rt.derivation_mismatches(text), text
    assert "working and the result disagree" in _annotated(text)


def test_the_span_working_line_is_not_accused():
    assert _SPAN_WORKING in _SLAB_PARAGRAPH
    assert rt.derivation_mismatches(_SPAN_WORKING) == []
    assert _annotated(_SPAN_WORKING) == _SPAN_WORKING


def test_a_wrong_step_is_judged_against_its_own_left_side():
    found = rt.derivation_mismatches("0.4 + 420/700 = 0.4 + 0.500 = 0.900")
    assert len(found) == 1
    expr, stated, computed = found[0]
    assert expr == "0.4 + 420/700"
    assert stated == pytest.approx(0.9)
    assert computed == pytest.approx(1.0)
    note = _annotated("0.4 + 420/700 = 0.4 + 0.500 = 0.900")
    assert "`0.4 + 420/700`" in note
    assert "not 0.4" not in note


def test_a_wrong_total_is_not_blamed_on_the_quotient():
    found = rt.derivation_mismatches("0.4 + 420/700 = 1.2")
    assert len(found) == 1
    expr, stated, computed = found[0]
    assert expr == "0.4 + 420/700"
    assert stated == pytest.approx(1.2)
    assert computed == pytest.approx(1.0)


# ── live defects this checker was built for ───────────────────────────────
# T20 and T12 must still fire. A correct 420/700 inside the fy sum must not.

def test_t20_4800_over_20_equals_200_mm_is_flagged():
    text = "Per Table 7.3.1.1 the minimum is L/20. For a 4.8 m span: 4800/20 = 200 mm."
    assert "4800/20 = 200 mm" in text
    found = rt.derivation_mismatches(text)
    assert len(found) == 1
    expr, stated, computed = found[0]
    assert expr == "4800/20"
    assert stated == pytest.approx(200.0)
    assert computed == pytest.approx(240.0)
    assert "working and the result disagree" in _annotated(text)


@pytest.mark.parametrize("op", ["x", "×", "*"])
def test_t12_4700_times_59161_equals_28062_is_flagged(op):
    text = f"Ec = 4700 {op} 5.9161 = 28,062 MPa"
    found = rt.derivation_mismatches(text)
    assert len(found) == 1, text
    expr, stated, computed = found[0]
    assert expr == f"4700 {op} 5.9161"
    assert stated == pytest.approx(28_062.0)
    assert computed == pytest.approx(27_805.67, abs=0.5)
    assert "working and the result disagree" in _annotated(text)


def test_fy700_quotient_equal_to_0_6_inside_the_sum_is_not_flagged():
    text = "0.4 + 420/700 = 0.4 + 0.600 = 1.000"
    assert rt.derivation_mismatches(text) == []
    annotated = _annotated(text)
    assert "working and the result disagree" not in annotated
    assert "420/700" not in annotated[len(text):]
    assert annotated == text
