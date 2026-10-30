"""The waste percentage the operator stated must beat the documented default.

The concrete calculator carries a project-documented waste factor of 5%, and
applies it when an ask does not name one -- that is deliberate and E4 depends
on it. But ``resolve_concrete_volume_calc`` substituted that 5% in two cases
where the operator HAD named a figure:

1. ``waste_factor`` was not bound from the text, so it arrived empty and the
   default filled it -- while "Add 7% waste" sat in the very blob being read.
2. The blob matched the documented-waste phrase, which overwrote an explicit
   ``waste_factor`` that was correctly bound.

Live SET5 E6, measured 6 times on dd4323e: three runs correct, one run
answered **189.0 m3** -- that is 180 x 1.05 -- and then carried it to SAR
84,936.60. The operator asked for 7%. Nothing in the answer said 5% had been
used instead, which makes this the dangerous shape again: a confident figure,
correct-looking working, wrong rate.

The documented default is a fallback for silence, never an override.
"""
import pytest

import app.lib.construction_formulas_quantities as q

ASK = "Add 7% waste to that total and price it at SAR 420 per cubic metre."


def _resolved(text, params=None):
    _calc, out = q.resolve_concrete_volume_calc("concrete_volume", params or {}, text)
    return out


# ── the live defect ────────────────────────────────────────────────────────

def test_a_percentage_in_the_text_is_used_not_the_default():
    assert _resolved(f"How much concrete for 24 pile caps? {ASK}")["waste_factor"] == \
        pytest.approx(0.07)


def test_the_documented_phrase_does_not_overwrite_a_bound_factor():
    # Path 2: the phrase matched and clobbered a correctly bound 7%.
    out = _resolved("Concrete volume for a raft 30x20x1.5 m with a waste factor of 7%.",
                    {"waste_factor": 0.07})
    assert out["waste_factor"] == pytest.approx(0.07)


def test_the_live_wrong_answer_is_not_reachable():
    # 180 x 1.05 = 189.0 was the figure the operator was given.
    assert _resolved(ASK)["waste_factor"] != pytest.approx(0.05)


# ── the phrasings an ask actually uses ─────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Add 7% waste to that total.", 0.07),
    ("add 7 % waste", 0.07),
    ("including 10% waste", 0.10),
    ("with a waste factor of 12%", 0.12),
    ("waste factor 2.5%", 0.025),
    ("apply waste of 7.5 percent", 0.075),
    ("allow 15% for waste", 0.15),
])
def test_the_stated_percentage_is_read(text, expected):
    assert q.waste_factor_from_text(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "Retention is 5% and advance payment recovery 15%.",
    "The contract allows 10% for overheads.",
    "Compaction to 95% modified Proctor.",
    "How much concrete for 24 pile caps, each 2.5 m by 2.5 m by 1.2 m deep?",
    "",
])
def test_an_unrelated_percentage_is_not_a_waste_factor(text):
    assert q.waste_factor_from_text(text) is None


def test_a_percentage_far_from_the_word_waste_is_not_claimed():
    # The number has to belong to the waste, not merely share a sentence.
    assert q.waste_factor_from_text(
        "Retention is 5% on a contract that also covers demolition, "
        "disposal of waste and reinstatement.") is None


# ── the default still works where it was designed to ───────────────────────

@pytest.mark.parametrize("text", [
    "Concrete volume for a raft 30x20x1.5 m including your documented waste factor.",
    "Slab 12 x 8 x 0.25 m, apply the documented waste factor please.",
])
def test_documented_waste_with_no_figure_binds_no_pinned_figure(text):
    """K-T5: no figure pinned in code; the caller's input governs."""
    out = _resolved(text)
    assert "waste_factor" not in out
    supplied = _resolved(text, {"waste_factor": 0.03})
    assert supplied["waste_factor"] == pytest.approx(0.03)


@pytest.mark.parametrize("text,wf", [
    ("Raft 30x20x1.5 m, documented waste factor 5%.", 0.05),
    ("Slab 12 x 8 x 0.25 m, documented waste factor 8%.", 0.08),
])
def test_documented_waste_with_a_figure_uses_that_figure(text, wf):
    out = _resolved(text)
    assert out["waste_factor"] == pytest.approx(wf)
    assert q.documented_concrete_waste_factor(text) == pytest.approx(wf)


@pytest.mark.parametrize("text", [
    "Concrete volume for a raft 30 x 20 x 1.5 m.",
    "Slab volume 12 x 8 x 0.25 m.",
])
def test_an_ask_with_no_waste_words_binds_no_factor(text):
    """The resolver injects nothing; the caller's input / kernel signature governs."""
    out = _resolved(text)
    assert "waste_factor" not in out


def test_a_nonsense_percentage_falls_back_rather_than_poisoning_the_quantity():
    # 900% waste is a parse artefact, not an instruction.
    assert q.waste_factor_from_text("add 900% waste") is None
