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

from app.lib import construction_formulas_quantities as q

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

def test_silence_still_gets_the_documented_five_percent():
    out = _resolved("Concrete volume for a raft 30x20x1.5 m including your "
                    "documented waste factor.")
    assert out["waste_factor"] == pytest.approx(q.DOCUMENTED_CONCRETE_WASTE_FACTOR)


def test_an_ask_with_no_waste_words_at_all_still_gets_the_default():
    out = _resolved("Concrete volume for a raft 30 x 20 x 1.5 m.")
    assert out["waste_factor"] == pytest.approx(0.05)


def test_the_kill_switch_still_zeroes_it(monkeypatch):
    monkeypatch.setenv("APPLY_DOCUMENTED_WASTE", "0")
    assert _resolved(ASK)["waste_factor"] == pytest.approx(0.0)


def test_a_nonsense_percentage_falls_back_rather_than_poisoning_the_quantity():
    # 900% waste is a parse artefact, not an instruction.
    assert q.waste_factor_from_text("add 900% waste") is None
