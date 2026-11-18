"""What is NOT a conflict, and the "agree" sentence under a stated conflict
(advisor 18:31Z, points 1 and 2).

1. Not a conflict, two inputs each, each beside a positive control that
   does conflict:
   * the same value in different units (1.8 m and 1800 mm);
   * a rounding / format of the same figure (4,858 and 4858.0);
   * a range that contains the other value (150-200 mm and 175 mm). A range
     is one value covering both ends, never its upper end alone.
2. When a conflict is stated, a model sentence asserting that the documents
   agree about that thing is replaced -- read by the typed values and the
   thing's words: it states a differing value, or uses a word of the thing,
   or names no other thing at all ("the sources are consistent"). An
   agreement sentence about another thing stays; a negated one ("do not
   agree") is no agreement.

Every document, value and question here is invented for this file; the
expected values are written by hand.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.evidence_conflicts import agreement_names_no_thing, asserts_agreement, detect, extract

_LINE = "The documents differ on this:"


def _rows(*pairs: tuple[str, str]) -> list[dict]:
    return [{"doc_id": f"d{i}", "doc_name": n, "snippet": t} for i, (n, t) in enumerate(pairs)]


# ── 1. not a conflict ───────────────────────────────────────────────────────

_SAME_VALUE = [
    # same value, different units
    pytest.param("What is the pile cap depth?",
                 ("Harlow Pile Cap Schedule.pdf", "Pile cap depth is 1.8 m."),
                 ("Harlow Foundation Specification.pdf", "Pile cap depth is 1800 mm."),
                 "The pile cap depth is 1.8 m.", "Pile cap depth is 1900 mm.", id="units-m-mm"),
    pytest.param("What is the transfer slab thickness?",
                 ("Wrenfield Slab Drawing.pdf", "Transfer slab thickness 0.45 m throughout."),
                 ("Wrenfield Structural Notes.pdf", "Transfer slab thickness 45 cm throughout."),
                 "The transfer slab thickness is 0.45 m.", "Transfer slab thickness 50 cm throughout.",
                 id="units-m-cm"),
    # a rounding / format of the same figure
    pytest.param("What is the bulk excavation volume for the basement?",
                 ("Harlow Earthworks Bill.pdf", "Basement bulk excavation volume 4,858 m3."),
                 ("Harlow Earthworks Report.pdf", "Basement bulk excavation volume 4858.0 m3."),
                 "The basement bulk excavation volume is 4,858 m3.",
                 "Basement bulk excavation volume 4,958 m3.", id="rounding-volume"),
    pytest.param("What is the provisional sum for landscaping?",
                 ("Wrenfield Bill of Quantities.pdf", "Provisional sum for landscaping AED 1,250,000."),
                 ("Wrenfield Cost Plan.pdf", "Provisional sum for landscaping AED 1250000.00."),
                 "The provisional sum for landscaping is AED 1,250,000.",
                 "Provisional sum for landscaping AED 1,520,000.", id="rounding-money"),
    # a range that contains the other value
    pytest.param("What is the topsoil strip depth?",
                 ("Harlow Site Clearance Specification.pdf", "Topsoil strip depth is 150-200 mm."),
                 ("Harlow Method Statement.pdf", "Topsoil strip depth 175 mm over the whole site."),
                 "The topsoil strip depth is 175 mm.", "Topsoil strip depth 250 mm over the whole site.",
                 id="range-hyphen"),
    # (the question names the slab, not only the kind "curing period": a
    # kind-only question is no conflict by itself and would test nothing)
    pytest.param("How long must the podium slab be cured?",
                 ("Wrenfield Concrete Specification.pdf", "The podium slab shall be cured for between 5 and 7 days."),
                 ("Wrenfield Pour Method Statement.pdf", "The podium slab shall be cured for 6 days after each pour."),
                 "The podium slab must be cured for 6 days.",
                 "The podium slab shall be cured for 10 days after each pour.", id="range-between"),
]


@pytest.mark.parametrize("question, a, b, answer, other", _SAME_VALUE)
def test_the_same_value_written_another_way_is_no_conflict(question, a, b, answer, other):
    rows = _rows(a, b)
    assert detect(answer, rows, question) == []
    out = answer_contract.apply(answer, sources=rows, question=question)
    assert _LINE not in out.text and not out.conflicts


@pytest.mark.parametrize("question, a, b, answer, other", _SAME_VALUE)
def test_positive_control_a_different_value_does_conflict(question, a, b, answer, other):
    rows = _rows(a, (b[0], other))
    assert len(detect(answer, rows, question)) == 1
    assert answer_contract.apply(answer, sources=rows, question=question).text.startswith(_LINE)


@pytest.mark.parametrize("text, want", [
    pytest.param("Strip 150-200 mm of topsoil.", ("150-200 mm", (0.15, 0.2)), id="hyphen"),
    pytest.param("Cure for between 5 and 7 days.", ("between 5 and 7 days", (5.0, 7.0)), id="between"),
    pytest.param("Lap 1.5 m to 2 m.", ("1.5 m to 2 m", (1.5, 2.0)), id="unit-both-ends"),
    pytest.param("Allow 2 to 3 months.", ("2 to 3 months", (56.0, 93.0)), id="month-range-both-ends"),
])
def test_a_range_is_one_value_covering_both_ends(text, want):
    vals = extract(text)
    assert [(v.raw, tuple(round(x, 6) for x in v.norm)) for v in vals] == [want]


@pytest.mark.parametrize("text, want", [
    pytest.param("Slabs are 150 and 200 mm.", ["200 mm"], id="and-without-between-is-not-a-range"),
    pytest.param("Allow 3 days to 2 m of fill.", ["3 days", "2 m"], id="two-kinds-are-not-a-range"),
])
def test_what_is_not_a_range(text, want):
    assert [v.raw for v in extract(text)] == want


# ── 2. the "agree" sentence under a stated conflict ─────────────────────────

_CONFLICT = [
    pytest.param("What is the ground floor slab thickness?",
                 ("Harlow Slab Drawing.pdf", "Ground floor slab thickness is 200 mm."),
                 ("Harlow Structural Specification.pdf", "Ground floor slab thickness is 250 mm."),
                 "The ground floor slab thickness is 200 mm.", id="slab-200-250"),
    pytest.param("What concrete grade is specified for the pile caps?",
                 ("Wrenfield Pile Cap Details.pdf", "Pile caps are concrete grade C40."),
                 ("Wrenfield Concrete Schedule.pdf", "Pile caps are concrete grade C45."),
                 "The pile caps are concrete grade C40.", id="grade-c40-c45"),
]


def _apply(question, a, b, answer):
    return answer_contract.apply(answer, sources=_rows(a, b), question=question).text


@pytest.mark.parametrize("question, a, b, answer", _CONFLICT)
@pytest.mark.parametrize("agree", ["The sources are consistent.", "Both documents match.",
                                   "There is no conflict between the documents."])
def test_a_valueless_agree_sentence_does_not_stand_below_the_conflict(question, a, b, answer, agree):
    text = _apply(question, a, b, f"{answer} {agree}")
    assert text.startswith(_LINE)
    assert agree not in text


@pytest.mark.parametrize("question, a, b, answer", _CONFLICT)
def test_an_agree_sentence_naming_the_thing_is_replaced(question, a, b, answer):
    thing = question.rstrip("?").split(" the ", 1)[1]
    agree = f"The documents are consistent about the {thing}."
    text = _apply(question, a, b, f"{answer} {agree}")
    assert text.startswith(_LINE) and agree not in text


@pytest.mark.parametrize("question, a, b, answer", _CONFLICT)
def test_an_agree_sentence_about_another_thing_stays(question, a, b, answer):
    other = "The documents are consistent on the fire rating of the core walls."
    text = _apply(question, a, b, f"{answer} {other}")
    assert text.startswith(_LINE) and other in text


@pytest.mark.parametrize("question, a, b, answer", _CONFLICT)
def test_a_negated_agreement_stating_both_values_is_left_as_written(question, a, b, answer):
    v1, v2 = extract(a[1])[0].raw, extract(b[1])[0].raw
    told = f"The documents do not agree: the drawing gives {v1} and the other gives {v2}."
    out = answer_contract.apply(told, sources=_rows(a, b), question=question)
    assert _LINE not in out.text and not out.conflicts
    assert told in out.text


@pytest.mark.parametrize("sentence, agrees", [
    ("The sources are consistent.", True),
    ("Both documents match.", True),
    ("There is no conflict between the documents.", True),
    ("The documents do not agree.", False),
    ("They are not consistent on this.", False),
    ("The drawings never match the bill.", False),
    ("The documents are no longer consistent.", False),
])
def test_the_agreement_reading(sentence, agrees):
    assert asserts_agreement(sentence) is agrees


@pytest.mark.parametrize("sentence, docs, none", [
    ("The sources are consistent.", [], True),
    ("The Harlow drawing and the Harlow bill agree.", ["Harlow Drawing.pdf", "Harlow Bill.pdf"], True),
    ("The Harlow drawing and the Harlow bill agree.", [], False),
    ("The sources are consistent on the fire rating.", [], False),
    ("Both documents give 200 mm, they match.", [], False),
])
def test_an_agreement_naming_no_other_thing(sentence, docs, none):
    assert agreement_names_no_thing(sentence, docs) is none
