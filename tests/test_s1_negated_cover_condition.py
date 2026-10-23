"""A negated condition must not be labelled as the positive one.

Live S1 on main opened with "50 mm is the concrete-cover figure for
concrete cast against or in contact with soil … in ST-200". 50 mm is the
figure for faces that are not in contact with soil. 75 mm is the figure
for concrete cast against or in contact with soil, and it belongs to
ST-200. 100 mm is the figure for the bottom of footings, and it belongs
to the footing drawing.

The same shape on compaction: "other than structural fill" is not the
structural-fill figure.

Synthetic chunks only. Names start with FIXTURE-d-20260928-.
"""
from __future__ import annotations

import re

import pytest

from app.agents.first_line_hard_rule import apply_first_line_hard_rule

SOIL = "concrete cast against or in contact with soil"
FOOTING = "the bottom of footings"

ST200 = "FIXTURE-d-20260928-ST-200.pdf"
FOOT_DWG = "FIXTURE-d-20260928-footing-drawing.pdf"
FILL_DOC = "FIXTURE-d-20260928-earthworks-fill.pdf"
LESS_THAN_DOC = "FIXTURE-d-20260928-cover-minimum-note.pdf"

COVER_100 = (
    "Nominal concrete cover to reinforcement is 100 mm at the bottom "
    "of footings."
)
# "not" here negates "less than", not the soil condition. The 75 mm
# figure stays soil contact.
LESS_THAN_75 = (
    "Clear cover to reinforcement shall not be less than 75 mm "
    "for concrete in contact with soil."
)

S1_PHRASINGS = [
    (
        "s1-per-the-specification",
        (
            "Per the project specification, what is the minimum concrete "
            "cover to reinforcement for foundations?"
        ),
    ),
    (
        "s1-as-per-required",
        (
            "As per the project specification, what concrete cover to "
            "reinforcement is required for a foundation?"
        ),
    ),
    (
        "s1-according-to-which",
        (
            "According to the project specification, which minimum cover "
            "to reinforcement applies for foundations?"
        ),
    ),
]
FOOT_ASK = (
    "Per the project specification, what is the minimum concrete cover "
    "to reinforcement at the bottom of footings?"
)
FILL_ASK = (
    "Per the project specification, to what degree must structural fill "
    "be compacted?"
)

# 75 mm keeps its own condition even when "Cl. 4.2" sits inside the
# sentence. 50 mm is a different sentence and a negated condition.
NOT_IN_CONTACT = (
    "For concrete cast against or in contact with soil, Cl. 4.2 states "
    "nominal concrete cover to reinforcement of 75 mm. "
    "Nominal concrete cover to reinforcement is 50 mm for faces not in "
    "contact with soil."
)
NOT_EXPOSED = (
    "Nominal concrete cover to reinforcement is 75 mm for concrete cast "
    "against or in contact with soil. "
    "Faces not exposed to soil have a nominal concrete cover to "
    "reinforcement of 50 mm."
)
OTHER_THAN_CAST = (
    "For concrete in contact with soil, Cl. 4.2 states nominal concrete "
    "cover to reinforcement of 75 mm. "
    "Faces other than cast against soil have a nominal concrete cover "
    "to reinforcement of 50 mm."
)
NEGATION_CHUNKS = [
    ("not-in-contact-with-soil", NOT_IN_CONTACT),
    ("faces-not-exposed-to-soil", NOT_EXPOSED),
    ("other-than-cast-against-soil", OTHER_THAN_CAST),
]
FILL_TEXT = (
    "Materials other than structural fill shall be compacted to 90% of "
    "maximum dry density. "
    "Structural fill shall be compacted to 95% of maximum dry density."
)


def _chunk(doc_id: str, name: str, text: str, score: str = "0.800") -> str:
    return (
        f"[doc_id={doc_id} chunk=0 score={score} class=project_corpus "
        f"src={name}] {text}"
    )


def _rag(*chunks: str) -> dict:
    return {"role": "system", "content": "\n\n".join(chunks)}


def _msgs(ask: str) -> list:
    return [{"role": "user", "content": ask}]


def _first(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=\.)\s+", text) if part.strip()]


def _cover_rag(st200_text: str) -> dict:
    return _rag(
        _chunk("st200", ST200, st200_text, "0.900"),
        _chunk("foot", FOOT_DWG, COVER_100, "0.840"),
    )


def _s1_body() -> str:
    """The model states 50 mm first. That figure is not the soil figure."""
    return (
        "I will answer from the retrieved context.\n\n"
        "The drawing note gives 50 mm.\n"
        "Concrete in contact with soil is 75 mm.\n"
        "The bottom of footings is 100 mm.\n"
    )


def _has_mm(line: str, number: str) -> bool:
    return bool(re.search(rf"(?i)\b{re.escape(number)}\s*mm\b", line))


def _has_percent(line: str, number: str) -> bool:
    return bool(re.search(rf"(?i)\b{re.escape(number)}\s*(?:%|percent\b)", line))


def _assert_50_never_labelled_soil(text: str) -> None:
    for sentence in _sentences(text):
        if _has_mm(sentence, "50"):
            assert SOIL not in sentence.lower(), sentence


def _assert_75_soil_on_st200(first: str) -> None:
    assert _has_mm(first, "75"), first
    assert SOIL in first.lower(), first
    assert ST200 in first, first
    assert FOOT_DWG not in first, first
    assert not _has_mm(first, "50"), first
    assert not _has_mm(first, "100"), first


@pytest.mark.parametrize("pid,ask", S1_PHRASINGS, ids=[p for p, _a in S1_PHRASINGS])
def test_s1_question_credits_75mm_soil_on_st200(pid, ask):
    """Three phrasings of S1. 75 mm is soil contact, credited to ST-200.

    50 mm is stated first in the body and must not take the soil label.
    """
    out = apply_first_line_hard_rule(
        _s1_body(), _cover_rag(NOT_IN_CONTACT), _msgs(ask),
    )
    first = _first(out)
    assert pid
    _assert_75_soil_on_st200(first)
    _assert_50_never_labelled_soil(out)


@pytest.mark.parametrize(
    "pid,st200_text", NEGATION_CHUNKS, ids=[p for p, _t in NEGATION_CHUNKS],
)
def test_s1_50mm_is_never_labelled_soil(pid, st200_text):
    """50 mm stays off the soil label under each negated wording."""
    out = apply_first_line_hard_rule(
        _s1_body(),
        _cover_rag(st200_text),
        _msgs(S1_PHRASINGS[0][1]),
    )
    first = _first(out)
    assert pid
    _assert_75_soil_on_st200(first)
    _assert_50_never_labelled_soil(out)


def test_s1_100mm_bottom_of_footings_credited_to_footing_drawing():
    """100 mm is the bottom-of-footings figure, credited to that drawing.

    The body still leads with 50 mm. That figure is not promoted.
    """
    out = apply_first_line_hard_rule(
        _s1_body(), _cover_rag(NOT_IN_CONTACT), _msgs(FOOT_ASK),
    )
    first = _first(out)
    assert _has_mm(first, "100"), first
    assert FOOTING in first.lower(), first
    assert FOOT_DWG in first, first
    assert ST200 not in first, first
    assert SOIL not in first.lower(), first
    assert not _has_mm(first, "50"), first
    assert not _has_mm(first, "75"), first
    _assert_50_never_labelled_soil(out)


def test_compaction_other_than_structural_fill_is_not_structural_fill():
    """90% is 'other than structural fill'. 95% is structural fill."""
    body = (
        "I will answer from the retrieved context.\n\n"
        "Other material is compacted to 90% of maximum dry density.\n"
        "Structural fill is compacted to 95% of maximum dry density.\n"
    )
    out = apply_first_line_hard_rule(
        body,
        _rag(_chunk("fill", FILL_DOC, FILL_TEXT)),
        _msgs(FILL_ASK),
    )
    first = _first(out)
    assert first.lower().startswith("95%"), first
    assert "structural fill" in first.lower(), first
    assert FILL_DOC in first, first
    assert not _has_percent(first, "90"), first
    for sentence in _sentences(out):
        if _has_percent(sentence, "90"):
            assert "structural fill" not in sentence.lower(), sentence


def test_not_be_less_than_still_labels_soil_contact():
    """'shall not be less than 75 mm' does not negate the soil condition."""
    ask = S1_PHRASINGS[0][1]
    body = (
        "I will answer from the retrieved context.\n\n"
        "The minimum cover is 75 mm.\n"
    )
    out = apply_first_line_hard_rule(
        body,
        _rag(_chunk("note", LESS_THAN_DOC, LESS_THAN_75)),
        _msgs(ask),
    )
    first = _first(out)
    assert _has_mm(first, "75"), first
    assert SOIL in first.lower(), first
    assert LESS_THAN_DOC in first, first
