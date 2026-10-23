"""Different conditions are not a conflict about which document governs.

PR #714 stops the first-line guard from asking "which document's figure
is meant?" when the retrieved figures agree. Figures that answer
different materials or conditions were still treated as a conflict.

FIX WAVE 4 #1: ask which document only when two figures answer the same
material and condition, or when no condition separates them. Otherwise
state each figure bound to its own condition and to the document,
clause, or drawing that actually contains it.

The live miss keeps the condition off the figure's own sentence: a
clause heading several lines above ("8.4 Backfill"), a section title
("Specification Section 9"), or only the document title ("Storm Water
network"). Those still bind. "not the specification" is emitted only
for a drawing, a report, or a design note, and never when the chunk
calls itself a Specification section.

Synthetic chunks only. Names start with FIXTURE-d-20260927-.
"""
from __future__ import annotations

import re

import pytest

from app.agents.first_line_hard_rule import (
    apply_first_line_hard_rule,
    first_line_hard_rule_enabled,
)

P1A_ASK = (
    "Per the project specification, to what degree must structural "
    "backfill under foundations be compacted?"
)
S1_ASK = (
    "Per the project specification, what is the minimum concrete cover "
    "to reinforcement for foundations?"
)
S2_ASK = (
    "Per the project specification, what compaction is required under "
    "road pavement?"
)

DOC_98 = "FIXTURE-d-20260927-spec-backfill-98.pdf"
DOC_95 = "FIXTURE-d-20260927-spec-structural-fill-95.pdf"
DOC_90 = "FIXTURE-d-20260927-spec-general-fill-90.pdf"
BACKFILL_98 = (
    "Clause 8.4. Compaction of structural backfill under foundations "
    "to minimum 98% of maximum dry density of the modified proctor test."
)
STRUCTURAL_FILL_95 = (
    "Clause 8.1. Structural fill shall be compacted to 95% of maximum "
    "dry density of the modified proctor test."
)
GENERAL_FILL_90 = (
    "Clause 8.2. General fill shall be compacted to 90% of maximum "
    "dry density of the modified proctor test."
)

SPEC_COVER = "FIXTURE-d-20260927-spec-cover-deferral.pdf"
DWG_SOIL = "FIXTURE-d-20260927-dwg-cover-soil.pdf"
DWG_FOOT = "FIXTURE-d-20260927-dwg-cover-footing.pdf"
SPEC_DEFERS = (
    "Clause 3.1.25.8. Concrete cover to reinforcement shall be as shown "
    "on the drawings. This clause states no cover figure."
)
COVER_75 = (
    "Nominal concrete cover to reinforcement is 75 mm for concrete cast "
    "against or in contact with soil."
)
COVER_100 = (
    "Nominal concrete cover to reinforcement is 100 mm at the bottom "
    "of footings."
)

DOC_S2 = "FIXTURE-d-20260927-geotech-subgrade.pdf"
SUBGRADE_95 = (
    "The sub-grade layer under the road pavement shall be compacted to "
    "95% of maximum dry density."
)

# Same material and the same condition. The clause numbers differ on
# purpose: a clause id is not a condition, and must not suppress the ask.
DOC_SAME_A = "FIXTURE-d-20260927-spec-backfill-copy-a.pdf"
DOC_SAME_B = "FIXTURE-d-20260927-spec-backfill-copy-b.pdf"
SAME_98 = (
    "Clause 8.4. Compaction of structural backfill under foundations "
    "to minimum 98% of maximum dry density of the modified proctor test."
)
SAME_95 = (
    "Clause 9.1. Compaction of structural backfill under foundations "
    "to minimum 95% of maximum dry density of the modified proctor test."
)

HEDGE = "which document"
NARRATIVE = "I will answer from the retrieved context."


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


def _percent_sentence(text: str, number: str) -> str:
    rx = re.compile(rf"(?i)\b{re.escape(number)}\s*(?:%|percent\b)")
    for sentence in _sentences(text):
        if rx.search(sentence):
            return sentence
    return ""


def _mm_sentence(text: str, number: str) -> str:
    rx = re.compile(rf"(?i)\b{re.escape(number)}\s*mm\b")
    for sentence in _sentences(text):
        if rx.search(sentence):
            return sentence
    return ""


def _has_percent(line: str, number: str) -> bool:
    return bool(re.search(rf"(?i)\b{re.escape(number)}\s*(?:%|percent\b)", line))


def test_p1a_different_materials_do_not_ask_which_document():
    """The body commits to 98% first. 95% and 90% are other subjects.

    The annotation names only that 98% and §8.4. It does not ask.
    """
    answer = (
        "I will answer from the retrieved context.\n\n"
        "Structural backfill under foundations is compacted to 98% of "
        "maximum dry density.\n"
        "General fill is 90% of maximum dry density.\n"
        "Structural fill is 95% of maximum dry density.\n"
    )
    out = apply_first_line_hard_rule(
        answer,
        _rag(
            _chunk("f95", DOC_95, STRUCTURAL_FILL_95, "0.860"),
            _chunk("f90", DOC_90, GENERAL_FILL_90, "0.840"),
            _chunk("f98", DOC_98, BACKFILL_98, "0.820"),
        ),
        _msgs(P1A_ASK),
    )
    first = _first(out)
    assert HEDGE not in out.lower(), first
    assert "?" not in first, first
    assert first.lower().startswith("98%"), first
    assert "8.4" in first, first
    assert DOC_98 in first, first
    assert not _has_percent(first, "95"), first
    assert not _has_percent(first, "90"), first


def test_s1_different_cover_conditions_name_the_drawing_not_the_spec():
    """The body commits to 75 mm first. 100 mm is a later, other subject.

    The annotation names 75 mm and its drawing. The specification
    deferral is not credited.
    """
    answer = (
        "I will answer from the retrieved context.\n\n"
        "Nominal concrete cover is 75 mm for concrete cast against soil.\n"
        "The bottom of footings is 100 mm.\n"
    )
    out = apply_first_line_hard_rule(
        answer,
        _rag(
            _chunk("spec", SPEC_COVER, SPEC_DEFERS, "0.900"),
            _chunk("soil", DWG_SOIL, COVER_75, "0.860"),
            _chunk("foot", DWG_FOOT, COVER_100, "0.840"),
        ),
        _msgs(S1_ASK),
    )
    first = _first(out)
    assert HEDGE not in out.lower(), first
    assert "?" not in first, first
    soil = _mm_sentence(first, "75")
    assert soil, first
    assert "contact with soil" in soil.lower() or "cast against" in soil.lower(), soil
    assert DWG_SOIL in soil, soil
    assert SPEC_COVER not in soil, soil
    assert not re.search(r"(?i)\b100\s*mm\b", first), first
    assert SPEC_COVER not in first, first


def test_s2_single_corpus_figure_is_credited_without_a_hedge():
    """The body commits to 95%. The annotation credits that document."""
    answer = (
        "I will answer from the retrieved context.\n\n"
        "The sub-grade is compacted to 95% of maximum dry density.\n"
    )
    out = apply_first_line_hard_rule(
        answer,
        _rag(_chunk("g95", DOC_S2, SUBGRADE_95)),
        _msgs(S2_ASK),
    )
    first = _first(out)
    assert HEDGE not in first.lower(), first
    assert "more than one" not in first.lower(), first
    assert first.lower().startswith("95%"), first
    assert _has_percent(first, "95"), first
    assert DOC_S2 in first, first
    assert not re.search(r"(?i)\b(?:90|98|100)\s*(?:%|percent\b)", first), first


def test_same_condition_from_two_documents_still_asks_which():
    """Negative control. Both figures are structural backfill under foundations.

    Clause 8.4 and clause 9.1 do not make them different conditions.
    Two documents, two figures, one condition: still ask which.
    """
    out = apply_first_line_hard_rule(
        NARRATIVE,
        _rag(
            _chunk("a", DOC_SAME_A, SAME_98, "0.860"),
            _chunk("b", DOC_SAME_B, SAME_95, "0.840"),
        ),
        _msgs(P1A_ASK),
    )
    first = _first(out)
    assert HEDGE in first.lower(), first
    assert _has_percent(first, "98") and _has_percent(first, "95"), first
    assert DOC_SAME_A in first and DOC_SAME_B in first, first


# ── Live shape: condition lives in a heading, a section title, or the name ──
#
# The figure line itself does not name the material. "8.4 Backfill" sits
# several sentences above 98%. "8.1 Structural Fill" and "Specification
# Section 9" sit above 95%. 90% names no material; the document title is
# the storm water network. "RSM 15492" is a lab reference, not the clause.

DOC_VOL5_4 = "FIXTURE-d-20260927-Vol 5 Geotechnical Report (4 of 5).pdf"
DOC_VOL5_2 = "FIXTURE-d-20260927-Vol 5 Geotechnical Report (2 of 5).pdf"
DOC_STORM = "FIXTURE-d-20260927-Storm Water network.pdf"
BACKFILL_98_HEADING = (
    "8.4 Backfill\n"
    "\n"
    "RSM 15492 is the laboratory reference cited in the borehole logs.\n"
    "Place the material in loose layers not exceeding the stated thickness.\n"
    "Each layer is tested before the next layer is placed.\n"
    "Compact the placed material to 98% of maximum dry density of the "
    "modified Proctor test.\n"
)
STRUCTURAL_95_SECTION = (
    "Specification Section 9\n"
    "8.1 Structural Fill\n"
    "\n"
    "The following clauses apply to engineered fill.\n"
    "Place and test each layer before the next is placed.\n"
    "Compact the material to 95% of maximum dry density of the "
    "modified Proctor test.\n"
)
STORM_90 = (
    "Bedding for the piped network shall be compacted to 90% of "
    "maximum dry density.\n"
)

P1A_PHRASINGS = [
    (
        "p1a-under-foundations",
        (
            "Per the project specification, to what degree must structural "
            "backfill under foundations be compacted?"
        ),
    ),
    (
        "p1a-below-foundations",
        (
            "As per the project specification, what compaction applies to "
            "backfill below foundations?"
        ),
    ),
    (
        "p1a-beneath-footings",
        (
            "According to the project specification, how thoroughly must "
            "backfill beneath footings be compacted?"
        ),
    ),
]

DWG_SOIL_H = "FIXTURE-d-20260927-Foundation Cover Drawing A.pdf"
DWG_FOOT_H = "FIXTURE-d-20260927-Foundation Cover Drawing B.pdf"
SPEC_DEFER_H = "FIXTURE-d-20260927-Vol 2 Specification cover deferral.pdf"
COVER_75_HEADING = (
    "Concrete cast against or in contact with soil\n"
    "\n"
    "General notes on this drawing govern unless a dimension is shown.\n"
    "Foundations on this sheet follow that note.\n"
    "Nominal concrete cover to reinforcement is 75 mm.\n"
)
COVER_100_HEADING = (
    "Bottom of footings\n"
    "\n"
    "Refer to the structural notes for reinforcement grades.\n"
    "Cover at the foundation element on this drawing is noted below.\n"
    "Nominal concrete cover to reinforcement is 100 mm.\n"
)
S1_PHRASINGS = [
    (
        "s1-minimum-cover",
        (
            "Per the project specification, what is the minimum concrete "
            "cover to reinforcement for foundations?"
        ),
    ),
    (
        "s1-required-cover",
        (
            "As per the project specification, what concrete cover to "
            "reinforcement is required for a foundation?"
        ),
    ),
    (
        "s1-which-cover",
        (
            "According to the project specification, which minimum cover "
            "to reinforcement applies for foundations?"
        ),
    ),
]
S2_PHRASINGS = [
    (
        "s2-under-pavement",
        (
            "Per the project specification, what compaction is required "
            "under road pavement?"
        ),
    ),
    (
        "s2-degree-pavement",
        (
            "As per the project specification, to what degree must material "
            "under the road pavement be compacted?"
        ),
    ),
    (
        "s2-percent-pavement",
        (
            "According to the project specification, what percentage "
            "compaction is required beneath the road pavement?"
        ),
    ),
]


# The 95% chunk used for (D) does not itself say "specification".
# Only the model's answer does. The filename remains a geotechnical report.
BARE_95 = (
    "Compact the layer to 95% of maximum dry density of the "
    "modified Proctor test.\n"
)

VERIFY_CASES = P1A_PHRASINGS + S1_PHRASINGS + S2_PHRASINGS


def _kind(pid: str) -> str:
    if pid.startswith("p1a"):
        return "p1a"
    if pid.startswith("s1"):
        return "s1"
    return "s2"


def _multi_rag(kind: str) -> dict:
    if kind == "p1a":
        return _rag(
            _chunk("v4", DOC_VOL5_4, BACKFILL_98_HEADING, "0.820"),
            _chunk("sw", DOC_STORM, STORM_90, "0.800"),
            _chunk("v2", DOC_VOL5_2, STRUCTURAL_95_SECTION, "0.780"),
        )
    if kind == "s1":
        return _rag(
            _chunk("spec", SPEC_DEFER_H, SPEC_DEFERS, "0.900"),
            _chunk("soil", DWG_SOIL_H, COVER_75_HEADING, "0.860"),
            _chunk("foot", DWG_FOOT_H, COVER_100_HEADING, "0.840"),
        )
    return _rag(
        _chunk("v2", DOC_VOL5_2, BARE_95, "0.820"),
        _chunk("sw", DOC_STORM, STORM_90, "0.800"),
    )


def _single_rag(kind: str) -> dict:
    if kind == "p1a":
        return _rag(_chunk("v4", DOC_VOL5_4, BACKFILL_98_HEADING))
    if kind == "s1":
        return _rag(_chunk("soil", DWG_SOIL_H, COVER_75_HEADING))
    return _rag(_chunk("v2", DOC_VOL5_2, BARE_95))


def _pass_through_text(kind: str) -> str:
    """(A) First line already carries the committed figure and its source."""
    if kind == "p1a":
        return (
            f"98% of maximum dry density is the compaction figure (§8.4) in "
            f"{DOC_VOL5_4}.\n\n"
            "Storm water bedding is compacted to 90% of maximum dry "
            "density. Structural fill is 95% of maximum dry density.\n"
        )
    if kind == "s1":
        return (
            f"75 mm is the concrete-cover figure in {DWG_SOIL_H}.\n\n"
            "The bottom of footings is 100 mm on the other drawing.\n"
        )
    return (
        f"95% of maximum dry density is the compaction figure in "
        f"{DOC_VOL5_2}.\n\n"
        "Specification Section 9.1 is the governing clause. Storm water "
        "bedding elsewhere is 90% of maximum dry density.\n"
    )


def _committed_text(kind: str) -> str:
    """(B) The first figure in the body is the one the model chose."""
    if kind == "p1a":
        return (
            "I will answer from the retrieved context.\n\n"
            "Backfill under foundations is 98% of maximum dry density.\n"
            "Storm water bedding is 90% of maximum dry density.\n"
            "Structural fill is 95% of maximum dry density.\n"
        )
    if kind == "s1":
        return (
            "I will answer from the retrieved context.\n\n"
            "Concrete cover is 75 mm where concrete is cast against soil.\n"
            "The bottom of footings is 100 mm.\n"
        )
    return (
        "I will answer from the retrieved context.\n\n"
        "Specification Section 9.1 requires 95% of maximum dry density.\n"
        "Storm water bedding is 90% of maximum dry density.\n"
    )


def _uncommitted_text(kind: str) -> str:
    """(C) No figure is stated, so no question may be invented."""
    if kind == "s2":
        return (
            "I will answer from the retrieved context. "
            "Specification Section 9.1 is the governing clause.\n"
        )
    return "I will answer from the retrieved context.\n"


def _body_spec_text(kind: str) -> str:
    """(D) The body names the class. The opening line has no figure."""
    if kind == "p1a":
        return (
            "The clause I am using is below.\n\n"
            "Specification Section 8.4 requires 98% of maximum dry density.\n"
        )
    if kind == "s1":
        return (
            "The clause I am using is below.\n\n"
            "Specification Section 4.2 gives a nominal concrete cover of "
            "75 mm.\n"
        )
    return (
        "The clause I am using is below.\n\n"
        "Specification Section 9.1 requires 95% of maximum dry density.\n"
    )


@pytest.mark.parametrize("pid,ask", VERIFY_CASES, ids=[p for p, _a in VERIFY_CASES])
def test_a_figure_and_source_pass_through(pid, ask):
    """(A) A first line that already has a figure and a source is kept."""
    kind = _kind(pid)
    text = _pass_through_text(kind)
    out = apply_first_line_hard_rule(text, _multi_rag(kind), _msgs(ask))
    assert out == text, _first(out)


@pytest.mark.parametrize("pid,ask", VERIFY_CASES, ids=[p for p, _a in VERIFY_CASES])
def test_b_prepends_only_the_committed_figure(pid, ask):
    """(B) Prepend only the first figure the body states. Do not ask."""
    kind = _kind(pid)
    text = _committed_text(kind)
    out = apply_first_line_hard_rule(text, _multi_rag(kind), _msgs(ask))
    first = _first(out)
    assert "which document" not in out.lower(), first
    assert "?" not in first, first
    if kind == "p1a":
        assert first.lower().startswith("98%"), first
        assert "8.4" in first, first
        assert DOC_VOL5_4 in first, first
        assert "rsm" not in first.lower(), first
        assert not _has_percent(first, "90"), first
        assert not _has_percent(first, "95"), first
    elif kind == "s1":
        assert re.search(r"(?i)\b75\s*mm\b", first), first
        assert DWG_SOIL_H in first, first
        assert SPEC_DEFER_H not in first, first
        assert not re.search(r"(?i)\b100\s*mm\b", first), first
    else:
        assert first.lower().startswith("95%"), first
        assert DOC_VOL5_2 in first, first
        assert not _has_percent(first, "90"), first
        assert "not the specification" not in out.lower(), out


@pytest.mark.parametrize("pid,ask", VERIFY_CASES, ids=[p for p, _a in VERIFY_CASES])
def test_c_does_not_invent_a_question(pid, ask):
    """(C) Different subjects, and no commitment, do not get a question."""
    kind = _kind(pid)
    text = _uncommitted_text(kind)
    out = apply_first_line_hard_rule(text, _multi_rag(kind), _msgs(ask))
    assert out == text, _first(out)
    assert "which document" not in out.lower(), out


@pytest.mark.parametrize("pid,ask", VERIFY_CASES, ids=[p for p, _a in VERIFY_CASES])
def test_d_body_specification_is_not_contradicted(pid, ask):
    """(D) The body says Specification Section. Do not deny that."""
    kind = _kind(pid)
    text = _body_spec_text(kind)
    out = apply_first_line_hard_rule(text, _single_rag(kind), _msgs(ask))
    first = _first(out)
    assert "not the specification" not in out.lower(), first
    assert "which document" not in out.lower(), first
    if kind == "p1a":
        assert _has_percent(first, "98"), first
        assert DOC_VOL5_4 in first, first
        assert "8.4" in first, first
    elif kind == "s1":
        assert re.search(r"(?i)\b75\s*mm\b", first), first
        assert DWG_SOIL_H in first, first
    else:
        assert _has_percent(first, "95"), first
        assert DOC_VOL5_2 in first, first


def test_e_kill_switch_off_leaves_the_answer(monkeypatch):
    """(E) FIRST_LINE_HARD_RULE=0 leaves the model's answer unchanged."""
    monkeypatch.setenv("FIRST_LINE_HARD_RULE", "0")
    assert first_line_hard_rule_enabled() is False
    text = _committed_text("p1a")
    out = apply_first_line_hard_rule(
        text, _multi_rag("p1a"), _msgs(P1A_PHRASINGS[0][1]),
    )
    assert out == text


def test_e_kill_switch_defaults_on(monkeypatch):
    """(E) With the env unset the guard stays on."""
    monkeypatch.delenv("FIRST_LINE_HARD_RULE", raising=False)
    assert first_line_hard_rule_enabled() is True
