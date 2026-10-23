"""First line credits the clause and the document that carry the figure.

Live measurement on 0b1d13a (42 asks x 6 runs), gate G8: the right
number with the wrong or missing source is a FAIL.

* P1a stated 98% and never put §8.4 on the first line (wrong label),
  6/6 on P1a, P1a-f1 and P1a-f3.
* S2 credited 95% to the wrong source. RSM 15492 was on the first line
  0/24 times. The owner bar is Specification Section 9.1 (RSM 15492),
  and the line never says "not the specification".

The class: when the chunk that holds the committed figure also carries
a clause or section id and a document id, the first line credits both.
It does not credit a different document, and it does not ask anything
back once the figure is stated.

Gate G7: the original phrasing, three rewordings, and one sibling
(a different clause and figure with the same shape). Synthetic chunks
only. Names start with FIXTURE-d-20260928-. Retrieval ranking is not
under test; the chunks are already on the turn.
"""
from __future__ import annotations

import re

import pytest

from app.agents.first_line_hard_rule import apply_first_line_hard_rule

P1A_DOC = "FIXTURE-d-20260928-earthworks-backfill.pdf"
P1A_DOC_ID = "RSM 14880"
P1A_CLAUSE = "8.4"
S2_DOC = "FIXTURE-d-20260928-geotech-volume.pdf"
S2_DOC_ID = "RSM 15492"
S2_CLAUSE = "9.1"
SIB_DOC = "FIXTURE-d-20260928-embankment-fill.pdf"
SIB_DOC_ID = "RSM 22011"
SIB_CLAUSE = "6.2"
WRONG = "FIXTURE-d-20260928-other-fill-note.pdf"

# The clause marker sits after the percent, and the document id is its
# own line. A laboratory-reference aside is a different sentence and is
# not this document.
P1A_TEXT = (
    f"{P1A_DOC_ID}\n"
    "Compaction of structural backfill under foundations to minimum "
    "98% of maximum dry density (§8.4).\n"
)
S2_TEXT = (
    f"{S2_DOC_ID}-Rev0.\n"
    "Specification Section 9.1\n"
    "The sub-grade layer under the road pavement shall be compacted "
    "to 95% of maximum dry density.\n"
)
SIB_TEXT = (
    f"{SIB_DOC_ID}\n"
    "6.2 Embankment fill\n"
    "Embankment fill shall be compacted to 92% of maximum dry density.\n"
)
WRONG_TEXT = (
    "General fill shall be compacted to 90% of maximum dry density.\n"
)

P1A_CASES = [
    (
        "p1a",
        (
            "Per the project specification, to what degree must structural "
            "backfill under foundations be compacted?"
        ),
    ),
    (
        "p1a-f1",
        (
            "As per the project specification, what compaction applies to "
            "structural backfill below foundations?"
        ),
    ),
    (
        "p1a-f2",
        (
            "Under the project specification, how thoroughly must "
            "structural backfill beneath footings be compacted?"
        ),
    ),
    (
        "p1a-f3",
        (
            "According to the project specification, to what percent must "
            "structural backfill under foundations be compacted?"
        ),
    ),
]
S2_CASES = [
    (
        "s2",
        (
            "Per the project specification, what compaction is required "
            "under road pavement?"
        ),
    ),
    (
        "s2-f1",
        (
            "As per the project specification, to what degree must material "
            "under the road pavement be compacted?"
        ),
    ),
    (
        "s2-f2",
        (
            "Under the project specification, what compaction is required "
            "for the sub-grade under road pavement?"
        ),
    ),
    (
        "s2-f3",
        (
            "According to the project specification, what percentage "
            "compaction is required beneath the road pavement?"
        ),
    ),
]
SIB_ASK = (
    "Per the project specification, to what degree must embankment fill "
    "be compacted?"
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


def _has_percent(line: str, number: str) -> bool:
    return bool(re.search(rf"(?i)\b{re.escape(number)}\s*(?:%|percent\b)", line))


def _p1a_answer() -> str:
    """98% is stated and pinned to the other document. §8.4 is absent."""
    return (
        f"98% of maximum dry density is the compaction figure in {WRONG}.\n\n"
        "Structural backfill under foundations uses that degree.\n"
    )


def _s2_answer() -> str:
    """95% is stated and pinned to the other document, and denied as a spec."""
    return (
        f"95% of maximum dry density is the compaction figure in {WRONG}. "
        "That document is not the specification.\n\n"
        "The sub-grade under the road pavement uses that degree.\n"
    )


def _sib_answer() -> str:
    return (
        f"92% of maximum dry density is the compaction figure in {WRONG}.\n\n"
        "Embankment fill uses that degree.\n"
    )


def _assert_quiet(first: str, out: str) -> None:
    assert "?" not in first, first
    assert "which document" not in out.lower(), first
    assert "which document" not in first.lower(), first


@pytest.mark.parametrize("pid,ask", P1A_CASES, ids=[p for p, _a in P1A_CASES])
def test_p1a_first_line_credits_clause_and_document(pid, ask):
    """98% on the first line, with §8.4 and the document that states it.

    The other fill note is not the source. Nothing is asked back.
    """
    out = apply_first_line_hard_rule(
        _p1a_answer(),
        _rag(
            _chunk("wrong", WRONG, WRONG_TEXT, "0.900"),
            _chunk("p1a", P1A_DOC, P1A_TEXT, "0.820"),
        ),
        _msgs(ask),
    )
    first = _first(out)
    assert pid
    assert _has_percent(first, "98"), first
    assert "§8.4" in first, first
    assert P1A_CLAUSE in first, first
    assert P1A_DOC_ID in first, first
    assert P1A_DOC in first, first
    assert WRONG not in first, first
    assert not _has_percent(first, "90"), first
    _assert_quiet(first, out)


@pytest.mark.parametrize("pid,ask", S2_CASES, ids=[p for p, _a in S2_CASES])
def test_s2_first_line_credits_section_and_rsm(pid, ask):
    """95% credits Specification Section 9.1 (RSM 15492).

    The other fill note is not the source. The line does not say the
    document is not the specification, and it does not ask.
    """
    out = apply_first_line_hard_rule(
        _s2_answer(),
        _rag(
            _chunk("wrong", WRONG, WRONG_TEXT, "0.900"),
            _chunk("s2", S2_DOC, S2_TEXT, "0.820"),
        ),
        _msgs(ask),
    )
    first = _first(out)
    assert pid
    assert _has_percent(first, "95"), first
    assert "Specification Section 9.1" in first, first
    assert S2_CLAUSE in first, first
    assert S2_DOC_ID in first, first
    assert S2_DOC in first, first
    assert WRONG not in first, first
    assert "not the specification" not in out.lower(), out
    assert not _has_percent(first, "90"), first
    _assert_quiet(first, out)


def test_sibling_embankment_first_line_credits_clause_and_document():
    """Same shape, different pair: 92% is §6.2 of RSM 22011.

    Embankment fill is not the general-fill note, and nothing is asked.
    """
    out = apply_first_line_hard_rule(
        _sib_answer(),
        _rag(
            _chunk("wrong", WRONG, WRONG_TEXT, "0.900"),
            _chunk("sib", SIB_DOC, SIB_TEXT, "0.820"),
        ),
        _msgs(SIB_ASK),
    )
    first = _first(out)
    assert _has_percent(first, "92"), first
    assert "§6.2" in first, first
    assert SIB_CLAUSE in first, first
    assert SIB_DOC_ID in first, first
    assert SIB_DOC in first, first
    assert WRONG not in first, first
    assert not _has_percent(first, "90"), first
    _assert_quiet(first, out)


# The truth chunk is already on the turn. The model still refuses.
# Live: S2-f1, S2-f2, and P1a-f2, with Specification Section 9.1
# (RSM 15492) and §8.4 / 98% in the handed chunks.
REFUSAL = (
    "I could not find that compaction requirement in the project "
    "specification. The retrieved excerpts do not state it."
)
REFUSAL_CASES = [
    ("p1a-f2", P1A_CASES[2][1], "p1a", "98", "§8.4", P1A_DOC_ID, P1A_DOC),
    ("s2-f1", S2_CASES[1][1], "s2", "95", "Specification Section 9.1", S2_DOC_ID, S2_DOC),
    ("s2-f2", S2_CASES[2][1], "s2", "95", "Specification Section 9.1", S2_DOC_ID, S2_DOC),
]
SIB_REWORD = (
    "According to the project specification, how thoroughly must "
    "embankment fill be compacted?"
)


def _truth_rag(kind: str) -> dict:
    if kind == "p1a":
        truth = ("p1a", P1A_DOC, P1A_TEXT)
    elif kind == "s2":
        truth = ("s2", S2_DOC, S2_TEXT)
    else:
        truth = ("sib", SIB_DOC, SIB_TEXT)
    return _rag(
        _chunk("wrong", WRONG, WRONG_TEXT, "0.910"),
        _chunk(truth[0], truth[1], truth[2], "0.840"),
    )


def _assert_handed(out: str, number: str, clause: str, doc_id: str, doc: str) -> None:
    first = _first(out)
    assert _has_percent(first, number), first
    assert clause in first, first
    assert doc_id in first, first
    assert doc in first, first
    assert WRONG not in first, first
    assert not _has_percent(first, "90"), first
    assert "could not" not in out.lower(), out
    assert "do not state" not in out.lower(), out
    assert "?" not in out, out
    assert "which document" not in out.lower(), out


@pytest.mark.parametrize(
    "pid,ask,kind,number,clause,doc_id,doc",
    REFUSAL_CASES,
    ids=[row[0] for row in REFUSAL_CASES],
)
def test_refusal_yields_to_the_handed_figure(
    pid, ask, kind, number, clause, doc_id, doc,
):
    """A reworded ask whose truth chunk was handed is answered, not refused."""
    assert pid
    out = apply_first_line_hard_rule(REFUSAL, _truth_rag(kind), _msgs(ask))
    _assert_handed(out, number, clause, doc_id, doc)


def test_sibling_refusal_yields_to_the_handed_figure():
    """Same refusal shape, a different clause and figure."""
    out = apply_first_line_hard_rule(
        REFUSAL, _truth_rag("sib"), _msgs(SIB_REWORD),
    )
    _assert_handed(out, "92", "§6.2", SIB_DOC_ID, SIB_DOC)


def test_refusal_stands_when_no_handed_chunk_matches():
    """A general-fill note is not structural backfill. The refusal stays."""
    out = apply_first_line_hard_rule(
        REFUSAL,
        _rag(_chunk("wrong", WRONG, WRONG_TEXT)),
        _msgs(P1A_CASES[2][1]),
    )
    assert out == REFUSAL
