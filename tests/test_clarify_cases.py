"""Tests for the clarify-before-choosing exit decision (PR 2 part b).

Each of the seven bounds from the design note is a test. The detector is
pure, so these run against recorded strings, not a retriever mock.
"""

from __future__ import annotations

import pytest

from app.agents.clarify_cases import (
    Case,
    _is_date_token,
    _number_of,
    _same_subject,
    _text_commits,
    _unit_of,
    cases_from_passages,
    clarifying_question,
    enabled,
)


def _cases(*rows: tuple[str, str, str]) -> list[Case]:
    return [Case(subject=s, value=v, source=src) for s, v, src in rows]


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("CLARIFY_CASES", raising=False)
    assert enabled() is True


def test_kill_switch_off(monkeypatch):
    monkeypatch.setenv("CLARIFY_CASES", "0")
    assert enabled() is False
    # Two differing cases: only the kill switch (not the case count, not the
    # subject, not the draft) stops this turn from asking -- so the OFF flag
    # is actually what is under test.
    q = clarifying_question(
        "what is the figure?",
        "The figure is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is None


# --- the value/subject readers (direct: they decide case 2 vs case 3) ------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("40 mm", 40.0),
        ("40", 40.0),
        ("1,200 mm", 1200.0),
        ("8.8.1", None),  # a clause id is more than one number: a string
    ],
)
def test_number_of(value, expected):
    assert _number_of(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("40 mm", "mm"),
        ("98%", "%"),
        ("40", ""),
    ],
)
def test_unit_of(value, expected):
    assert _unit_of(value) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("concrete cover", "concrete cover", True),
        ("concrete cover", "backfill depth", False),
        ("", "", True),  # empty subject: the empty-token early return
        ("the", "the", True),  # stopword-only subject -> empty tokens too
    ],
)
def test_same_subject(a, b, expected):
    assert _same_subject(a, b) == expected


@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("The figure is 40 mm.", "40 mm", True),
        # The number matches even though the value string is not spelled out:
        # the numeric branch (not the string-containment branch) commits it.
        ("The figure is 40 millimeters.", "40 mm", True),
        ("The figure is 300 mm.", "40 mm", False),
    ],
)
def test_text_commits(text, value, expected):
    assert _text_commits(text, value) == expected


# Bound 2 -- same value in every case: answer, never ask.
def test_bound2_same_value_no_ask():
    q = clarifying_question(
        "what is the cover?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("concrete cover", "40 mm", "b.pdf"),
        ),
    )
    assert q is None


# Bound 3 -- the question names its case: no ask.
def test_bound3_question_names_case():
    q = clarifying_question(
        "what is the concrete cover to reinforcement?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover to reinforcement", "40 mm", "a.pdf"),
            ("concrete cover to foundation", "50 mm", "b.pdf"),
        ),
    )
    assert q is None


# Bound 3 -- the question names the value: no ask.
def test_bound3_question_names_value():
    q = clarifying_question(
        "is the cover 40 mm?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill", "98%", "b.pdf"),
        ),
    )
    assert q is None


# Bound 4 -- documents state one thing plainly (one case): no invented ask.
def test_bound4_single_case_no_ask():
    q = clarifying_question(
        "what is the cover?",
        "The cover is 40 mm.",
        _cases(("concrete cover", "40 mm", "a.pdf")),
    )
    assert q is None


# Bound 5 -- the question lists only the cases found, each with its source.
def test_bound5_lists_only_found_cases():
    q = clarifying_question(
        "what is the figure?",
        "The figure is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is not None
    assert "40 mm" in q and "a.pdf" in q
    assert "300 mm" in q and "b.pdf" in q
    assert "50 mm" not in q  # not invented


# Bound 6 -- one question, never a chain; "all" is the caller's next turn.
@pytest.mark.parametrize(
    "question,draft,cases",
    [
        (
            "what is the figure?",
            "The figure is 40 mm.",
            _cases(
                ("concrete cover", "40 mm", "a.pdf"),
                ("backfill depth", "300 mm", "b.pdf"),
            ),
        ),
        (
            "which clause applies?",
            "Clause 8.8.1 applies.",
            _cases(
                ("design", "8.8.1", "a.pdf"),
                ("construction", "8.8.2", "b.pdf"),
            ),
        ),
    ],
)
def test_bound6_one_question_not_a_chain(question, draft, cases):
    q = clarifying_question(question, draft, cases)
    assert q is not None
    assert q.count("Which one is meant?") == 1
    assert q.count("?") == 1


# Bound 7 -- a merely broad question with one value is answered, not bounced.
def test_bound7_broad_question_answered():
    q = clarifying_question(
        "summarise the specification figures",
        "The specification gives a cover of 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("concrete cover", "40 mm", "b.pdf"),
        ),
    )
    assert q is None


# Case 2 -- different subjects, different values, question names none: ASK.
def test_case2_different_subjects_asks():
    q = clarifying_question(
        "what is the required value?",
        "The required value is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is not None
    assert "Which one is meant?" in q


# Case 3 -- same subject, different values: the both-values statement (#453),
# not this mechanism.
def test_case3_same_subject_different_values_no_ask():
    q = clarifying_question(
        "what is the concrete cover?",
        "The concrete cover is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("concrete cover", "50 mm", "b.pdf"),
        ),
    )
    assert q is None


# The draft must have silently picked exactly ONE value to trigger.
def test_draft_commits_to_none_no_ask():
    q = clarifying_question(
        "what is the figure?",
        "See the documents.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is None


def test_draft_commits_to_both_no_ask():
    q = clarifying_question(
        "what is the figure?",
        "The figures are 40 mm and 300 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is None


# Clause ids compare as strings, not floats (8.8.1 vs 8.8.2 is a conflict,
# not the same number).
def test_clause_ids_are_strings():
    q = clarifying_question(
        "which clause applies?",
        "Clause 8.8.1 applies.",
        _cases(
            ("design", "8.8.1", "a.pdf"),
            ("construction", "8.8.2", "b.pdf"),
        ),
    )
    assert q is not None
    assert "8.8.1" in q and "8.8.2" in q


def test_clause_ids_same_value_no_ask():
    q = clarifying_question(
        "which clause applies?",
        "Clause 8.8.1 applies.",
        _cases(
            ("design", "8.8.1", "a.pdf"),
            ("construction", "8.8.1", "b.pdf"),
        ),
    )
    assert q is None


# Same subject conservatively; two different structures never collapse.
def test_different_structures_never_collapse():
    q = clarifying_question(
        "what is the cover?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover to reinforcement", "40 mm", "a.pdf"),
            ("concrete cover to foundation", "50 mm", "b.pdf"),
        ),
    )
    assert q is not None


# --- the reader: value claims from handed passages (pure) ------------------


@pytest.mark.parametrize(
    "source,text",
    [
        ("a.pdf", "The programme runs from Jan-25 to Feb-25."),
        ("b.pdf", "The certificate is dated 30-Jan-25."),
        ("c.pdf", "Revised Feb-2025."),
    ],
)
def test_cases_from_passages_skips_dates(source, text):
    # A calendar date matches the letter-number reference SHAPE but is not a
    # clause or document reference; it must not become a case.
    assert cases_from_passages([(source, text)]) == []


def test_cases_from_passages_keeps_references_after_date_skip():
    cases = cases_from_passages([
        ("a.pdf", "Clause 8.8.1 applies. Dated Jan-25."),
    ])
    values = {c.value for c in cases}
    assert "8.8.1" in values
    assert not any("Jan-25" in c.value for c in cases)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("Jan-25", True),
        ("Feb-2025", True),
        ("8.8.1", False),
        ("C35/45", False),
        ("AB-2001-101", False),
    ],
)
def test_is_date_token(token, expected):
    assert _is_date_token(token) is expected


@pytest.mark.parametrize(
    "source,text,expected_value,expected_subject",
    [
        (
            "a.pdf",
            "The concrete cover to reinforcement shall be 40 mm.",
            "40 mm",
            "concrete cover",
        ),
        (
            "b.pdf",
            "Backfill shall be compacted to 98% of maximum dry density.",
            "98%",
            "backfill",
        ),
    ],
)
def test_cases_from_passages_reads_a_measure(
    source, text, expected_value, expected_subject,
):
    cases = cases_from_passages([(source, text)])
    assert len(cases) == 1
    assert cases[0].value == expected_value
    assert cases[0].source == source
    assert expected_subject in cases[0].subject.lower()


@pytest.mark.parametrize(
    "source,text,expected_value",
    [
        ("a.pdf", "Clause 8.8.1 applies to the works.", "8.8.1"),
        ("b.pdf", "The blinding grade is C35/45.", "C35/45"),
    ],
)
def test_cases_from_passages_reads_a_reference(source, text, expected_value):
    cases = cases_from_passages([(source, text)])
    assert len(cases) == 1
    assert cases[0].value == expected_value
    assert cases[0].source == source


@pytest.mark.parametrize(
    "source,text",
    [
        ("a.pdf", "See page 12 for the detail."),
        ("b.pdf", "Refer to clause 4.2 before proceeding."),
    ],
)
def test_cases_from_passages_skips_bare_numbers(source, text):
    assert cases_from_passages([(source, text)]) == []


@pytest.mark.parametrize(
    "text,first_hint,second_hint",
    [
        (
            "The concrete cover shall be 40 mm. The backfill depth shall be 300 mm.",
            "concrete cover",
            "backfill depth",
        ),
        (
            "The blinding is C35/45. The capping concrete is C25/30.",
            "blinding",
            "capping concrete",
        ),
    ],
)
def test_cases_from_passages_two_sentences_two_subjects(
    text, first_hint, second_hint,
):
    cases = cases_from_passages([("a.pdf", text)])
    assert len(cases) == 2
    subjects = {c.subject.lower() for c in cases}
    assert any(first_hint in s for s in subjects)
    assert any(second_hint in s for s in subjects)


# End-to-end: the reader feeds the detector (case 2 -- different subjects).
@pytest.mark.parametrize(
    "passages,draft",
    [
        (
            [
                ("a.pdf", "The concrete cover to reinforcement is 40 mm."),
                ("b.pdf", "The backfill depth is 300 mm."),
            ],
            "The cover is 40 mm.",
        ),
        (
            [
                ("c.pdf", "The blinding grade is C35/45."),
                ("d.pdf", "The capping concrete grade is C25/30."),
            ],
            "The blinding grade is C35/45.",
        ),
    ],
)
def test_reader_feeds_detector_asks(passages, draft):
    cases = cases_from_passages(passages)
    q = clarifying_question("what is the required value?", draft, cases)
    assert q is not None
    assert "Which one is meant?" in q
