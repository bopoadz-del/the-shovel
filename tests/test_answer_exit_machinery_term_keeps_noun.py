"""A removed machinery term never leaves its determiner or preposition bare.

When the term was the noun after "the" / "an" / "of the" / "via", deleting it
alone left "The returns them" or "out of the for you". The term is then said
in the reader's word for what the answer read; with no such lead word it is
still deleted. Two inputs per decision, every input invented here.
"""

from __future__ import annotations

import re

import pytest

from app.agents.answer_exit import _machine_wording

_BARE = re.compile(r"\b(?:the|a|an|of|via|these)\s+(?:[.,;:!?]|for\b|returns\b|holds\b|today\b|are\b|slowly\b|$)",
                   re.IGNORECASE)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The index returns them in pages.", "The project material returns them in pages."),
        ("Ask and I will pull it out of the index for you.",
         "Ask and I will pull it out of the project material for you."),
        ("An embedding holds it.", "The project material holds it."),
        ("These vectors are stale.", "These project document sections are stale."),
        ("Results came via RAG today.", "Results came via the project material today."),
        ("I read the tokens slowly.", "I read the project document sections slowly."),
    ],
)
def test_term_after_a_lead_word_becomes_the_readers_noun(text, expected):
    out = _machine_wording(text)
    assert out == expected
    assert not _BARE.search(out)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The index returns them in pages.", "The reference material returns them in pages."),
        ("These vectors are stale.", "These reference documents are stale."),
    ],
)
def test_general_tier_uses_its_own_noun_and_agrees_in_number(text, expected):
    assert _machine_wording(text, tier="general") == expected


@pytest.mark.parametrize(
    "text",
    [
        "Pipeline steps ran twice.",
        "It ran top-k twice.",
    ],
)
def test_term_without_a_lead_word_is_still_deleted(text):
    out = _machine_wording(text)
    assert "pipeline" not in out.lower() and "top-k" not in out.lower()
    assert "project material" not in out and "project document sections" not in out


@pytest.mark.parametrize(
    "text",
    [
        "the indexed files are listed",
        "This project has no indexed documents yet",
    ],
)
def test_document_status_indexed_is_not_replaced(text):
    assert _machine_wording(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "The slab is 24 m3 in the north bay.",
        "Pull the rows out of the bill for you.",
    ],
)
def test_text_without_a_machinery_term_is_unchanged(text):
    assert _machine_wording(text) == text
