"""M11: general-only answers must be worded as reference material, not this
project's documents.

Static imports so the mutation gate ties these tests to the exact functions
they cover (_machine_wording, check_text).
"""
from __future__ import annotations

import pytest

from app.agents.answer_exit import _machine_wording, check_text

_GENERAL_SOURCES = [{"layer": "general_knowledge", "doc_name": "Dubai Building Code"}]
_PROJECT_SOURCES = [{"layer": "own", "doc_name": "Contract.pdf"}]


@pytest.mark.parametrize("tier,expected,forbidden", [
    ("general", "reference material", "project document"),
    ("project", "project document sections", "reference material"),
])
def test_machine_wording_respects_tier(tier, expected, forbidden):
    out = _machine_wording("I don't have that in the excerpts.", tier=tier)
    assert expected in out
    assert forbidden not in out


@pytest.mark.parametrize("sources,expected,forbidden", [
    (_GENERAL_SOURCES, "reference material", "project document"),
    (_PROJECT_SOURCES, "project document sections", "reference material"),
])
def test_check_text_wording_is_tier_aware(sources, expected, forbidden):
    out = check_text("I don't have that in the excerpts.", sources=sources)
    assert expected in out
    assert forbidden not in out


@pytest.mark.parametrize("text", [
    "I don't have that in the search results.",
    "I don't have that in the retrieval results.",
])
def test_machine_wording_rewrites_search_noun(text):
    # Exercises the `noun.startswith(('search', 'retrieval'))` branch so a
    # drop_rule mutant that removes it cannot pass unnoticed.
    assert "project document sections" in _machine_wording(text, tier="project")


@pytest.mark.parametrize("text", ["", "   "])
def test_check_text_guard_returns_unchanged(text):
    # Exercises the isinstance/blank-text guard so a drop_rule mutant that
    # removes it cannot pass unnoticed.
    assert check_text(text) == text


@pytest.mark.parametrize("text", [123, b"raw"])
def test_check_text_non_string_returns_unchanged(text):
    # Without the isinstance guard, `.strip()` on a non-string raises
    # AttributeError outside the try, so this kills the drop_rule mutant
    # that removes the guard.
    assert check_text(text) == text
