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
