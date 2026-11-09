"""Tests for evidence-plan generation (item 2): prompt + parse."""
from __future__ import annotations

import pytest

from app.core.rag.plan_generation import (
    EVIDENCE_PLAN_PROMPT,
    extract_plan_text,
    parse_generated_plan,
)


def test_prompt_asks_for_items_not_figures():
    assert "evidence plan" in EVIDENCE_PLAN_PROMPT
    assert "not" in EVIDENCE_PLAN_PROMPT and "compute" in EVIDENCE_PLAN_PROMPT
    assert "figure" in EVIDENCE_PLAN_PROMPT.lower()


@pytest.mark.parametrize("response,want", [
    ({"choice": {"message": {"content": "bulk excavation\nblinding"}}},
     "bulk excavation\nblinding"),
    ({"choice": {"message": {"content": "formwork\npainting"}}},
     "formwork\npainting"),
])
def test_extract_plan_text_reads_the_message_content(response, want):
    assert extract_plan_text(response) == want


@pytest.mark.parametrize("response,want", [
    ({"choices": [{"message": {"content": "lift walls"}}]}, "lift walls"),
    ({"choices": [{"message": {"content": "staircases"}}]}, "staircases"),
])
def test_extract_plan_text_handles_the_choices_list_shape(response, want):
    assert extract_plan_text(response) == want


@pytest.mark.parametrize("response", [
    {},
    {"choice": {"message": {}}},
    "not a dict",
])
def test_extract_plan_text_empty_on_missing_content(response):
    assert extract_plan_text(response) == ""


@pytest.mark.parametrize("content,want", [
    ("bulk excavation\nblinding\nlift walls",
     ("bulk excavation", "blinding", "lift walls")),
    ("formwork\npainting\nceilings",
     ("formwork", "painting", "ceilings")),
])
def test_parse_generated_plan_splits_lines(content, want):
    plan = parse_generated_plan({"choice": {"message": {"content": content}}})
    assert plan.items == want
