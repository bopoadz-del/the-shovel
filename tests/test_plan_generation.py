"""Tests for evidence-plan generation (item 2): prompt + parse."""
from __future__ import annotations

from app.core.rag.plan_generation import (
    EVIDENCE_PLAN_PROMPT,
    extract_plan_text,
    parse_generated_plan,
)


def test_prompt_asks_for_items_not_figures():
    assert "evidence plan" in EVIDENCE_PLAN_PROMPT
    assert "not" in EVIDENCE_PLAN_PROMPT and "compute" in EVIDENCE_PLAN_PROMPT
    assert "figure" in EVIDENCE_PLAN_PROMPT.lower()


def test_extract_plan_text_reads_the_message_content():
    response = {"choice": {"message": {"content": "bulk excavation\nblinding"}}}
    assert extract_plan_text(response) == "bulk excavation\nblinding"


def test_extract_plan_text_handles_the_choices_list_shape():
    response = {"choices": [{"message": {"content": "lift walls"}}]}
    assert extract_plan_text(response) == "lift walls"


def test_extract_plan_text_empty_on_missing_content():
    assert extract_plan_text({}) == ""
    assert extract_plan_text({"choice": {"message": {}}}) == ""
    assert extract_plan_text("not a dict") == ""


def test_parse_generated_plan_splits_lines():
    plan = parse_generated_plan(
        {"choice": {"message": {"content": "bulk excavation\nblinding\nlift walls"}}}
    )
    assert plan.items == ("bulk excavation", "blinding", "lift walls")
