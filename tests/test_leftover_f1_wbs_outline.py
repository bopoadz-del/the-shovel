"""Live UI pack F1: a high-level WBS ask must show a hierarchy.

Live leftover on Master Corpus (cb24f2b, still open after 8f4b465):

    "Answer only from the client project documents. Generate a high-level
     WBS for the demolition and site clearance scope in this project's BOQ."

    Actual PARTIAL: "Schedule built: 204 activities over 688 working days
    (44 on the critical path)… workbook is ready to download."
    No phases, packages, or numbered WBS codes in the chat answer.

THE CAUSE. ``generate_wbs`` already builds ``wbs_tree`` (phase → package).
``deliver_schedule`` printed only the CPM headline + workbook offer.
These tests pin the glass (no live LLM, no corpus). The outline structures
the tree the scheduler already returned — it does not invent BOQ items.
"""
from __future__ import annotations

import asyncio
import re

from app.agents.runtime import _format_wbs_result
from app.core.predefined_reasoning import (
    deliver_schedule,
    format_wbs_outline,
    message_wants_wbs_outline,
    run_workflow,
)
from app.schemas.project_session import ProjectSession

F1_ASK = (
    "Answer only from the client project documents. Generate a high-level "
    "WBS for the demolition and site clearance scope in this project's BOQ."
)
F1_SHOW = (
    "build / show the high-level WBS (work breakdown structure) for the "
    "project schedule from client documents."
)
SCHEDULE_ONLY = "produce the schedule"

# A tree shaped like generate_wbs output — phases plus packages, no leaves.
_TREE = {
    "1": "Site Preparation",
    "1.1": "Survey & Permits",
    "1.2": "Earthworks",
    "2": "Foundations",
    "2.1": "Piling",
}

_WBS_CODE_LINE = re.compile(r"^(?:### )?(?P<code>\d+(?:\.\d+)*)\s+\S", re.M)


def _codes_in(text: str) -> list[str]:
    return [m.group("code") for m in _WBS_CODE_LINE.finditer(text or "")]


def test_f1_phrasing_wants_the_outline():
    assert message_wants_wbs_outline(F1_ASK)
    assert message_wants_wbs_outline(F1_SHOW)
    assert message_wants_wbs_outline("Generate a WBS for a 10-storey tower")
    assert message_wants_wbs_outline("show the work breakdown structure")
    assert not message_wants_wbs_outline(SCHEDULE_ONLY)
    assert not message_wants_wbs_outline("how long is procurement on the schedule?")
    assert not message_wants_wbs_outline("")


def test_outline_is_numbered_and_multi_level():
    out = format_wbs_outline(_TREE)
    codes = _codes_in(out)
    assert "1" in codes and "1.2" in codes
    assert "2" in codes and "2.1" in codes
    assert "### 1 Site Preparation" in out
    assert "1.2 Earthworks" in out
    assert "High-level WBS" in out
    # Hierarchy, not a single flat activity dump.
    assert any("." in c for c in codes)
    assert any("." not in c for c in codes)


def test_outline_does_not_invent_nodes():
    out = format_wbs_outline({"1": "Earthworks", "1.1": "Site clearance"})
    assert "Foundations" not in out
    assert "Hall A" not in out
    assert format_wbs_outline({}) == ""
    assert format_wbs_outline(None) == ""
    assert format_wbs_outline({"1": "", "1.1": None}) == ""


def _session_with_tree(tree=_TREE) -> ProjectSession:
    s = ProjectSession.new("f1")
    s.data["schedule_summary"] = {
        "activities": 204,
        "duration_days": 688,
        "critical_count": 44,
        "total_man_days": 11832,
    }
    s.data["wbs"] = {
        "wbs_tree": tree,
        "scaffold": {
            "declaration": (
                "Template scaffold, project_type inferred: building "
                "— not derived from this project's BOQ, drawings or contract."
            ),
            "derived_from_boq": False,
        },
    }
    return s


def test_deliver_f1_ask_shows_hierarchy_not_only_workbook():
    text = deliver_schedule(_session_with_tree(), True, F1_ASK)
    codes = _codes_in(text)
    assert "1" in codes and "1.2" in codes, text
    assert "Site Preparation" in text and "Earthworks" in text
    assert "Template scaffold" in text
    assert "Schedule built: 204 activities" in text
    assert "download" in text.lower()
    # The PARTIAL glass was this headline with no tree.
    assert text.strip() != (
        "Schedule built: 204 activities over 688 working days "
        "(44 on the critical path). Total effort is about 11,832 man-days. "
        "The cost-loaded workbook (CPM, cumulative man-days S-curve, "
        "manpower histogram, milestones) is ready to download."
    )


def test_deliver_schedule_only_keeps_headline_without_inventing_tree():
    text = deliver_schedule(_session_with_tree(), True, SCHEDULE_ONLY)
    assert "Schedule built: 204 activities" in text
    assert "### 1 Site Preparation" not in text
    assert "High-level WBS" not in text


def test_deliver_wbs_ask_without_tree_does_not_fabricate():
    s = ProjectSession.new("f1-empty")
    s.data["schedule_summary"] = {
        "activities": 12,
        "duration_days": 40,
        "critical_count": 3,
    }
    text = deliver_schedule(s, False, F1_ASK)
    assert "Schedule built: 12 activities" in text
    assert _codes_in(text) == []
    assert "Hall" not in text


def test_format_wbs_result_renders_tree_codes():
    out = _format_wbs_result({
        "actual_count": 20,
        "project_type": "building",
        "summary": {"activity_count": 20},
        "wbs_tree": _TREE,
        "scaffold": {"declaration": "Template scaffold, project_type inferred: building"},
    })
    assert "Template scaffold" in out
    assert "1.2 Earthworks" in out
    assert out.index("Template scaffold") < out.index("1.2 Earthworks")
    assert out.index("1.2 Earthworks") < out.index("Template activities:")


def test_f1_run_workflow_answer_has_numbered_wbs():
    """End-to-end predefined path: the live leftover prompt, no LLM."""
    ctx = {
        "message": F1_ASK,
        "project_name": "Master Corpus",
        "project_id": "master_corpus",
        "params": {
            "project_type": "infrastructure",
            "target_count": 60,
            "start_date": "2026-01-05",
        },
    }
    out = asyncio.run(run_workflow("generate_wbs", ctx, ProjectSession.new("f1-wf")))
    assert out["handled"] and out["status"] == "success", out
    answer = out["answer"]
    codes = _codes_in(answer)
    assert any(c == "1" for c in codes), answer
    assert any("." in c for c in codes), answer
    assert "High-level WBS" in answer
    assert "Schedule built:" in answer
    assert "download" in answer.lower()
    # Infrastructure template carries these packages; do not invent others.
    assert "Earthworks" in answer or "Site Preparation" in answer
    assert "Hall A" not in answer
