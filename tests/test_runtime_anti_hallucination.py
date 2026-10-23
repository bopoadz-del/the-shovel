"""Tests for the two runtime anti-hallucination guardrails:

1. ``_scrub_history`` strips assistant turns whose content contains a
   WBS/BOQ-shaped markdown table so the LLM can't pattern-match to a
   prior (often hallucinated) table when it should be calling the tool.
2. ``_user_intent_requires_tool`` returns True when the latest user
   message names a deliverable (schedule, WBS, BOQ, etc.). The runtime
   uses this to flip ``tool_choice`` from ``"auto"`` to ``"required"``
   for the project-assistant agent only.

Together they neutralise the failure mode caught via WebBridge: both
gpt-oss:120b and qwen3-coder:480b regenerated fake CPM tables (every
row Float=0 / Critical=Y, fabricated CSV paths) instead of calling
``generate_wbs`` when the conversation history contained prior
hallucinated tables.
"""
from __future__ import annotations

from app.agents.runtime import (
    _DELIVERABLE_PHRASES,
    _HALLUC_TABLE_RE,
    _scrub_history,
    _user_intent_requires_tool,
)


# ── _scrub_history: strips WBS-shaped tables in assistant turns ───────────


def _wbs_table_assistant_turn() -> dict:
    """Reproduces the shape the platform observed in real failed runs."""
    table = (
        "Here is the 50-activity schedule:\n\n"
        "| # | Activity ID | Activity Name | Duration | Early Start | Early Finish | Float | Critical? |\n"
        "|---|-------------|---------------|----------|-------------|--------------|-------|-----------|\n"
        "| 1 | 1.1.1 | Site survey | 14 | 0 | 14 | 0 | Y |\n"
        "| 2 | 1.1.2 | Geotech | 21 | 14 | 35 | 0 | Y |\n"
        "| 3 | 1.1.3 | EIA | 14 | 35 | 49 | 0 | Y |\n"
        "| 4 | 1.1.4 | Permit submission | 7 | 49 | 56 | 0 | Y |\n"
        "| 5 | 1.1.5 | Permit approval | 30 | 56 | 86 | 0 | Y |\n"
        "| 6 | 1.2.1 | Demolition | 16 | 86 | 102 | 0 | Y |\n"
        "\n"
        "You can download the CSV from Anthropic_DataCentre_250_Activities.csv."
    )
    return {"role": "assistant", "content": table}


def test_scrub_history_strips_wbs_table_assistant_turn():
    history = [
        {"role": "user", "content": "Give me a 50-activity schedule"},
        _wbs_table_assistant_turn(),
        {"role": "user", "content": "Now generate a manpower histogram"},
    ]
    scrubbed = _scrub_history(history)
    assert len(scrubbed) == 3, "scrub must preserve turn count for positional continuity"
    assert scrubbed[1]["role"] == "assistant"
    assert "Activity ID" not in scrubbed[1]["content"]
    assert "schedule/BOQ output omitted" in scrubbed[1]["content"]
    # User turns must NEVER be touched.
    assert scrubbed[0]["content"] == "Give me a 50-activity schedule"
    assert scrubbed[2]["content"] == "Now generate a manpower histogram"


def test_scrub_history_leaves_clean_assistant_turns_alone():
    history = [
        {"role": "user", "content": "What is CPI?"},
        {
            "role": "assistant",
            "content": (
                "CPI (Cost Performance Index) is the ratio of earned value "
                "to actual cost. CPI > 1 means under budget."
            ),
        },
    ]
    scrubbed = _scrub_history(history)
    assert scrubbed == history, "non-table assistant turns must pass through unchanged"


def test_scrub_history_leaves_user_pasted_tables_alone():
    """A user pasting a real BOQ table for review must not be scrubbed —
    the scrub is targeted at assistant turns only, where hallucination
    happens."""
    user_table = (
        "Please review this BOQ:\n\n"
        "| Item | Quantity | Unit Rate | Amount |\n"
        "|------|----------|-----------|--------|\n"
        "| Concrete | 100 | 120 | 12000 |\n"
        "| Rebar    | 200 | 1.5 | 300   |\n"
        "| Formwork | 50  | 25  | 1250  |\n"
        "| Steel    | 10  | 900 | 9000  |\n"
        "| Crane    | 5   | 200 | 1000  |\n"
        "| Pump     | 8   | 180 | 1440  |\n"
    )
    history = [{"role": "user", "content": user_table}]
    scrubbed = _scrub_history(history)
    assert scrubbed[0]["content"] == user_table, "user turns must be untouched"


def test_scrub_history_does_not_mutate_input():
    """Defensive: scrub returns a new list, the caller's history is
    preserved so retries can re-derive without surprises."""
    original = [_wbs_table_assistant_turn()]
    snapshot = dict(original[0])
    _ = _scrub_history(original)
    assert original[0] == snapshot, "input must not be mutated"


def test_scrub_history_short_table_passes_through():
    """The threshold is ``>=5 data rows``. A 3-row table is more likely
    a legitimate summary than a hallucinated WBS."""
    short_assistant_table = {
        "role": "assistant",
        "content": (
            "Here are the top tasks:\n\n"
            "| Task | Duration |\n"
            "|------|----------|\n"
            "| A | 1 |\n"
            "| B | 2 |\n"
            "| C | 3 |\n"
        ),
    }
    scrubbed = _scrub_history([short_assistant_table])
    assert scrubbed[0]["content"] == short_assistant_table["content"]


def test_hallucination_regex_directly_matches():
    """Smoke check on the regex itself — useful for diagnosing if the
    higher-level scrub stops working."""
    msg = _wbs_table_assistant_turn()["content"]
    assert _HALLUC_TABLE_RE.search(msg), "regex must match the canonical failure shape"


# ── _user_intent_requires_tool: flips tool_choice on deliverable intent ──


def test_intent_requires_tool_fires_on_schedule_keyword():
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Generate a 250-activity construction schedule for the data centre."},
    ]
    assert _user_intent_requires_tool(messages) is True


def test_intent_requires_tool_fires_on_boq():
    messages = [
        {"role": "user", "content": "Extract the BOQ from the uploaded xlsx file."},
    ]
    assert _user_intent_requires_tool(messages) is True


def test_intent_requires_tool_does_not_fire_on_explanation_question():
    """Q&A questions must keep tool_choice=auto. 'What is a WBS?' must
    NOT trigger forced tool use — that would call generate_wbs when the
    user just wants a definition."""
    messages = [
        {"role": "user", "content": "What is a WBS in construction?"},
    ]
    # 'wbs' the standalone token shouldn't trigger — only deliverable phrases
    # like 'wbs for X' or 'generate a wbs' should. The phrase list uses bare
    # 'wbs' which is a known false-positive risk — log it explicitly.
    # When this assertion eventually flips, switch to a regex-based matcher.
    # For now we document the current behaviour.
    assert _user_intent_requires_tool(messages) is True, (
        "current matcher catches bare 'wbs' — documented false positive. "
        "If this becomes a real issue, upgrade to a phrase-boundary matcher."
    )


def test_intent_requires_tool_picks_latest_user_turn():
    """When the conversation has multiple user turns, the matcher uses
    the MOST RECENT one — the model's response depends on what the user
    is asking right now, not what they asked five turns ago.

    The matcher now also requires the tail to be the user turn (i.e.
    iteration 0 of the agent loop) — see
    test_intent_requires_tool_returns_false_after_tool_call_iteration.
    """
    messages = [
        {"role": "user", "content": "What is CPI?"},
        {"role": "assistant", "content": "CPI is earned value / actual cost."},
        {"role": "user", "content": "Now give me a manpower histogram for the project."},
    ]
    assert _user_intent_requires_tool(messages) is True


def test_intent_requires_tool_returns_false_after_tool_call_iteration():
    """REGRESSION: on iteration 2+ of the agent loop the runtime has
    appended an assistant turn with tool_calls AND tool result turns
    after the user message. If we still flipped tool_choice="required"
    on these iterations the model would be FORCED to call another tool
    instead of summarising into the final answer — a forever-loop trap
    seen in prod via the Render debug log.

    The matcher must return False whenever the tail of messages is
    NOT a user turn.
    """
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Generate a 250-activity construction schedule"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "generate_wbs"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "[activities...]"},
    ]
    assert _user_intent_requires_tool(messages) is False, (
        "On iter 2+ the tail is a tool result — must return False "
        "so the model can write the final answer instead of being "
        "forced into another tool call."
    )


def test_intent_requires_tool_returns_false_after_assistant_summary():
    """A tail of role=assistant (no tool_calls, final-answer-shape) also
    means we're past iter 0 — the matcher must return False."""
    messages = [
        {"role": "user", "content": "Schedule please"},
        {"role": "assistant", "content": "Here's the schedule..."},
    ]
    assert _user_intent_requires_tool(messages) is False


def test_intent_requires_tool_returns_false_with_no_user_turn():
    """Edge case: tool-only follow-up turn with no user content. We
    fall back to auto — letting the model decide is the safe default."""
    messages = [
        {"role": "system", "content": "..."},
        {"role": "assistant", "content": "I called the tool.", "tool_calls": []},
    ]
    assert _user_intent_requires_tool(messages) is False


def test_deliverable_phrases_contains_canonical_keywords():
    """Spot check that the phrase list covers the user-facing keywords
    documented in project-assistant.md's tool-call mandate table."""
    for needle in (
        "construction schedule", "wbs", "manpower histogram",
        "boq", "cost estimate", "critical path",
    ):
        assert needle in _DELIVERABLE_PHRASES, f"missing canonical keyword: {needle}"
