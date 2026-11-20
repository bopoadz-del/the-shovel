"""Three general rules for the thing a conflict is about (owner A2).

1. A clause that holds ONE value is named by its WHOLE clause, not only the
   words within 12 of the value: a note that names the thing far from the
   value is still about that thing.
2. A trailing "what do X and Y each say?" clause names the sources, not the
   thing: its words never inflate the thing threshold.
3. A kind word takes only the single word running straight into it as its
   kind phrase ("concrete grade"); a longer run is a thing ("defects
   notification period").

Every document, value and question here is invented; expected values are
written by hand. Each decision is shown on two inputs.
"""

from __future__ import annotations

import pytest

from app.agents.evidence_conflicts import detect


def _row(doc: str, text: str, sec: str = "p. 1") -> dict:
    return {"doc_name": doc, "page_or_section": sec, "snippet": text}


# ── 1. a clause holding one value is named by its whole clause ──────────────

_FAR_THING = [
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        [_row("Wenford Pile Cap Schedule.pdf",
              "Pile cap reinforcement for the deep foundations of the four-storey car park "
              "structure shall be fixed with a minimum concrete cover of 75 mm."),
         _row("Wenford Method Statement.pdf",
              "The pile cap reinforcement cover is 50 mm.")],
        "The pile cap reinforcement cover is 75 mm.",
        {("Wenford Pile Cap Schedule.pdf", "75 mm"),
         ("Wenford Method Statement.pdf", "50 mm")},
        id="pile-cap-cover-far",
    ),
    pytest.param(
        "What is the thickness of the basement retaining walls?",
        [_row("Oakdene Retaining Wall Schedule.pdf",
              "Basement retaining walls forming the permanent enclosure of the "
              "underground car park and the adjacent plant rooms shall be 350 mm thick."),
         _row("Oakdene Method Statement.pdf",
              "The basement retaining wall thickness is 300 mm.")],
        "The basement retaining wall is 350 mm thick.",
        {("Oakdene Retaining Wall Schedule.pdf", "350 mm"),
         ("Oakdene Method Statement.pdf", "300 mm")},
        id="retaining-wall-thickness-far",
    ),
]


@pytest.mark.parametrize("question, rows, answer, pairs", _FAR_THING)
def test_a_thing_word_far_from_the_value_is_still_the_thing(question, rows, answer, pairs):
    found = detect(answer, rows, question)
    assert len(found) == 1
    assert {(v["document"], v["value"]) for v in found[0]["values"]} == pairs


# ── 2. a trailing source query names sources, not the thing ─────────────────

_COMPOUND = [
    pytest.param(
        "What reinforcement yield strength is required for the portal frame: what do "
        "the fabrication drawing and the erection method statement each say?",
        [_row("Windrush Fabrication Drawing.pdf",
              "The bars shall be high yield with a minimum strength of 460N/mm²."),
         _row("Windrush Erection Method Statement.pdf",
              "All reinforcement shall have a yield strength of 380N/mm².")],
        "The reinforcement yield strength is 460N/mm².",
        {("Windrush Fabrication Drawing.pdf", "460N/mm²"),
         ("Windrush Erection Method Statement.pdf", "380N/mm²")},
        id="yield-each-say",
    ),
    pytest.param(
        "What cover is required to the basement walls: what do the wall details "
        "and the waterproofing specification each say?",
        [_row("Foxglove Wall Details.pdf", "Wall cover shall be 50 mm."),
         _row("Foxglove Waterproofing Specification.pdf", "Basement wall cover is 40 mm.")],
        "The basement wall cover is 50 mm.",
        {("Foxglove Wall Details.pdf", "50 mm"),
         ("Foxglove Waterproofing Specification.pdf", "40 mm")},
        id="cover-each-say",
    ),
]


@pytest.mark.parametrize("question, rows, answer, pairs", _COMPOUND)
def test_a_source_query_does_not_inflate_the_thing(question, rows, answer, pairs):
    found = detect(answer, rows, question)
    assert len(found) == 1
    assert {(v["document"], v["value"]) for v in found[0]["values"]} == pairs


# ── 3. a kind word takes one word, not a whole run ─────────────────────────

_MULTI_WORD_THING = [
    pytest.param(
        "How long is the defects notification period?",
        [_row("Brightwater Contract Data.pdf", "Defects Notification Period: 365 days."),
         _row("Brightwater Particular Conditions.pdf",
              "The Defects Notification Period shall be 24 months.")],
        "The defects notification period is 365 days.",
        {("Brightwater Contract Data.pdf", "365 days"),
         ("Brightwater Particular Conditions.pdf", "24 months")},
        id="defects-notification-period",
    ),
    pytest.param(
        "How long is the defects liability period?",
        [_row("Havering Contract Data.pdf", "Defects Liability Period: 52 weeks."),
         _row("Havering Particular Conditions.pdf",
              "The Defects Liability Period shall be 18 months.")],
        "The defects liability period is 52 weeks.",
        {("Havering Contract Data.pdf", "52 weeks"),
         ("Havering Particular Conditions.pdf", "18 months")},
        id="defects-liability-period",
    ),
]


@pytest.mark.parametrize("question, rows, answer, pairs", _MULTI_WORD_THING)
def test_a_multi_word_thing_before_a_kind_word_is_not_the_kind(question, rows, answer, pairs):
    found = detect(answer, rows, question)
    assert len(found) == 1
    assert {(v["document"], v["value"]) for v in found[0]["values"]} == pairs
