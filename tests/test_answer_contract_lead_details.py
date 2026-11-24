"""Rule 5 of the answer contract: the lead is the answer, the rest is detail.

Every input is built here; each decision is checked from both sides. No word
count is asserted as a limit -- the rule reads the answer's shape.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract, answer_exit


def _apply(text: str, **kw):
    return answer_contract.apply(text, **kw)


def _words(text: str) -> set[str]:
    return set(text.replace("\n", " ").split())


# ── later paragraphs leave the lead, word for word ─────────────────────────

@pytest.mark.parametrize("text,lead,details", [
    ("The retaining wall footing is 600 mm deep.\n\n"
     "The drawings also show a 75 mm blinding layer under the footing.\n\n"
     "Reinforcement laps follow the general notes.",
     "The retaining wall footing is 600 mm deep.",
     "The drawings also show a 75 mm blinding layer under the footing.\n\n"
     "Reinforcement laps follow the general notes."),
    ("The handrail height on the ramp is 1,000 mm.\n\n"
     "Guarding at the mezzanine edge is 1,100 mm.",
     "The handrail height on the ramp is 1,000 mm.",
     "Guarding at the mezzanine edge is 1,100 mm."),
])
def test_later_paragraphs_move_to_details_lookup(text, lead, details):
    done = _apply(text)
    assert done.text == lead
    assert done.details == details


@pytest.mark.parametrize("text,opening,details", [
    ("No document in the project states the paint colour for the stair core.\n\n"
     "The finishes schedule covers the lobby and the corridors only.",
     "No document in the project states",
     "The finishes schedule covers the lobby and the corridors only."),
    ("The uploaded files do not give a warranty period for the roof membrane.\n\n"
     "The roofing specification names the product and its thickness.\n\n"
     "The contract conditions refer to a defects period in general terms.",
     "The uploaded files do not give a warranty period",
     "The roofing specification names the product and its thickness.\n\n"
     "The contract conditions refer to a defects period in general terms."),
])
def test_later_paragraphs_move_to_details_not_found(text, opening, details):
    done = _apply(text)
    assert done.text.startswith(opening)
    assert done.details == details


def test_nothing_is_deleted():
    text = ("Alpha gate opens at 07:00.\n\nBravo note one sentence.\n\n- item one. extra words here.\n- item two")
    done = _apply(text)
    assert _words(text) <= _words(done.text) | _words(done.details or "")


# ── one paragraph is untouched ──────────────────────────────────────────────

def test_one_paragraph_answer_is_untouched():
    text = "The slab is 250 mm thick on level 2."
    done = _apply(text)
    assert done.text == text and done.details is None


def test_one_paragraph_of_several_sentences_is_untouched():
    text = "The contractor is Northgate Ltd. The engineer is the city works office."
    done = _apply(text)
    assert done.text == text and done.details is None


# ── a list the lead introduces stays, items cut to their first sentence ─────

@pytest.mark.parametrize("text,lead,moved", [
    ("The submittals due this month are:\n\n"
     "- Fire stopping. The sample was rejected once and is being resubmitted.\n"
     "- Door hardware\n\n"
     "Two further submittals are due next month.",
     "The submittals due this month are:\n\n"
     "- Fire stopping.\n"
     "- Door hardware",
     ("- Fire stopping. The sample was rejected once and is being resubmitted.",
      "Two further submittals are due next month.")),
    ("The open RFIs on the facade are:\n\n"
     "- Bracket spacing\n"
     "- Sealant colour. The architect asked for a second sample panel.\n\n"
     "The structural RFIs are all closed.",
     "The open RFIs on the facade are:\n\n"
     "- Bracket spacing\n"
     "- Sealant colour.",
     ("- Sealant colour. The architect asked for a second sample panel.",
      "The structural RFIs are all closed.")),
])
def test_introduced_list_stays_with_items_cut(text, lead, moved):
    done = _apply(text)
    assert done.text == lead
    for piece in moved:
        assert piece in done.details


@pytest.mark.parametrize("text,lead,details_start", [
    ("The pump room is on basement level 1.\n\n"
     "- Access is from the north stair.\n"
     "- The door is 1.2 m wide.",
     "The pump room is on basement level 1.",
     "- Access is from the north stair."),
    ("The generator sits on the roof plant deck.\n\n"
     "1. Fuel is piped from the basement tank.\n"
     "2. The flue rises 3 m above the deck.",
     "The generator sits on the roof plant deck.",
     "1. Fuel is piped from the basement tank."),
])
def test_list_not_introduced_by_the_lead_moves(text, lead, details_start):
    done = _apply(text)
    assert done.text == lead
    assert done.details.startswith(details_start)


@pytest.mark.parametrize("text,lead,details", [
    ("## Answer\n\n"
     "The cover to the raft is 75 mm.\n\n"
     "The piling cap cover is also stated as 75 mm in the same note.\n\n"
     "## Background\n\n"
     "The structural notes were revised twice.",
     "## Answer\n\nThe cover to the raft is 75 mm.\n\n"
     "The piling cap cover is also stated as 75 mm in the same note.",
     "## Background\n\nThe structural notes were revised twice."),
    ("### Fire rating\n\n"
     "The shaft walls are rated 120 minutes.\n\n"
     "### Inspection\n\n"
     "The rating comes from the fire strategy report.\n\n"
     "Doors into the shaft are rated 60 minutes.",
     "### Fire rating\n\nThe shaft walls are rated 120 minutes.",
     "### Inspection\n\nThe rating comes from the fire strategy report.\n\n"
     "Doors into the shaft are rated 60 minutes."),
])
def test_opening_heading_keeps_its_whole_section(text, lead, details):
    done = _apply(text)
    assert done.text == lead
    assert done.details == details


def test_opening_heading_with_one_section_is_untouched():
    text = ("**Lintel bearings**\n\n"
            "- Masonry: 150 mm\n- Steel: 100 mm\n\n"
            "Precast lintels take 200 mm.")
    done = _apply(text)
    assert done.text == text and done.details is None


# ── markers stay with the sentence that carries them ────────────────────────

def test_marker_of_a_cut_item_stays_on_the_item():
    text = ("The holds are:\n\n"
            "- Hold one. Raised by the engineer [2].\n"
            "- Hold two [1]")
    done = _apply(text)
    assert "- Hold one. [2]" in done.text
    assert "- Hold two [1]" in done.text


def test_marker_in_details_is_not_repeated_on_the_lead():
    rows = [{"doc_name": "spec.pdf", "page_or_section": "p. 4"}, {"doc_name": "dwg.pdf", "page_or_section": "p. 9"}]
    text = ("The concrete grade is C40 (Source: spec.pdf, p. 4).\n\n"
            "Exposure class XC3 applies to the podium (Source: dwg.pdf, p. 9).")
    done = _apply(text, sources=rows)
    assert done.text.startswith("The concrete grade is C40")
    assert "[1]" in done.text and "[2]" not in done.text
    assert "[2]" in done.details


# ── where the rule does not apply ───────────────────────────────────────────

def test_computed_turn_is_not_split():
    text = "Volume = 30 x 20 x 1.5 = 900 m3\n\nThe raft dimensions come from the drawing."
    done = _apply(text, computed=True)
    assert done.details is None
    assert "The raft dimensions" in done.text


def test_asking_turn_is_not_split():
    text = "I need the wall height to continue.\n\nThe length is known from the plan."
    done = _apply(text, asking=True)
    assert done.details is None


def test_fenced_block_is_not_split():
    text = "The schedule export reads:\n\n```\nA100 Start 01-Jan\n```\n\nThat is the first activity."
    done = _apply(text)
    assert done.details is None


@pytest.mark.parametrize("text,details", [
    ("The schedule export reads A100 first.\n\nThat is the first activity.",
     "That is the first activity."),
    ("The log lists RFI-014 as the latest entry.\n\nIt was raised by the facade trade.\n\nNo reply is recorded.",
     "It was raised by the facade trade.\n\nNo reply is recorded."),
])
def test_fence_absent_splits(text, details):
    done = _apply(text)
    assert done.details == details


# ── the exit carries details beside the answer ──────────────────────────────

@pytest.mark.parametrize("answer,lead,details", [
    ("The gate is grey.\n\nThe fence is green.", "The gate is grey.", "The fence is green."),
    ("Level 3 has 42 parking bays.\n\nLevel 2 has 38.\n\nThe basement has none.",
     "Level 3 has 42 parking bays.", "Level 2 has 38.\n\nThe basement has none."),
])
def test_exit_result_carries_details(answer, lead, details):
    out = answer_exit.check_result({"answer": answer, "sources": []})
    assert out["answer"] == lead
    assert out["details"] == details


def test_exit_result_without_details_has_no_field():
    out = answer_exit.check_result({"answer": "The gate is grey.", "sources": []})
    assert "details" not in out


# ── a conflict-replaced answer is read whole: rule 5 yields ─────────────────

_CONFLICT_CASES = [
    ([{"doc_name": "Pile Cap Specification.pdf",
       "snippet": "Minimum cover to pile cap reinforcement is 75 mm."},
      {"doc_name": "Pile Cap Method Statement.pdf",
       "snippet": "Spacers give a cover to pile cap reinforcement of 50 mm."}],
     "What cover is required to the pile cap reinforcement?",
     "The pile caps are cast in grade C40 concrete.\n\n"
     "The cover to the pile cap reinforcement is 50 mm.\n\n"
     "Laps are forty bar diameters.",
     ["The pile caps are cast in grade C40 concrete.", "Laps are forty bar diameters."]),
    ([{"doc_name": "Podium Slab Specification.pdf", "snippet": "The podium slab thickness is 250 mm."},
      {"doc_name": "Podium Slab Drawing Notes.pdf", "snippet": "The podium slab thickness is 300 mm."}],
     "What is the podium slab thickness?",
     "The podium is a flat slab on drop panels.\n\n"
     "The podium slab thickness is 300 mm.\n\n"
     "Movement joints are at 30 m centres.",
     ["The podium is a flat slab on drop panels.", "Movement joints are at 30 m centres."]),
]


@pytest.mark.parametrize("rows,question,text,kept", _CONFLICT_CASES)
def test_conflict_replaced_answer_keeps_every_other_paragraph_in_the_text(rows, question, text, kept):
    done = _apply(text, sources=rows, question=question)
    assert done.conflicts
    assert done.details is None
    assert done.text.startswith("The documents differ")
    for par in kept:
        assert par in done.text


@pytest.mark.parametrize("rows,question,text,kept", _CONFLICT_CASES)
def test_same_answer_without_a_conflict_still_splits(rows, question, text, kept):
    # The other side: the same answer over documents that agree has no
    # conflict to state, so rule 5 applies and the later paragraphs move.
    agree = [dict(rows[0]), dict(rows[1], snippet=rows[0]["snippet"])]
    done = _apply(text.replace(" 50 mm", " 75 mm").replace(" 300 mm", " 250 mm"),
                  sources=agree, question=question)
    assert done.conflicts == []
    assert done.text == kept[0]
    assert kept[1] in (done.details or "")
