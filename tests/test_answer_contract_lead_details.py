"""Rule 5 of the answer contract: the lead is the answer, the rest is detail.

Every input is built here; each decision is checked from both sides. No word
count is asserted as a limit -- the rule reads the answer's shape.
"""

from __future__ import annotations

from app.agents import answer_contract, answer_exit


def _apply(text: str, **kw):
    return answer_contract.apply(text, **kw)


def _words(text: str) -> set[str]:
    return set(text.replace("\n", " ").split())


# ── later paragraphs leave the lead, word for word ─────────────────────────

def test_later_paragraphs_move_to_details_lookup():
    text = ("The retaining wall footing is 600 mm deep.\n\n"
            "The drawings also show a 75 mm blinding layer under the footing.\n\n"
            "Reinforcement laps follow the general notes.")
    done = _apply(text)
    assert done.text == "The retaining wall footing is 600 mm deep."
    assert done.details == ("The drawings also show a 75 mm blinding layer under the footing.\n\n"
                            "Reinforcement laps follow the general notes.")


def test_later_paragraphs_move_to_details_not_found():
    text = ("No document in the project states the paint colour for the stair core.\n\n"
            "The finishes schedule covers the lobby and the corridors only.")
    done = _apply(text)
    assert done.text.startswith("No document in the project states")
    assert done.details == "The finishes schedule covers the lobby and the corridors only."


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

def test_introduced_list_stays_with_items_cut():
    text = ("The submittals due this month are:\n\n"
            "- Fire stopping. The sample was rejected once and is being resubmitted.\n"
            "- Door hardware\n\n"
            "Two further submittals are due next month.")
    done = _apply(text)
    assert done.text == ("The submittals due this month are:\n\n"
                         "- Fire stopping.\n"
                         "- Door hardware")
    assert "- Fire stopping. The sample was rejected once and is being resubmitted." in done.details
    assert "Two further submittals are due next month." in done.details


def test_list_not_introduced_by_the_lead_moves():
    text = ("The pump room is on basement level 1.\n\n"
            "- Access is from the north stair.\n"
            "- The door is 1.2 m wide.")
    done = _apply(text)
    assert done.text == "The pump room is on basement level 1."
    assert done.details.startswith("- Access is from the north stair.")


def test_opening_heading_keeps_its_whole_section():
    text = ("## Answer\n\n"
            "The cover to the raft is 75 mm.\n\n"
            "The piling cap cover is also stated as 75 mm in the same note.\n\n"
            "## Background\n\n"
            "The structural notes were revised twice.")
    done = _apply(text)
    assert done.text == ("## Answer\n\nThe cover to the raft is 75 mm.\n\n"
                         "The piling cap cover is also stated as 75 mm in the same note.")
    assert done.details == "## Background\n\nThe structural notes were revised twice."


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


def test_fence_absent_splits():
    text = "The schedule export reads A100 first.\n\nThat is the first activity."
    done = _apply(text)
    assert done.details == "That is the first activity."


# ── the exit carries details beside the answer ──────────────────────────────

def test_exit_result_carries_details():
    out = answer_exit.check_result({"answer": "The gate is grey.\n\nThe fence is green.", "sources": []})
    assert out["answer"] == "The gate is grey."
    assert out["details"] == "The fence is green."


def test_exit_result_without_details_has_no_field():
    out = answer_exit.check_result({"answer": "The gate is grey.", "sources": []})
    assert "details" not in out
