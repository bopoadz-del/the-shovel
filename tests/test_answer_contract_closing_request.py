"""A closing paragraph that only asks the reader for a next step is padding.

Rule 2 dropped an offer only when a sentence opened on an offer phrase
("Let me know ...", "If you'd like ..."). A document answer could still close
on a paragraph that asks the reader to supply something -- "If the pipework
is billed in another file, point me at it and I'll pull the line" -- because
that sentence opens on a condition about the documents, not on "if you".
The contract now reads the closing paragraph of every answer:

* when every sentence asks the reader for a next step (an imperative to
  supply or direct something that names the writer as receiver, or the
  writer's offer conditional on the reader) and it states no figure and no
  marker, it goes;
* a closing paragraph with a figure or a citation marker, an instruction
  that is the answer itself, a paragraph that also states something, an
  answer that is nothing but the request, and an ``asking`` turn stay.

Every input here is written by the test; none is a project file or an exam
question.
"""
from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.answer_contract import _drop_closing_request

# ── a closing request paragraph with no figure and no marker: it goes ───────

@pytest.mark.parametrize("text, kept", [
    (("The lobby floor is honed granite.\n\n"
      "If the stair finishes are scheduled in a separate register, point me at that file and I'll pull the line."),
     "The lobby floor is honed granite."),
    ("Fire doors are FD60 throughout.\n\nUpload the door hardware schedule and I'll check the closers.",
     "Fire doors are FD60 throughout."),
    (("The roof membrane is a single-ply PVC sheet.\n\n"
      "Please share the manufacturer's data sheet with me. I can confirm the warranty term if you send it."),
     "The roof membrane is a single-ply PVC sheet."),
    ("The facade is unitised curtain wall.\n\nLet me know which elevation you need.",
     "The facade is unitised curtain wall."),
])
def test_closing_request_paragraph_goes(text, kept):
    assert _drop_closing_request(text) == kept


@pytest.mark.parametrize("text, kept", [
    (("The ceiling void is used as a return-air plenum.\n\n"
      "Once the mechanical drawings are uploaded, tell me which floor and I'll trace the ductwork."),
     "The ceiling void is used as a return-air plenum."),
    (("The kitchen floor is quarry tile.\n\n"
      "If the drainage layout is on another sheet, please send it to me and I'll check the falls."),
     "The kitchen floor is quarry tile."),
])
def test_closing_request_goes_through_apply(text, kept):
    assert answer_contract.apply(text).text == kept


@pytest.mark.parametrize("text, kept", [
    (("Glazing is laminated to the atrium.\n\n"
      "Send me the glass schedule.\n\n"
      "If the canopy is detailed elsewhere, point me at that sheet."),
     "Glazing is laminated to the atrium."),
    (("Handrails are stainless steel.\n\n"
      "Let me know which stair you mean.\n\n"
      "I can check the fixings if you share the balustrade drawings."),
     "Handrails are stainless steel."),
])
def test_two_closing_request_paragraphs_both_go(text, kept):
    assert _drop_closing_request(text) == kept


# ── the other side: what stays ──────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    # a figure in the closing paragraph
    "Slab is post-tensioned.\n\nSend me sheet 14 and I'll check the tendon layout.",
    # a citation marker in the closing paragraph
    "Slab is post-tensioned.\n\nThe tendon profile is on the structural set [1]; send me that file and I'll read it.",
    # an instruction that is the answer itself, not a request for the writer
    "Raise the query as an RFI.\n\nSend it to the Engineer through the document control system.",
    # the closing paragraph also states something
    "The parapet is precast.\n\nThe coping detail is not in the documents. Upload the facade set and I'll check it.",
    # the request is the whole answer
    "Upload the priced bill and I'll compare the rates.",
])
def test_closing_paragraph_that_stays(text):
    assert _drop_closing_request(text) == text.strip()


@pytest.mark.parametrize("text, wanted", [
    (("The pump duty depends on the static head.\n\n"
      "Tell me the building height and I'll size the pump."), "Tell me the building height"),
    (("The cable size depends on the run length.\n\n"
      "Give me the distance to the board and I'll size the cable."), "Give me the distance to the board"),
])
def test_asking_turn_keeps_its_request(text, wanted):
    assert wanted in answer_contract.apply(text, asking=True).text


@pytest.mark.parametrize("text", [
    # Only a closing paragraph is padding: a request that is followed by
    # more of the answer is not dropped by this rule.
    "Upload the door schedule and I'll check it.\n\nFire doors are fire-rated throughout.",
    "Point me at the finishes register.\n\nThe lobby floor is honed granite.",
])
def test_request_paragraph_before_the_answer_stays(text):
    assert _drop_closing_request(text) == text
