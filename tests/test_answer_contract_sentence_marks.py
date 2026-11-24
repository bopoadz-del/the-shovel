"""The answer contract splits sentences outside paired brackets and quotes.

Rules 2 and 3 (padding and machinery) and the conflict rule keep or drop
WHOLE sentences. A split inside a bracket or a quotation (at the stop of
"e.g." or "rev.") let a rule drop the head of a sentence and keep its tail,
so an answer could begin mid-clause: `"Deck A - lump sum") rather than ...`.
The sentence is now read outside paired marks, so a rule drops or keeps the
whole sentence and no fragment can become the first line. Every input here
is written by the test; none is a project file or an exam question.
"""
from __future__ import annotations

import pytest

from app.agents import answer_contract


def _first_line(text: str) -> str:
    return next(ln.strip() for ln in text.split("\n") if ln.strip())


# ── a stop outside any mark still ends the sentence ─────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("The slab is 12.5 m wide. See p.12 for the grid. Done!",
     ["The slab is 12.5 m wide.", "See p.12 for the grid.", "Done!"]),
    ("Is the kerb precast? Yes. It is 150 mm high.",
     ["Is the kerb precast?", "Yes.", "It is 150 mm high."]),
])
def test_stop_outside_marks_ends_sentence(text, expected):
    assert [s.strip() for s in answer_contract._sentences(text)] == expected


# ── a stop inside a paired bracket or quotation does not ────────────────────

@pytest.mark.parametrize("text, expected", [
    ('The bill has lump sums only (e.g. "Deck A. lump", "Deck B") rather than a rate. Next.',
     ['The bill has lump sums only (e.g. "Deck A. lump", "Deck B") rather than a rate.', "Next."]),
    ("Cover is 40 mm [see Sheet 4, rev. C] on the pier. Next.",
     ["Cover is 40 mm [see Sheet 4, rev. C] on the pier.", "Next."]),
    ('The note reads "Do not cut. Core only" at the wall. Next.',
     ['The note reads "Do not cut. Core only" at the wall.', "Next."]),
    ("The label says “Keep clear. Live” on the door. Next.",
     ["The label says “Keep clear. Live” on the door.", "Next."]),
])
def test_stop_inside_paired_mark_does_not_end_sentence(text, expected):
    assert [s.strip() for s in answer_contract._sentences(text)] == expected


# ── an unpaired mark is ordinary punctuation, never a sentence-long hold ────

@pytest.mark.parametrize("text, expected", [
    ("Item (a is open. The rest is fine. Done.",
     ["Item (a is open.", "The rest is fine.", "Done."]),
    ('He wrote "keep. It stays. Done.',
     ['He wrote "keep.', "It stays.", "Done."]),
])
def test_unpaired_mark_does_not_hold(text, expected):
    assert [s.strip() for s in answer_contract._sentences(text)] == expected


@pytest.mark.parametrize("text, opened, unopened", [
    ("a (b) [c] “d” \"e\"", {"(", "[", "“", '"'}, set()),
    ("a (b [c “d \"e", set(), {"(", "[", "“", '"'}),
])
def test_paired_marks(text, opened, unopened):
    got = answer_contract._paired(text)
    assert opened <= got and not (unopened & got)


# ── a closing mark right after a stop ends the sentence ─────────────────────

@pytest.mark.parametrize("text, expected", [
    ('The sign said "Stop." Then the gate.', ['The sign said "Stop."', "Then the gate."]),
    ("(See the note.) The pile is 600 mm.", ["(See the note.)", "The pile is 600 mm."]),
])
def test_close_after_stop_ends_sentence(text, expected):
    assert [s.strip() for s in answer_contract._sentences(text)] == expected


# ── through the contract: no rule leaves a fragment as the first line ───────

@pytest.mark.parametrize("text, first", [
    # a machinery-only sentence with a bracketed example goes WHOLE
    ('I searched the project documents for deck tiling (e.g. "Deck A - lump", "Deck B - lump") '
     "rather than a measured quantity.\n\nNo measured deck tiling quantity is in the bill.",
     # the absence lead that is left is turned to the contract form (absence_names_thing)
     "A measured deck tiling quantity is not in the bill."),
    ("I checked the documents on file for the kerb mix [see the general notes, rev. C] in full.\n\n"
     "The kerb mix is not stated in the documents.",
     "The kerb mix is not stated in the documents."),
])
def test_machinery_sentence_goes_whole(text, first):
    out = answer_contract.apply(text).text
    assert _first_line(out) == first


@pytest.mark.parametrize("text, first", [
    # an offer whose example carries a stop goes WHOLE
    ("Let me know the bill (e.g. Bill No. 4 or 5) and I will read it. The pile is 600 mm.",
     "The pile is 600 mm."),
    ('Let me know which sheet ("Sheet 4. rev C" or "Sheet 5") you need. The wall is 300 mm.',
     "The wall is 300 mm."),
])
def test_offer_sentence_goes_whole(text, first):
    out = answer_contract.apply(text).text
    assert _first_line(out) == first


# ── a last sentence with no stop of its own is still a sentence ─────────────

@pytest.mark.parametrize("text, expected", [
    ("The pile is 600 mm. Cut-off level to follow", ["The pile is 600 mm.", "Cut-off level to follow"]),
    ("No stop at all (see note. A)", ["No stop at all (see note. A)"]),
])
def test_tail_without_stop_is_kept(text, expected):
    assert [s.strip() for s in answer_contract._sentences(text)] == expected
