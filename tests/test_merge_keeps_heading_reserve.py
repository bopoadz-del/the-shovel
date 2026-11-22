"""A floor merge keeps the room the cut left for a split piece's heading.

``bound_chunks`` cuts an oversize chunk at ``ceiling - HEADING_RESERVE`` when
its caller files every piece under its page's heading. The floor merge then
joined a short neighbour onto a piece against the full ceiling, so the joined
piece could fill the reserve; ``attach_page_headings`` keeps the ceiling and
left the heading off that piece, which was then filed under nothing. A joined
pair that stays a split piece now has to fit the cut. Unsplit chunks and
callers that attach no heading keep the full ceiling. Every input below is
invented; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import HEADING_RESERVE, attach_page_headings, bound_chunks

CEILING = 2000
FLOOR = 200


@pytest.fixture(autouse=True)
def _bounds(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_CHARS", str(CEILING))
    monkeypatch.setenv("CHUNK_MIN_CHARS", str(FLOOR))


def _section(words: int, clauses: int = 9) -> str:
    return "\n\n".join(
        (f"Clause {n} text " + "word " * words).strip() + "." for n in range(clauses)
    )


def _note(repeats: int) -> str:
    return ("Note: " + "the general requirements of the preceding part apply " * repeats).strip()


def _headed(chunks: list[str], heading: str) -> tuple[list[str], list[int | None]]:
    parents: list[int | None] = []
    out = bound_chunks(chunks, split_parents=parents)
    headed, _ = attach_page_headings(out, [1] * len(out), parents, {1: heading})
    return headed, parents


# A note under the floor before a long section; the long heading is what the
# reserve was kept for. Two section shapes so the first piece lands near the cut.
@pytest.mark.parametrize("words", [70, 88])
def test_every_split_piece_keeps_its_heading_after_a_merge(words):
    heading = "4.0 SECTION 4 IN-SITU CONCRETE WORKS AND FORMWORK TO SUBSTRUCTURE"
    note = _note(3)
    assert len(note) < FLOOR
    headed, parents = _headed([note, _section(words)], heading)
    pieces = [c for c, p in zip(headed, parents) if p is not None]
    assert len(pieces) >= 2
    assert all(c.startswith(heading + "\n") for c in pieces)
    assert all(len(c) <= CEILING for c in headed)
    # The note is kept exactly once: as its own chunk when joining it would
    # have taken the heading's room.
    assert sum(note in c for c in headed) == 1


@pytest.mark.parametrize("repeats,words", [(1, 60), (1, 80)])
def test_a_short_note_that_leaves_room_still_merges_into_the_piece(repeats, words):
    heading = "2.0 BLOCKWORK"
    note = _note(repeats)
    headed, parents = _headed([note, _section(words)], heading)
    assert parents[0] is not None
    assert note in headed[0]
    assert headed[0].startswith(heading + "\n")


@pytest.mark.parametrize("words", [70, 88])
def test_without_headings_the_merge_keeps_the_full_ceiling(words):
    note = _note(3)
    on_main = bound_chunks([note, _section(words)])
    # Plain callers (no split_parents) cut at the full ceiling and merge
    # against it: the note joins the first piece as before.
    assert note in on_main[0]
    assert all(len(c) <= CEILING for c in on_main)


TAIL_HEADING = "7.0 DOORS AND IRONMONGERY SCHEDULE FOR ALL INTERNAL AND EXTERNAL OPENINGS"


def _with_tail(words: int, tail_words: int) -> str:
    return _section(words, clauses=2) + "\n\n" + ("Tail " + "word " * tail_words).strip() + "."


@pytest.mark.parametrize("tail_words", [20, 30])
def test_a_short_tail_piece_is_not_merged_into_the_reserve(tail_words):
    headed, parents = _headed([_with_tail(364, tail_words)], TAIL_HEADING)
    cut = CEILING - HEADING_RESERVE
    assert all(p is not None for p in parents)
    assert all(c.startswith(TAIL_HEADING + "\n") for c in headed)
    assert all(len(c) - len(TAIL_HEADING) - 1 <= cut for c in headed)
    assert all(len(c) <= CEILING for c in headed)


# The other side of the tail decision: a short trailer after an unsplit chunk
# gets no heading, so it still joins up to the full ceiling (past the cut).
@pytest.mark.parametrize("words", [360, 370])
def test_a_short_trailer_after_an_unsplit_chunk_still_merges_to_the_ceiling(words):
    unit = ("Schedule row " + "word " * words).strip() + "."
    trailer = ("Trailer " + "word " * 20).strip() + "."
    headed, parents = _headed([unit, trailer], TAIL_HEADING)
    assert CEILING - HEADING_RESERVE < len(unit) + 1 + len(trailer) <= CEILING
    assert headed == [unit + "\n" + trailer]
    assert parents == [None]
