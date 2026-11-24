"""A merge never refills the heading room the split kept for a piece (RQ-2).

``bound_chunks`` cuts a split chunk at ``ceiling - HEADING_RESERVE`` when the
caller will put a page heading on each piece. The floor merge then joined a
short tail piece back onto its neighbour up to the full ceiling, so the piece
had no room left and ``attach_page_headings`` left its heading off.

Every text here is invented; nothing names a document.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import (
    HEADING_RESERVE,
    _merge_pairs,
    attach_page_headings,
    bound_chunks,
)

HEADING = "7.3 INVENTED SECTION HEADING FOR PIECES"


def _page(sentences: int, tail_chars: int) -> str:
    body = " ".join(
        f"Clause {i} of this made-up page states one requirement." for i in range(sentences)
    )
    return f"{body} {'q' * tail_chars}."


@pytest.fixture(autouse=True)
def _bounds(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "2000")
    monkeypatch.setenv("CHUNK_MIN_CHARS", "200")


def _headed_pieces(text: str) -> list[str]:
    parents: list[int | None] = []
    chunks = bound_chunks([text], split_parents=parents)
    out, _ = attach_page_headings(chunks, [1] * len(chunks), parents, {1: HEADING})
    return [c for c, p in zip(out, parents) if p is not None]


@pytest.mark.parametrize(("sentences", "tail"), [(68, 100), (69, 50)])
def test_every_split_piece_keeps_its_heading(sentences, tail):
    pieces = _headed_pieces(_page(sentences, tail))
    assert len(pieces) >= 2
    assert all(p.startswith(HEADING + "\n") for p in pieces)
    assert all(len(p) <= 2000 for p in pieces)


@pytest.mark.parametrize("tail", [60, 100])
def test_a_tail_too_long_for_the_piece_room_stays_its_own_chunk(tail):
    # A piece at the cut plus a short tail fits the ceiling but not the cut.
    cut = 2000 - HEADING_RESERVE
    pairs = [("a" * (cut - 10), 5), ("b" * tail, 5)]
    out = _merge_pairs(pairs, 200, 2000, piece_ceiling=cut)
    assert [len(c) for c, _ in out] == [cut - 10, tail]


@pytest.mark.parametrize("tail", [5, 8])
def test_a_tail_that_fits_the_piece_room_is_still_joined(tail):
    cut = 2000 - HEADING_RESERVE
    pairs = [("a" * (cut - 20), 5), ("b" * tail, 5)]
    out = _merge_pairs(pairs, 200, 2000, piece_ceiling=cut)
    assert len(out) == 1 and len(out[0][0]) == cut - 20 + 1 + tail


@pytest.mark.parametrize("size", [1700, 1850])
def test_unsplit_chunks_still_merge_up_to_the_ceiling(size):
    # No parent id: no heading is attached, so the full ceiling applies.
    pairs = [("c" * size, None), ("d" * 120, None)]
    out = _merge_pairs(pairs, 200, 2000, piece_ceiling=2000 - HEADING_RESERVE)
    assert len(out) == 1 and out[0][1] is None


@pytest.mark.parametrize("size", [1850, 1870])
def test_a_short_head_joined_onto_a_split_piece_respects_the_room(size):
    # The short chunk comes first and the join keeps the next piece's parent.
    cut = 2000 - HEADING_RESERVE
    pairs = [("e" * 120, None), ("f" * size, 9)]
    out = _merge_pairs(pairs, 200, 2000, piece_ceiling=cut)
    assert [p for _, p in out] == [None, 9]


@pytest.mark.parametrize("size", [1850, 1870])
def test_without_a_piece_ceiling_the_old_ceiling_is_kept(size):
    # Callers that attach no heading (no split_parents) keep merging as before.
    pairs = [("g" * size, 3), ("h" * 100, 3)]
    out = _merge_pairs(pairs, 200, 2000)
    assert len(out) == 1
