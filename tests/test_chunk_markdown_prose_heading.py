"""A knowledge-note prose piece opens on the heading it sits under.

``chunk_markdown`` char-chunks the prose between tables. Only the first piece
of a section held its ``#`` line, so every later piece of a long section
reached the index with no word of the section it belongs to (on main, 290 of
717 knowledge-note chunks held no heading line). Each prose piece whose first
line is not a heading now gets the heading in force where it starts.

Every input is invented here; each decision is checked on two inputs.
"""
from __future__ import annotations

import pytest

from app.core.doc_index import chunk_markdown, chunk_text_with_overlap


def _prose(word: str, sentences: int) -> str:
    return " ".join(
        f"The {word} clause number {i} sets a duty that the {word} party keeps."
        for i in range(sentences)
    )


def _first_line(chunk: str) -> str:
    return chunk.splitlines()[0].strip()


LONG_SECTIONS = [
    ("## Retention release", "retention"),
    ("### Pump curve checks", "impeller"),
]


@pytest.mark.parametrize("heading,word", LONG_SECTIONS)
def test_every_piece_of_a_long_section_opens_on_its_heading(heading, word):
    note = f"# Note\n\n{heading}\n\n{_prose(word, 40)}\n"
    chunks = [c for c in chunk_markdown(note) if word in c]
    assert len(chunks) >= 3
    assert heading in chunks[0].splitlines()
    assert all(_first_line(c) == heading for c in chunks[1:])


@pytest.mark.parametrize("heading,word", LONG_SECTIONS)
def test_a_piece_is_its_old_text_behind_the_heading(heading, word):
    body = f"{heading}\n\n{_prose(word, 40)}"
    old = chunk_text_with_overlap(body, target_chars=500, overlap=50)
    new = chunk_markdown(body)
    assert len(new) == len(old)
    assert new[0] == old[0]
    for was, now in zip(old[1:], new[1:]):
        assert now == f"{heading}\n{was}"


@pytest.mark.parametrize("heading,word", LONG_SECTIONS)
def test_a_piece_that_opens_on_a_heading_gets_no_second_one(heading, word):
    note = f"{heading}\n\n{_prose(word, 3)}\n"
    chunks = chunk_markdown(note)
    assert len(chunks) == 1
    assert chunks[0].count(heading) == 1


TABLE_THEN_PROSE = [
    ("## Bond values", "| Item | Value |\n|---|---|\n| bond | 10 |", "surety"),
    ("#### Crane limits", "| Load | Radius |\n| :-- | --: |\n| 5 t | 12 m |", "outrigger"),
]


@pytest.mark.parametrize("heading,table,word", TABLE_THEN_PROSE)
def test_prose_after_a_table_opens_on_the_section_heading(heading, table, word):
    note = f"{heading}\n\n{table}\n\n{_prose(word, 2)}\n"
    chunks = chunk_markdown(note)
    prose = [c for c in chunks if word in c and "|" not in c]
    assert len(prose) == 1
    assert _first_line(prose[0]) == heading


@pytest.mark.parametrize("word", ["preamble", "foreword"])
def test_text_before_any_heading_is_left_as_is(word):
    note = _prose(word, 30)
    old = chunk_text_with_overlap(note, target_chars=500, overlap=50)
    assert chunk_markdown(note) == old
    assert len(old) >= 2


SECTION_PAIRS = [
    ("## Alpha duties", "alpha", "## Beta duties", "beta"),
    ("### Hoist checks", "hoist", "### Sling checks", "sling"),
]


@pytest.mark.parametrize("h1,w1,h2,w2", SECTION_PAIRS)
def test_a_piece_takes_the_heading_in_force_where_it_starts(h1, w1, h2, w2):
    note = f"{h1}\n\n{_prose(w1, 25)}\n\n{h2}\n\n{_prose(w2, 25)}\n"
    for chunk in chunk_markdown(note):
        lines = chunk.splitlines()
        first, body = lines[0].strip(), "\n".join(lines[1:])
        assert first in (h1, h2)
        # The text right under the lead belongs to the section the lead names:
        # of the two section words, the one met first is that section's.
        at1, at2 = body.find(f" {w1} "), body.find(f" {w2} ")
        hits = [(at, w) for at, w in ((at1, w1), (at2, w2)) if at >= 0]
        assert hits
        assert min(hits)[1] == (w1 if first == h1 else w2)


NEXT_SECTION_AFTER_TABLE = [
    ("## Bond values", "| Item | Value |\n|---|---|\n| bond | 10 |", "## Insurance", "insurer"),
    ("### Crane limits", "| Load | Radius |\n|---|---|\n| 5 t | 12 m |", "### Rigging", "shackle"),
]


@pytest.mark.parametrize("h1,table,h2,word", NEXT_SECTION_AFTER_TABLE)
def test_a_piece_that_opens_on_a_new_heading_keeps_only_that_heading(h1, table, h2, word):
    # The prose after the table opens on h2, while h1 is the heading in force
    # where that prose segment begins: h1 must not be put in front of h2.
    note = f"{h1}\n\n{table}\n\n{h2}\n\n{_prose(word, 2)}\n"
    prose = [c for c in chunk_markdown(note) if word in c]
    assert len(prose) == 1
    assert _first_line(prose[0]) == h2
    assert h1 not in prose[0]
