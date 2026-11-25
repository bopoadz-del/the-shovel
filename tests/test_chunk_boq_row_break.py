"""The ``finer`` chunker breaks ABOVE the next BOQ row.

``chunk_text_with_overlap`` documents a first-choice break: a line that opens
with an item code (``D 512.3``), so a row's code, description and rate stay in
one chunk. ``_BOQ_ITEM_PATTERN`` was compiled without ``re.M``; ``^`` then only
matches at offset 0 and ``finditer(text, soft_end, hard_end)`` never matched,
so every break fell back to ``\\n`` / ``. `` inside a row.

Every input is invented here. No document, name or figure from a corpus.
"""

from __future__ import annotations

import re

import pytest

from app.core.doc_index import chunk_text_with_overlap

_CODE_AT_START = re.compile(r"^[A-Z] \d+\.\d+")


def _bill(letter: str, section: int, count: int, first: str, second: str, third: str) -> list[str]:
    """Rows of three lines each: code + description, wrapped text, unit/qty/rate."""
    rows = []
    for k in range(count):
        rows.append(
            f"{letter} {section}.{k + 1}\t{first}\n"
            f"\t{second} {k + 3}\n"
            f"\t{third}\tm3\t{120 + k}\t{455 + k * 7}.00"
        )
    return rows


_BILLS = [
    _bill(
        "D", 512, 16,
        "Supply and place reinforced concrete in ground beams",
        "including formwork, curing and protection as specified in clause",
        "measured net, no allowance for waste",
    ),
    _bill(
        "E", 604, 16,
        "Excavate trench for drainage pipe not exceeding 2 m deep",
        "and dispose of surplus material off site, refer to note",
        "rates include trimming and compaction",
    ),
]


@pytest.mark.parametrize("rows", _BILLS, ids=["bill-D", "bill-E"])
def test_every_row_is_whole_in_some_chunk(rows: list[str]) -> None:
    text = "\n".join(rows)
    chunks = chunk_text_with_overlap(text, target_chars=500, overlap=50)
    torn = [row.split("\t")[0] for row in rows if not any(row in c for c in chunks)]
    assert torn == [], f"rows not whole in any chunk: {torn}"


@pytest.mark.parametrize("rows", _BILLS, ids=["bill-D", "bill-E"])
def test_without_overlap_each_chunk_opens_on_a_row_code(rows: list[str]) -> None:
    text = "\n".join(rows)
    chunks = chunk_text_with_overlap(text, target_chars=500, overlap=0)
    assert len(chunks) > 2
    opening = [c.splitlines()[0][:12] for c in chunks if not _CODE_AT_START.match(c)]
    assert opening == []


def _filler(word: str, count: int) -> list[str]:
    return [f"{word} line {k} of the method statement describes a hold point." for k in range(count)]


# A code that is NOT at a line start (mid-line, or a lone capital on its own
# line above a numbered line) is no row break: the chunks must be the same as
# for the same text with that letter lower-cased, which no pattern can match.
_NOT_ROW_STARTS = [
    "\n".join(_filler("Site", 7) + ["Concrete as for item D 512.4 above, see the bill."] + _filler("Site", 12)),
    "\n".join(_filler("Plant", 7) + ["B", "4.2 Scope of the temporary works package."] + _filler("Plant", 12)),
]


@pytest.mark.parametrize("text", _NOT_ROW_STARTS, ids=["code-mid-line", "letter-on-own-line"])
def test_code_not_at_line_start_is_no_row_break(text: str) -> None:
    lowered = text.replace("item D 512.4", "item d 512.4").replace("\nB\n4.2", "\nb\n4.2")
    assert lowered != text
    breaks = [len(c) for c in chunk_text_with_overlap(text, target_chars=300, overlap=50)]
    assert breaks == [len(c) for c in chunk_text_with_overlap(lowered, target_chars=300, overlap=50)]


# Text with no row codes keeps the fallback breaks: every chunk but the last
# stays at or under the soft target.
_PROSE = [
    " ".join(f"Sentence {k} explains the site logistics plan in plain words." for k in range(40)),
    "\n".join(_filler("Crane", 40)),
]


@pytest.mark.parametrize("text", _PROSE, ids=["sentences", "lines"])
def test_text_without_row_codes_keeps_fallback_breaks(text: str) -> None:
    chunks = chunk_text_with_overlap(text, target_chars=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks[:-1])


# A prose line that opens like a code but is not one: a lower-case letter, or a
# capital and a number with no dotted part. Neither is a row break: the chunks
# are those of the same text with that letter swapped for a sign no pattern
# can match (same length, so only a row break could move a boundary).
_LOOK_ALIKES = [
    ("a 2.5 m wide haul road runs along the east boundary of the site.", "a 2.5", "~ 2.5"),
    ("A 4 storey frame is braced by the core walls on each level.", "A 4 s", "~ 4 s"),
    ("b 12.75 m of kerb is renewed where the trench crosses the road.", "b 12.75", "~ 12.75"),
    ("C 30 concrete is placed in the blinding under every footing.", "C 30 c", "~ 30 c"),
]


@pytest.mark.parametrize("line, code, sign", _LOOK_ALIKES,
                         ids=["lower-case-dotted", "capital-undotted",
                              "lower-case-dotted-2", "capital-undotted-2"])
def test_a_look_alike_line_start_is_no_row_break(line: str, code: str, sign: str) -> None:
    text = "\n".join(_filler("Site", 7) + [line] + _filler("Site", 12))
    swapped = text.replace("\n" + code, "\n" + sign)
    assert swapped != text and len(swapped) == len(text)
    breaks = [len(c) for c in chunk_text_with_overlap(text, target_chars=300, overlap=50)]
    assert breaks == [len(c) for c in chunk_text_with_overlap(swapped, target_chars=300, overlap=50)]


# The same line with a real row code at its start IS a break: the control that
# shows the comparison above can see a row break at all.
@pytest.mark.parametrize("code", ["D 512.4", "E 604.12"], ids=["code-D", "code-E"])
def test_a_row_code_at_line_start_moves_the_break(code: str) -> None:
    text = "\n".join(_filler("Site", 7) + [f"{code} concrete is placed in the blinding."] + _filler("Site", 12))
    swapped = text.replace("\n" + code[0], "\n~")
    breaks = [len(c) for c in chunk_text_with_overlap(text, target_chars=300, overlap=50)]
    assert breaks != [len(c) for c in chunk_text_with_overlap(swapped, target_chars=300, overlap=50)]


# A row match starts on the row's own line, never on a blank line above it,
# so the break the chunker takes from ``match.start()`` is that line's start.
@pytest.mark.parametrize("above, row", [
    ("Bill notes end here.\n\n\n", "D 512.1\tSupply concrete"),
    ("Preliminaries close.\n  \t\n", "  E 604.3\tExcavate trench"),
], ids=["blank-lines-above", "whitespace-line-above"])
def test_a_row_match_starts_on_its_own_line(above: str, row: str) -> None:
    from app.core.doc_index import _BOQ_ITEM_PATTERN
    match = _BOQ_ITEM_PATTERN.search(above + row)
    assert match is not None
    assert match.start() == len(above)


# The gap between the letter and the number may be any run of spaces or tabs
# on the row's own line (``D  512.1``, ``E<tab>604.3``), not one space only: a
# bill typed with those gaps keeps every row whole, as the one-space bill does.
def _spaced_bill(letter: str, section: int, gap: str) -> list[str]:
    return [row.replace(f"{letter} {section}.", f"{letter}{gap}{section}.", 1)
            for row in _bill(letter, section, 16,
                             "Supply and fix precast kerb units on a concrete bed",
                             "including haunching and jointing as specified in clause",
                             "measured along the face of the kerb")]


@pytest.mark.parametrize("letter, section, gap", [
    ("D", 512, "  "), ("D", 512, "\t"), ("E", 604, "   "), ("E", 604, " \t"),
], ids=["two-spaces", "tab", "three-spaces", "space-tab"])
def test_a_wide_gap_after_the_letter_keeps_every_row_whole(letter: str, section: int, gap: str) -> None:
    rows = _spaced_bill(letter, section, gap)
    assert all(row.startswith(f"{letter}{gap}{section}.") for row in rows)
    chunks = chunk_text_with_overlap("\n".join(rows), target_chars=500, overlap=50)
    torn = [row.split("\n")[0][:12] for row in rows if not any(row in c for c in chunks)]
    assert torn == [], f"rows not whole in any chunk: {torn}"


@pytest.mark.parametrize("code", ["D  512.4", "E\t604.12"], ids=["two-spaces", "tab"])
def test_a_wide_gap_row_code_at_line_start_moves_the_break(code: str) -> None:
    text = "\n".join(_filler("Site", 7) + [f"{code} concrete is placed in the blinding."] + _filler("Site", 12))
    swapped = text.replace("\n" + code[0], "\n~")
    breaks = [len(c) for c in chunk_text_with_overlap(text, target_chars=300, overlap=50)]
    assert breaks != [len(c) for c in chunk_text_with_overlap(swapped, target_chars=300, overlap=50)]


# A dotted number glued to a word (``D 512.4mm``) is a size, not an item code:
# the code must end at a word boundary, so that line is no row break.
@pytest.mark.parametrize("line, code", [
    ("D 512.4mm bars are lapped at every construction joint in the wall.", "D 512.4mm"),
    ("E 60.25x the bar diameter is the lap length used in the slab.", "E 60.25x"),
], ids=["unit-glued", "factor-glued"])
def test_a_number_glued_to_a_word_is_no_row_break(line: str, code: str) -> None:
    text = "\n".join(_filler("Site", 7) + [line] + _filler("Site", 12))
    swapped = text.replace("\n" + code[0], "\n~")
    assert swapped != text and len(swapped) == len(text)
    breaks = [len(c) for c in chunk_text_with_overlap(text, target_chars=300, overlap=50)]
    assert breaks == [len(c) for c in chunk_text_with_overlap(swapped, target_chars=300, overlap=50)]
