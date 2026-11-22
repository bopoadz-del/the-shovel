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
