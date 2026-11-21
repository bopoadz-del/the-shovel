"""A written name split across two adjacent markdown table cells.

A table prints one record across cells, and the column rule falls between the
words of a person's name as easily as between a name and a job title:
``| Surname | Given | Engineer |`` is a byline set in columns. The run reader
splits tokens on the separators a name is written with -- space, tab, dash --
and the cell bar is not one of them, so each cell was a one-word run and the
name scored zero across every cell boundary.

The bar still cannot join the separator set: in code and prose it separates a
pipeline or a boolean alternative, and nobody writes a name there. So a line
that is a table row is read a second time with its bars read as the spaces
between cells -- and only that line. The phrase is built from one physical
line, so two rows are never joined across the line break between them; that
is measured below, not promised.

What is *not* widened: a name piece in one row and its other piece in the next
row stays a miss, a single cell stays a single run, and a pipeline on one line
is not a table row. The words are invented and assembled at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_table_cell", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_table_cell"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


# -- the spellings a table writes a name in ----------------------------------
CELL_FORMS = [
    ("surname first, given in the next cell", f"| {LAST} | {FIRST} | Engineer |"),
    ("given first, surname in the next cell", f"| {FIRST} | {LAST} | Engineer |"),
    ("no spaces around the bars", f"|{LAST}|{FIRST}|"),
    ("given name as an initial", f"| Q. | {LAST} | Engineer |"),
    ("initial in the first cell", f"| Q. | {LAST} |"),
    ("the name in the middle of a record", f"| 12 | {LAST} | {FIRST} | Engineer |"),
    ("a row inside a longer table", f"| a | b |\n| {LAST} | {FIRST} |\n| c | d |"),
]


CONTROLS = [
    ("two rows, given then surname", f"| {FIRST} | x |\n| {LAST} | y |"),
    ("two rows, surname then given", f"| {LAST} | x |\n| {FIRST} | y |"),
    ("two rows split by a blank line", f"| {FIRST} | x |\n\n| {LAST} | y |"),
    ("one cell carrying the surname", f"| {LAST} | x |"),
    ("one cell carrying the given name", f"| {FIRST} | x |"),
    ("a different surname in the next cell", f"| {FIRST} | Ashdown | x |"),
    ("a different given name in the next cell", f"| Bramwell | {LAST} | x |"),
    ("unrelated titlecase cells", "| Rate | Card | x |"),
    ("a pipeline on one line", f"echo {FIRST} | tee | grep {LAST}"),
    ("prose carrying a pipe", f"{LAST} | {FIRST} was written down"),
]


@pytest.mark.parametrize(
    "label, text", CELL_FORMS, ids=[row[0] for row in CELL_FORMS]
)
def test_a_name_split_across_cells_is_read(label, text):
    """The column rule is layout; the words either side of it are the name."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", CELL_FORMS, ids=[row[0] for row in CELL_FORMS]
)
def test_the_inverted_entry_reads_the_cells_too(label, text):
    """The entry's word order cannot decide whether a cell boundary is seen."""
    assert _hits(text, INVERTED), label


@pytest.mark.parametrize(
    "label, text", CONTROLS, ids=[row[0] for row in CONTROLS]
)
@pytest.mark.parametrize("entry", [FULL, INVERTED], ids=["full", "inverted"])
def test_nothing_else_is_assembled_out_of_cells(label, text, entry):
    """Two rows, one cell, or a pipeline are not one written name."""
    assert _hits(text, entry) == [], label


def test_a_hit_reports_the_line_of_its_own_row():
    """A name on the second row is reported on the second line, not the first."""
    text = f"| a | b |\n| {LAST} | {FIRST} |"
    assert [line for line, _ in _hits(text)] == [2]


# -- the piece the rule is made of -------------------------------------------
ROW_PHRASES = [
    ("a two-cell row", f"| {LAST} | {FIRST} |", [LAST, FIRST]),
    ("a three-cell row", "| a | b | c |", ["a", "b", "c"]),
    ("a single-cell row is not a row", f"| {LAST} |", None),
    ("a row without its closing bar", "| a | b", None),
    ("a pipeline is not a row", "echo a | b", None),
    ("prose with a bar is not a row", "a | b", None),
]


@pytest.mark.parametrize(
    "label, line, expected", ROW_PHRASES, ids=[row[0] for row in ROW_PHRASES]
)
def test_only_a_table_row_is_read_as_cells(label, line, expected):
    """The row shape is markdown's; nothing else is joined."""
    phrase = scan_secrets._table_row_phrase(line)
    assert (phrase.split() if phrase is not None else None) == expected, label


def test_the_phrase_never_spans_more_than_one_line():
    """The mechanism's safety is structural: one line in, one line out."""
    text = f"| {FIRST} | x |\n| {LAST} | y |"
    for line in text.splitlines():
        phrase = scan_secrets._table_row_phrase(line)
        assert phrase is not None
        assert "\n" not in phrase
