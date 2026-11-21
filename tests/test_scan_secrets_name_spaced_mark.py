"""A markdown link whose layout carries a space of its own.

A link's mark is not always welded to the last word of its label. A destination
may name a title after the address -- ``[Q. Surname](url "note")`` -- and a
reference may be spelled as a phrase -- ``[Q. Surname][see note]``. The mark's
inside then holds the space the tokeniser splits on, so the token that ends the
label keeps only the mark's first character, and the surname arrives as
``Surname](url`` or ``Surname][see``: name-shaped in no case at all, so the run
reader ends at the name and loses every spelling it exists to read.

The mark is layout, exactly as the welded destination and reference already
are: it is found on the block, by balancing the pair that opens it, and taken
off before any token is built. The label's closing bracket goes with it, and
what follows the mark stays, so a sentence's comma or full stop still ends the
entry and a possessive clitic still attaches to the word before it.

What is *not* widened: nothing is assembled out of a destination or a
reference, a mark that never closes is punctuation and not a link, a mark that
spans a line break is not joined, and a name split across two bracket groups
is still two words and not one name. The words are invented and assembled at
runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_spaced_mark", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_spaced_mark"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"
URL = "https://example.invalid/doc"


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


def _runs(text: str) -> list[list[str]]:
    return [words for _, words in scan_secrets.name_runs(text)]


# -- a destination that carries a quoted title ---------------------------------
TITLE_FORMS = [
    ("label, given name as an initial", f'[Q. {LAST}]({URL} "note")'),
    ("label, reversed", f'[{INVERTED}]({URL} "note")'),
    ("label, plural after an initial", f'[Q. {LAST}s]({URL} "note")'),
    ("label, all capitals", f'[Q. {LAST.upper()}]({URL} "note")'),
    ("label in a sentence", f'the crew of [Q. {LAST}]({URL} "note") mobilised'),
    ("link closed by a sentence comma", f'see [Q. {LAST}]({URL} "note"), then'),
    ("link closed by a full stop", f'[Q. {LAST}]({URL} "note").'),
    ("image alt label", f'![Q. {LAST}]({URL} "note")'),
    ("label in a table cell", f'| [Q. {LAST}]({URL} "note") | note |'),
]

# -- a reference spelled as a phrase ------------------------------------------
SPACED_REFERENCE_FORMS = [
    ("label, given name as an initial", f"[Q. {LAST}][see note]"),
    ("label, reversed", f"[{INVERTED}][see note]"),
    ("label, plural after an initial", f"[Q. {LAST}s][see note]"),
    ("label, all capitals", f"[Q. {LAST.upper()}][see note]"),
    ("label in a sentence", f"the crew of [Q. {LAST}][see note] mobilised"),
    ("link closed by a sentence comma", f"see [Q. {LAST}][see note], then"),
    ("link closed by a full stop", f"[Q. {LAST}][see note]."),
    ("image alt label", f"![Q. {LAST}][see note]"),
    ("label in a table cell", f"| [Q. {LAST}][see note] | note |"),
    ("reference with a bracketed note", f"[Q. {LAST}][see [1] note]"),
]

# The forms whose last name word is welded to the mark need the run reader;
# the literal phrase covers the rest. Both marks are taken the same way.
RUN_FORMS = [
    ("initial, destination with a title", f'[Q. {LAST}]({URL} "note")'),
    ("reversed, destination with a title", f'[{LAST}, Q.]({URL} "note")'),
    ("initial, spaced reference", f"[Q. {LAST}][see note]"),
    ("reversed, spaced reference", f"[{LAST}, Q.][see note]"),
    ("plural, spaced reference", f"[Q. {LAST}s][see note]"),
]

CONTROLS = [
    ("one-word label", f'[{LAST}]({URL} "note")'),
    ("one-word label, spaced reference", f"[{LAST}][see note]"),
    ("longer surname in the label", f'[Q. {LAST}field]({URL} "note")'),
    ("another person in the label", f'[Q. Ashdown]({URL} "note")'),
    ("unbalanced destination is not a link", f'[Q. {LAST}]({URL} "note"'),
    ("unbalanced reference is not a link", f"[Q. {LAST}][see note"),
    ("ordinary double index", f"array[{LAST}][0]"),
    ("name split across two bracket groups", f"[{FIRST}][{LAST}]"),
]


@pytest.mark.parametrize(
    "label, text", TITLE_FORMS, ids=[row[0] for row in TITLE_FORMS]
)
def test_a_destination_carrying_a_title_does_not_hide_the_name(label, text):
    """The title sits inside the destination; the label is still the name."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", SPACED_REFERENCE_FORMS, ids=[row[0] for row in SPACED_REFERENCE_FORMS]
)
def test_a_reference_spelled_as_a_phrase_does_not_hide_the_name(label, text):
    """A reference's own words are layout, the label's words are the name."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", RUN_FORMS, ids=[row[0] for row in RUN_FORMS]
)
def test_the_forms_the_run_reader_owns_survive_a_spaced_mark(label, text):
    """The space inside the mark is what made the initial and plural miss."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", CONTROLS, ids=[row[0] for row in CONTROLS]
)
@pytest.mark.parametrize("entry", [FULL, INVERTED], ids=["full", "inverted"])
def test_nothing_is_assembled_out_of_a_spaced_mark(label, text, entry):
    """A mark spells no name, and two bracket groups are not one name."""
    assert _hits(text, entry) == [], label


# -- the piece the rule is made of --------------------------------------------
def _mark_end(text: str, token_upto: int) -> tuple[int, int] | None:
    """The mark when the token ``text[:token_upto]`` opens one."""
    return scan_secrets._link_mark_at(text, 0, token_upto)


MARK_ENDS = [
    ("bare destination", f"{LAST}]({URL})", (len(LAST), len(f"{LAST}]({URL})"))),
    (
        "destination with a title",
        f'{LAST}]({URL} "note")',
        (len(LAST), len(f'{LAST}]({URL} "note")')),
    ),
    ("spaced reference", f"{LAST}][see note]", (len(LAST), len(f"{LAST}][see note]"))),
    ("unbalanced destination", f'{LAST}]({URL} "note"', None),
    ("unbalanced reference", f"{LAST}][see note", None),
    ("mark spanning a line break", f'{LAST}]({URL}\n"note")', None),
    ("no mark at all", LAST, None),
]

# A token cut at the mark's first space is what the tokeniser really hands over.
TOKEN_MARKS = [
    ("destination with a title", f'{LAST}]({URL} "note")', len(f"{LAST}]({URL}")),
    ("spaced reference", f"{LAST}][see note]", len(f"{LAST}][see")),
    (
        "reference with a bracketed note",
        f"{LAST}][see [1] note]",
        len(f"{LAST}][see"),
    ),
    ("destination with parentheses in it", f'{LAST}]({URL}_(x) "note")', len(f"{LAST}]({URL}_(x)")),
]


@pytest.mark.parametrize(
    "label, text, expected", MARK_ENDS, ids=[row[0] for row in MARK_ENDS]
)
def test_a_mark_is_read_through_only_when_its_pair_closes(label, text, expected):
    """A whole mark on one line is read through; an unbalanced one is not."""
    assert _mark_end(text, len(text)) == expected, label


@pytest.mark.parametrize(
    "label, text, token_upto", TOKEN_MARKS, ids=[row[0] for row in TOKEN_MARKS]
)
def test_the_token_the_tokeniser_hands_over_is_read_through(label, text, token_upto):
    """The space inside the mark is where the tokeniser cuts the token."""
    assert _mark_end(text, token_upto) == (len(LAST), len(text)), label


def test_a_token_that_opens_no_mark_is_left_alone():
    """The boundary a line wrote is kept when no mark is on it."""
    assert _mark_end(f"{LAST} crew", len(LAST)) is None


RUNS = [
    (
        "the initial keeps its dot before the mark",
        f'[Q. {LAST}]({URL} "note")',
        [["q.", LAST.casefold()]],
    ),
    (
        "the label's words are one run",
        f'[{LAST}][see note] {FIRST}',
        [[LAST.casefold(), FIRST.casefold()]],
    ),
    (
        "a full stop after the link ends the run",
        f'[Q. {LAST}]({URL} "note"). Bell Corvis',
        [["q.", LAST.casefold()], ["bell", "corvis"]],
    ),
    (
        "a possessive after the link stays on the word",
        f"[Q. {LAST}][see note]'s crew",
        [["q.", LAST.casefold()]],
    ),
]


@pytest.mark.parametrize(
    "label, text, expected", RUNS, ids=[row[0] for row in RUNS]
)
def test_the_run_reads_through_a_spaced_mark(label, text, expected):
    """The run keeps the label's words and ends where the line ends them."""
    assert _runs(text) == expected, label


def test_a_mark_that_spans_a_line_break_is_not_joined():
    """Two lines are two lines; the mark is not read across the break."""
    text = f'[Q. {LAST}]({URL}\n"note") crew'
    assert [run for run in _runs(text) if run[:1] == ["q."]] == []


def test_the_phrase_matcher_keeps_its_own_reading():
    """The literal phrase is still matched wherever the line writes it."""
    assert _hits(f"[x]({URL}/{LAST})", LAST)
