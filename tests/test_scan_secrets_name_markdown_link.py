"""A written name set as the label of a markdown link.

A link writes its label in brackets and the place it points at in the
parentheses that follow: ``[Q. Surname](x)``. To a reader the label is the
words and the destination is where they lead; to the tokeniser the label's
last word and the destination are welded into one token, because no space
falls between ``]`` and ``(``. A surname that arrives as ``Surname](x)`` is
name-shaped in no case at all, so the run reader ended at the name and lost
every spelling it exists to read -- an initial, a reversed entry, a plural.

The destination is layout, exactly as the table bar is: it is taken off before
the word is judged, and its opening parenthesis is found by balancing, so a
URL that itself carries parentheses is still one destination. The label's
closing bracket goes with it -- together they are the link's closing mark, and
leaving the bracket behind would hide an initial's own dot. What follows the
destination is kept, so a comma or full stop after the link still ends the
entry.

What is *not* widened: nothing is assembled out of a destination (a name that
appears only in the URL stays quiet), an unbalanced mark is punctuation and
not a link, and an unrelated label beside a name in the target is not a name.
The words are invented and assembled at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_markdown_link", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_markdown_link"] = scan_secrets
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


# -- the spellings a link writes a name in -----------------------------------
LINK_FORMS = [
    ("label, given first, url destination", f"[{FULL}]({URL})"),
    ("label, given first, bare destination", f"[{FULL}](x)"),
    ("label, reversed", f"[{INVERTED}]({URL})"),
    ("label, given name as an initial", f"[Q. {LAST}](x)"),
    ("label, reversed with an initial", f"[{LAST}, Q.](x)"),
    ("label, all capitals", f"[{FULL.upper()}](x)"),
    ("label with a parenthesised destination", f"[Q. {LAST}](https://e.invalid/a_(b))"),
    ("label in a sentence", f"the crew of [Q. {LAST}](x) mobilised"),
    ("image alt label", f"![Q. {LAST}](x.png)"),
    ("label in a table cell", f"| [Q. {LAST}](x) | note |"),
    ("given name linked, surname after it", f"[{FIRST}](x) {LAST}"),
    ("link closed by a sentence comma", f"see [Q. {LAST}](x), then"),
    ("link closed by a full stop", f"[Q. {LAST}](x)."),
]

# The forms whose last name word is welded to the destination need the run
# reader; the literal phrase covers the rest. Both entry orders are read.
RUN_FORMS = [
    ("initial", f"[Q. {LAST}]({URL})"),
    ("reversed with an initial", f"[{LAST}, Q.](x)"),
    ("plural after an initial", f"[Q. {LAST}s](x)"),
    ("plural, reversed", f"[{LAST}s, {FIRST}](x)"),
]

CONTROLS = [
    ("name only in the destination", f"[note](https://example.invalid/{LAST}-{FIRST})"),
    ("one-word label", f"[{LAST}](x)"),
    ("unrelated label and destination", "[charter](https://example.invalid/a)"),
    ("longer surname in the label", f"[Q. {LAST}field](x)"),
    ("another person in the label", "[Q. Ashdown](x)"),
    ("unbalanced mark is not a link", f"[Q. {LAST}](x"),
    ("name in the destination, unrelated label", f"[Rate Card]({LAST})"),
]


@pytest.mark.parametrize(
    "label, text", LINK_FORMS, ids=[row[0] for row in LINK_FORMS]
)
def test_a_name_set_as_a_link_label_is_read(label, text):
    """The destination is layout; the words of the label are the name."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", LINK_FORMS, ids=[row[0] for row in LINK_FORMS]
)
def test_the_inverted_entry_reads_the_label_too(label, text):
    """The entry's word order cannot decide whether a destination is seen."""
    assert _hits(text, INVERTED), label


@pytest.mark.parametrize(
    "label, text", RUN_FORMS, ids=[row[0] for row in RUN_FORMS]
)
def test_the_forms_the_run_reader_owns_survive_the_destination(label, text):
    """The welded token is what made the initial, plural and reversal miss."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", CONTROLS, ids=[row[0] for row in CONTROLS]
)
@pytest.mark.parametrize("entry", [FULL, INVERTED], ids=["full", "inverted"])
def test_nothing_is_assembled_out_of_a_destination(label, text, entry):
    """A destination spells no name, and one word is not half a name."""
    assert _hits(text, entry) == [], label


# -- the piece the rule is made of -------------------------------------------
DESTINATION_SHAPES = [
    ("bare destination", f"{LAST}](x)", LAST),
    ("url destination", f"{LAST}]({URL})", LAST),
    ("initial before the mark", "Q.](x)", "Q."),
    ("parenthesised destination", "crew](a_(b))", "crew"),
    ("punctuation after the link", "crew](x),", "crew,"),
    ("unbalanced parentheses", f"{LAST}](x", f"{LAST}](x"),
    ("no link at all", LAST, LAST),
]


@pytest.mark.parametrize(
    "label, token, expected",
    DESTINATION_SHAPES,
    ids=[row[0] for row in DESTINATION_SHAPES],
)
def test_only_a_balanced_destination_is_taken_off(label, token, expected):
    """The link's closing mark is ``](...)``; anything else is left alone."""
    assert scan_secrets._without_link_destination(token) == expected, label


RUNS = [
    ("the initial keeps its dot before the destination", f"[Q. {LAST}](x)", [["q.", LAST.casefold()]]),
    ("the label's words are one run", f"[{LAST}](x) {FIRST}", [[LAST.casefold(), FIRST.casefold()]]),
    (
        "a full stop after the link ends the run",
        f"[Q. {LAST}](x). Bell Corvis",
        [["q.", LAST.casefold()], ["bell", "corvis"]],
    ),
]


@pytest.mark.parametrize(
    "label, text, expected", RUNS, ids=[row[0] for row in RUNS]
)
def test_the_run_reads_through_the_label(label, text, expected):
    """The run keeps the label's words and ends where the line ends them."""
    runs = [words for _, words in scan_secrets.name_runs(text)]
    assert runs == expected, label
