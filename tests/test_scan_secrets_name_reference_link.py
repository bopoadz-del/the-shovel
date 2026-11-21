"""A written name set as the label of a markdown reference link.

A reference link writes its label in brackets and names the reference to follow
in a second bracket pair: ``[Q. Surname][ref]``, or ``[Q. Surname][]`` when the
reference is the label itself. To a reader the label is the words and the second
pair is where they are defined; to the tokeniser the label's last word and the
reference are welded into one token, because no space falls between ``]`` and
``[``. A surname that arrives as ``Surname][ref]`` is name-shaped in no case at
all, so the run reader ended at the name and lost every spelling it exists to
read -- an initial, a reversed entry, a plural.

The reference is layout, exactly as the parenthesised destination is: it is
taken off before the word is judged, and its brackets are found by balancing,
so a reference that carries a bracketed note is still one reference. The
label's closing bracket goes with it -- together they are the link's closing
mark, and leaving the bracket behind would hide an initial's own dot. What
follows the reference is kept, so a comma or full stop after the link still
ends the entry.

What is *not* widened: nothing is assembled out of a reference (a name that
appears only in the reference stays quiet), a name split across two bracket
groups is not one name, an unbalanced mark is punctuation and not a link, and a
reference whose label carries a space is still cut at the space by the run
reader. The words are invented and assembled at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_reference_link", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_reference_link"] = scan_secrets
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


# -- the spellings a reference link writes a name in --------------------------
LINK_FORMS = [
    ("label, given first", f"[{FULL}][ref]"),
    ("label, reversed", f"[{INVERTED}][ref]"),
    ("label, given name as an initial", f"[Q. {LAST}][ref]"),
    ("label, reversed with an initial", f"[{LAST}, Q.][ref]"),
    ("label, all capitals", f"[{FULL.upper()}][ref]"),
    ("collapsed reference", f"[Q. {LAST}][]"),
    ("label with a bracketed reference", f"[Q. {LAST}][ref[1]]"),
    ("label in a sentence", f"the crew of [Q. {LAST}][ref] mobilised"),
    ("image alt label", f"![Q. {LAST}][ref]"),
    ("label in a table cell", f"| [Q. {LAST}][ref] | note |"),
    ("link closed by a sentence comma", f"see [Q. {LAST}][ref], then"),
    ("link closed by a full stop", f"[Q. {LAST}][ref]."),
]

# The forms whose last name word is welded to the reference need the run
# reader; the literal phrase covers the rest. Both entry orders are read.
RUN_FORMS = [
    ("initial", f"[Q. {LAST}][ref]"),
    ("reversed with an initial", f"[{LAST}, Q.][ref]"),
    ("plural after an initial", f"[Q. {LAST}s][ref]"),
    ("plural, reversed", f"[{LAST}s, {FIRST}][ref]"),
]

CONTROLS = [
    ("name only in the reference", f"[note][{LAST}]"),
    ("one-word label", f"[{LAST}][ref]"),
    ("unrelated label and reference", "[charter][ref]"),
    ("longer surname in the label", f"[Q. {LAST}field][ref]"),
    ("another person in the label", "[Q. Ashdown][ref]"),
    ("unbalanced mark is not a link", f"[Q. {LAST}][ref"),
    ("ordinary double index", f"array[{LAST}][0]"),
    ("name split across two bracket groups", f"[{FIRST}][{LAST}]"),
]


@pytest.mark.parametrize(
    "label, text", LINK_FORMS, ids=[row[0] for row in LINK_FORMS]
)
def test_a_name_set_as_a_reference_label_is_read(label, text):
    """The reference is layout; the words of the label are the name."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", LINK_FORMS, ids=[row[0] for row in LINK_FORMS]
)
def test_the_inverted_entry_reads_the_label_too(label, text):
    """The entry's word order cannot decide whether a reference is seen."""
    assert _hits(text, INVERTED), label


@pytest.mark.parametrize(
    "label, text", RUN_FORMS, ids=[row[0] for row in RUN_FORMS]
)
def test_the_forms_the_run_reader_owns_survive_the_reference(label, text):
    """The welded token is what made the initial, plural and reversal miss."""
    assert _hits(text), label


@pytest.mark.parametrize(
    "label, text", CONTROLS, ids=[row[0] for row in CONTROLS]
)
@pytest.mark.parametrize("entry", [FULL, INVERTED], ids=["full", "inverted"])
def test_nothing_is_assembled_out_of_a_reference(label, text, entry):
    """A reference spells no name, and two bracket groups are not one name."""
    assert _hits(text, entry) == [], label


# -- the piece the rule is made of -------------------------------------------
REFERENCE_SHAPES = [
    ("bare reference", f"{LAST}][ref]", LAST),
    ("collapsed reference", f"{LAST}][]", LAST),
    ("reference with a bracketed note", f"{LAST}][ref[1]]", LAST),
    ("initial before the mark", "Q.][ref]", "Q."),
    ("punctuation after the reference", "crew][ref],", "crew,"),
    ("unbalanced reference", f"{LAST}][ref", f"{LAST}][ref"),
    ("no reference at all", LAST, LAST),
]


@pytest.mark.parametrize(
    "label, token, expected",
    REFERENCE_SHAPES,
    ids=[row[0] for row in REFERENCE_SHAPES],
)
def test_only_a_balanced_reference_is_taken_off(label, token, expected):
    """The link's closing mark is ``][...]``; anything else is left alone."""
    assert scan_secrets._without_link_reference(token) == expected, label


RUNS = [
    (
        "the initial keeps its dot before the reference",
        f"[Q. {LAST}][ref]",
        [["q.", LAST.casefold()]],
    ),
    (
        "the label's words are one run",
        f"[{LAST}][ref] {FIRST}",
        [[LAST.casefold(), FIRST.casefold()]],
    ),
    (
        "a full stop after the link ends the run",
        f"[Q. {LAST}][ref]. Bell Corvis",
        [["q.", LAST.casefold()], ["bell", "corvis"]],
    ),
]


@pytest.mark.parametrize(
    "label, text, expected", RUNS, ids=[row[0] for row in RUNS]
)
def test_the_run_reads_through_the_reference(label, text, expected):
    """The run keeps the label's words and ends where the line ends them."""
    runs = [words for _, words in scan_secrets.name_runs(text)]
    assert runs == expected, label
