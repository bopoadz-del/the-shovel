"""A possessive clitic on a written name: ``Q. Vantrey's crew``.

A document attributes what follows to the person it has just named, and the
possessive is how it does it: ``Vantrey's crew``, ``the project's Vantrey's
sheet``. The mark and its ``s`` belong to the sentence, not to the name, but
the run reader judged the token whole -- ``Vantrey's`` was name-shaped in no
case at all, so the run ended at the name. Measured on the branch head that
carries the initial rule, the possessive form of the name scored 0 of 3
order-independent spellings with an inverted entry, and the initial form with
a possessive scored 0 of 2; both score in full now.

What is *not* widened: the plural is a different word and stays a miss, a
possessive on a name not on the list stays a miss, and a possessive of a
number or an identifier is not a written name. The clitic is taken off the
token and the remaining word still has to be read as the denylisted word --
no vocabulary enters the scanner.

The words are invented and assembled at runtime, so the repository contains no
name and no fixture the real scan would have to chase.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_possessive", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_possessive"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"
STRAIGHT = "'"
CURLY = "\u2019"


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


# -- the forms a document writes the possessive in ---------------------------
POSSESSIVE_FORMS = [
    ("possessive on the surname", f"{FULL}{STRAIGHT}s crew mobilised"),
    ("possessive, curly mark", f"{FULL}{CURLY}s crew mobilised"),
    ("possessive, given name as an initial", f"Q. {LAST}{STRAIGHT}s crew mobilised"),
    ("possessive, initial, curly mark", f"Q. {LAST}{CURLY}s crew mobilised"),
    ("possessive in a table cell", f"| {FULL}{STRAIGHT}s crew | 12 |"),
    ("possessive with a full stop after it", f"Signed by {FULL}{STRAIGHT}s crew."),
    ("possessive in a name written in capitals", f"{FULL.upper()}{STRAIGHT}S CREW"),
    ("possessive in a run of prose", f"The mobilisation is {FULL}{STRAIGHT}s to run"),
]

CONTROLS = [
    ("the plural is a different word", f"Q. {LAST}s crew mobilised"),
    ("a plural possessive is a different word", f"{LAST}s{STRAIGHT} crew mobilised"),
    ("a possessive of another name", f"Q. Rate{STRAIGHT}s crew mobilised"),
    ("a possessive of a lower-case word", f"the site{STRAIGHT}s crew mobilised"),
    ("a possessive of a number", f"the 2010{STRAIGHT}s crew mobilised"),
    ("the given name alone, possessive", f"{FIRST}{STRAIGHT}s crew mobilised"),
    ("a different word ending in the mark", f"{LAST}ald{STRAIGHT}s crew mobilised"),
    ("the initial form with a plural", f"| Q. {LAST}s | 12 |"),
]


@pytest.mark.parametrize(
    "label, text, entry",
    [(label, text, FULL) for label, text in POSSESSIVE_FORMS],
    ids=[row[0] for row in POSSESSIVE_FORMS],
)
def test_a_possessive_names_the_person_it_is_attached_to(label, text, entry):
    """The clitic is the sentence's; the word underneath it is the name."""
    assert _hits(text, entry), label


@pytest.mark.parametrize(
    "label, text, entry",
    [(label, text, INVERTED) for label, text in POSSESSIVE_FORMS],
    ids=[row[0] for row in POSSESSIVE_FORMS],
)
def test_the_possessive_is_read_for_an_inverted_entry_too(label, text, entry):
    """The entry's word order cannot decide whether the clitic is seen."""
    assert _hits(text, entry), label


@pytest.mark.parametrize(
    "label, text, entry",
    [(label, text, entry) for label, text in CONTROLS for entry in (FULL, INVERTED)],
    ids=[f"{row[0]}-{index}" for index, row in enumerate(CONTROLS) for _ in (0, 1)],
)
def test_nothing_else_loses_its_last_letters_to_the_rule(label, text, entry):
    """Only a clitic comes off: a plural, another name, a number do not."""
    assert _hits(text, entry) == [], label


@pytest.mark.parametrize(
    "label, entry, text",
    [
        ("the name written out still matches", FULL, f"{FULL} signed"),
        ("the surname-first form still matches", FULL, f"{LAST}, {FIRST}"),
        ("the initial form still matches", FULL, f"Q. {LAST} signed"),
        ("an unrelated name still misses", "Bramwell Ashdown", f"{FULL} signed"),
    ],
)
def test_the_forms_that_matched_before_still_match(label, entry, text):
    """The clitic rule adds a spelling; it does not replace one."""
    expected = label != "an unrelated name still misses"
    assert bool(_hits(text, entry)) is expected, label


# -- the piece the rule is made of -------------------------------------------
WITHOUT_POSSESSIVE = [
    ("a straight mark", f"{LAST}{STRAIGHT}s", f"{LAST}{STRAIGHT}s", (LAST, LAST)),
    ("a curly mark", f"{LAST}{CURLY}s", f"{LAST}{CURLY}s", (LAST, LAST)),
    (
        "marks after the clitic are kept",
        f"{LAST}{STRAIGHT}s.",
        f"{LAST}{STRAIGHT}s",
        (LAST, f"{LAST}."),
    ),
    ("capitals after the clitic", f"{LAST}{STRAIGHT}S", f"{LAST}{STRAIGHT}S", (LAST, LAST)),
    (
        "a plural is not a possessive",
        f"{LAST}s{STRAIGHT}",
        f"{LAST}s{STRAIGHT}",
        (f"{LAST}s{STRAIGHT}", f"{LAST}s{STRAIGHT}"),
    ),
    (
        "a number keeps its mark",
        f"2010{STRAIGHT}s",
        f"2010{STRAIGHT}s",
        (f"2010{STRAIGHT}s", f"2010{STRAIGHT}s"),
    ),
    ("an apostrophe with no s", f"{LAST}{STRAIGHT}", f"{LAST}{STRAIGHT}", (f"{LAST}{STRAIGHT}", f"{LAST}{STRAIGHT}")),
    ("a word with no mark", LAST, LAST, (LAST, LAST)),
]


@pytest.mark.parametrize(
    "label, shape, core, expected",
    WITHOUT_POSSESSIVE,
    ids=[row[0] for row in WITHOUT_POSSESSIVE],
)
def test_only_a_clitic_is_taken_off_the_word(label, shape, core, expected):
    """The rule removes one clitic and leaves every other mark in place."""
    assert scan_secrets._without_possessive(shape, core) == expected, label


@pytest.mark.parametrize(
    "label, text, written",
    [
        ("after the name written out", f"{FULL}{STRAIGHT}s crew", ["quillon", "vantrey"]),
        ("after an initial", f"Q. {LAST}{CURLY}s crew", ["q.", "vantrey"]),
    ],
)
def test_the_run_carries_the_word_the_clitic_was_attached_to(label, text, written):
    """The run sees the name, not the token the sentence built around it."""
    runs = scan_secrets.name_runs(text)
    assert [run for _, run in runs] == [written], label
