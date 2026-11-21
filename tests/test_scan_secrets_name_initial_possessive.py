"""A written initial that carries the possessive: ``Surname, Q.'s crew``.

An entry copied out of an index, a byline or a table cell prints the given
name as its initial, and a sentence attributes what follows to the person it
has just named. Both happen in the same line: ``Surname, Q.'s crew``. The
clitic was only taken off a word of letters, and ``Q.`` is a letter and a dot,
so the token stayed as it was written, was not a name word, and ended the run
with the surname alone in it. Measured on the branch base, 8 of 8 such
spellings missed against both entry orders, and 8 of 8 read now.

The dot is the reader's evidence that the letter stands for a name, so it
stays with the letter: the word read off the token is the letter alone and its
shape keeps the dot, which is exactly the pair the tokeniser produces for
``Q.`` written with no clitic at all. Nothing is widened past that -- a
two-letter word with a dot, a number with a dot, an initial with no dot, and
another person's initial all stay quiet.

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
    "scan_secrets_name_initial_possessive", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_initial_possessive"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str, str, str]:
    """The invented identities, assembled so no tracked line spells them."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    other_first = "Bell" + "amy"
    s_final_last = "Cor" + "vis"
    return first, last, other_first, s_final_last


FIRST, LAST, OTHER_FIRST, S_FINAL_LAST = _words()
INITIAL = FIRST[0]
S = "'"
CURLY = "\u2019"
DASH = "\u2013"
NBSP = "\u00a0"
EDGE = "\"'`.,;:!?()[]{}<>*_/\\|~#@$%^&+="


def _hits(text: str, entry: str) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


def _orders(first: str, last: str) -> tuple[str, str]:
    """The entry as a list types it and as a document prints it."""
    return f"{first} {last}", f"{last}, {first}"


# -- the spellings a document writes -----------------------------------------
INITIAL_POSSESSIVES = [
    ("straight mark", f"{LAST}, {INITIAL}.{S}s crew", _orders(FIRST, LAST)),
    ("curly mark", f"{LAST}, {INITIAL}.{CURLY}s crew", _orders(FIRST, LAST)),
    ("capital s after the mark", f"{LAST}, {INITIAL}.{S}S crew", _orders(FIRST, LAST)),
    ("name in capitals", f"{LAST.upper()}, {INITIAL}.{S}S CREW", _orders(FIRST, LAST)),
    ("table cell", f"| {LAST}, {INITIAL}.{S}s crew | 3 |", _orders(FIRST, LAST)),
    ("nothing written after the clitic", f"signed by {LAST}, {INITIAL}.{S}s", _orders(FIRST, LAST)),
    ("no-break space before the initial", f"{LAST},{NBSP}{INITIAL}.{S}s crew", _orders(FIRST, LAST)),
    ("en dash after the noun", f"{LAST}, {INITIAL}.{S}s {DASH} crew", _orders(FIRST, LAST)),
    (
        "s-final surname, initial possessive",
        f"{S_FINAL_LAST}, {OTHER_FIRST[0]}.{S}s crew",
        _orders(OTHER_FIRST, S_FINAL_LAST),
    ),
]


@pytest.mark.parametrize(("label", "text", "entries"), INITIAL_POSSESSIVES)
def test_an_initial_possessive_reads_the_entry(
    label: str, text: str, entries: tuple[str, str]
) -> None:
    reads = [entry for entry in entries if _hits(text, entry)]
    assert len(reads) == 2, label


# -- what the reader must not widen to ---------------------------------------
QUIET = [
    ("another person's initial", f"{LAST}, Z.{S}s crew"),
    ("another surname", f"{S_FINAL_LAST}, {INITIAL}.{S}s crew"),
    ("a two-letter word with a dot", f"{LAST}, {INITIAL}o.{S}s crew"),
    ("a number with a dot", f"{LAST}, 2.{S}s crew"),
    ("an initial with no dot", f"{LAST}, {INITIAL}{S}s crew"),
    ("the surname alone", f"{LAST}{S}s crew mobilised"),
]


@pytest.mark.parametrize(("label", "text"), QUIET)
@pytest.mark.parametrize("order", [0, 1])
def test_the_widening_stays_quiet(label: str, text: str, order: int) -> None:
    assert not _hits(text, _orders(FIRST, LAST)[order]), label


# -- the rule underneath, read on its own ------------------------------------
@pytest.mark.parametrize(
    ("label", "shape", "expected"),
    [
        ("a straight mark", f"{INITIAL}.{S}s", (INITIAL, f"{INITIAL}.")),
        ("a curly mark", f"{INITIAL}.{CURLY}s", (INITIAL, f"{INITIAL}.")),
        ("capitals after the clitic", f"{INITIAL}.{S}S", (INITIAL, f"{INITIAL}.")),
        ("a two-letter word with a dot", f"{INITIAL}o.{S}s", (f"{INITIAL}o.{S}s",) * 2),
        ("a number with a dot", f"2.{S}s", (f"2.{S}s",) * 2),
    ],
)
def test_a_written_initial_takes_the_clitic_off(label: str, shape: str, expected) -> None:
    """The initial's dot stays in the shape; a longer word is not an initial."""
    core = shape.strip(EDGE)
    assert scan_secrets._without_possessive(shape, core) == expected, label


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"{LAST}, {INITIAL}.{S}s crew was briefed", [[LAST.lower(), f"{INITIAL.lower()}."]]),
        (f"{LAST}, {INITIAL}.{CURLY}s crew was briefed", [[LAST.lower(), f"{INITIAL.lower()}."]]),
        (f"{LAST}, {INITIAL}. crew was briefed", [[LAST.lower(), f"{INITIAL.lower()}."]]),
    ],
)
def test_the_run_keeps_the_initial_and_its_dot(text: str, expected: list) -> None:
    runs = [run for _, run in scan_secrets.name_runs(text) if len(run) >= 2]
    assert runs == expected
