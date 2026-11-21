"""A family named in the plural is the same name: ``Q. Vantreys' crew``.

A document attributes work to a household, not to one person, and English
writes that household with the plural of the surname: ``the Surnames' crew``,
``the Surnames arrived``. The run reader compared a written word against a
denylisted word exactly, so the plural was a different word in every spelling
the literal phrase match cannot reach -- a reversed index entry, and a byline
that abbreviates the given name. Measured on the branch base, 6 of 6 such
spellings missed; 6 of 6 read now.

What is *not* widened: only the two regular English plurals are folded --
``-s``, and ``-es`` after a word that already ends in ``s``. A longer word, an
irregular plural, ``-es`` on a word that does not end in ``s``, and a
lower-case plural all stay misses, and the rest of the name still has to be
read beside the plural word. A surname and its given name split by a
lower-case word remain a miss, because the run ends at that word.

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
    "scan_secrets_name_plural", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_plural"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str, str, str]:
    """The invented identities, assembled so no tracked line spells them."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    other_first = "Bell" + "amy"
    other_last = "Cor" + "vis"
    return first, last, other_first, other_last


FIRST, LAST, OTHER_FIRST, OTHER_LAST = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"
OTHER = f"{OTHER_FIRST} {OTHER_LAST}"
OTHER_INVERTED = f"{OTHER_LAST}, {OTHER_FIRST}"


def _hits(text: str, entry: str) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


# -- the spellings a plural name is written in -------------------------------
PLURAL_FORMS = [
    ("a reversed index entry", f"{LAST}s, {FIRST}", FULL),
    ("an initial and a plural possessive", f"Q. {LAST}s' crew mobilised", FULL),
    (
        "the curly mark on the plural possessive",
        f"Q. {LAST}s\u2019 crew mobilised",
        FULL,
    ),
    (
        "the inverted entry, the initial and the plural",
        f"Q. {LAST}s' crew mobilised",
        INVERTED,
    ),
    (
        "-es after a surname already ending in s",
        f"B. {OTHER_LAST}es' crew mobilised",
        OTHER,
    ),
    (
        "the inverted entry and the -es plural",
        f"B. {OTHER_LAST}es' crew mobilised",
        OTHER_INVERTED,
    ),
]

CONTROLS = [
    ("a longer word is not a plural", f"{LAST}field, {FIRST}"),
    ("a longer word after an initial", f"Q. {LAST}field crew mobilised"),
    ("an irregular plural", f"{LAST}ses' crew, {FIRST}"),
    ("an irregular plural after an initial", f"Q. {LAST}ses crew mobilised"),
    ("-es only follows a word ending in s", f"Q. {LAST}es crew mobilised"),
    ("a lower-case plural is not a written name", f"{LAST.lower()}s' crew, {FIRST}"),
    ("a plural of another name", f"Q. Ashdowns crew mobilised"),
    ("a possessive of another name", f"Q. Rate's crew mobilised"),
]


@pytest.mark.parametrize(
    "label, text, entry", PLURAL_FORMS, ids=[row[0] for row in PLURAL_FORMS]
)
def test_a_name_written_in_the_plural_is_read(label, text, entry):
    """The household is the name the denylist entry names in the singular."""
    assert _hits(text, entry), label


@pytest.mark.parametrize(
    "label, text", [(row[0], row[1]) for row in CONTROLS]
)
@pytest.mark.parametrize("entry", [FULL, INVERTED], ids=["full", "inverted"])
def test_only_the_regular_plural_is_the_same_word(label, text, entry):
    """A longer word, an irregular plural and a lower-case word stay misses."""
    assert _hits(text, entry) == [], label


# -- the piece the rule is made of -------------------------------------------
WORD_CASES = [
    ("the word itself", LAST, LAST, True),
    ("the -s plural", f"{LAST}s", LAST, True),
    ("the -es plural of a word in s", f"{OTHER_LAST}es", OTHER_LAST, True),
    ("a longer word", f"{LAST}field", LAST, False),
    ("an irregular plural", f"{LAST}ses", LAST, False),
    ("-es on a word not ending in s", f"{LAST}es", LAST, False),
    ("a different word", "Ashdowns", LAST, False),
]


@pytest.mark.parametrize(
    "label, written, word, expected",
    [(label, w.lower(), word.lower(), expected) for label, w, word, expected in WORD_CASES],
    ids=[row[0] for row in WORD_CASES],
)
def test_only_the_regular_plurals_are_folded(label, written, word, expected):
    """The comparison folds two spellings and nothing else."""
    assert scan_secrets._word_reads_as(written, word) is expected, label


RUN_CASES = [
    ("an initial and a plural possessive", f"Q. {LAST}s' crew", ["q.", f"{LAST}s".lower()]),
    ("a reversed index entry", f"{LAST}s, {FIRST}", [f"{LAST}s".lower(), FIRST.lower()]),
]


@pytest.mark.parametrize(
    "label, text, written", RUN_CASES, ids=[row[0] for row in RUN_CASES]
)
def test_the_run_carries_the_plural_word(label, text, written):
    """The plural is one word in the run, not a different token."""
    assert [run for _, run in scan_secrets.name_runs(text)] == [written], label
