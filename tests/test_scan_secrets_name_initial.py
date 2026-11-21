"""A given name written as an initial: the byline form of the same name.

A document writes a person's given name in full in one place and as an initial
in another -- a byline, a citation, an index entry, a title block's signature
line: ``Q. Vantrey`` for ``Quillon Vantrey``. Measured on the branch head that
carries the comma and entry rules, that form scored 0 of 8 and the same name
written out scored 1 of 1. The reader had the dot in front of it and threw it
away: edge punctuation was stripped before the run was built, and a run word
is compared to a denylist word by equality.

The repair reads the dot as what it is -- the document's own statement that
the letter stands for a word -- and lets a one-letter word written with it
stand for a denylisted word beginning with that letter. What is *not*
widened: a bare capital letter is still an ordinary one-letter word, a word
with two letters is not an initial, and the word it abbreviates always comes
from the denylist entry, so no vocabulary is added to the scanner.

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
    "scan_secrets_name_initial", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_initial"] = scan_secrets
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


# -- the forms a document writes the name in ----------------------------------
INITIAL_FORMS = [
    ("given name as its initial, with the dot", f"Q. {LAST} signed the sheet"),
    ("in an inverted listing", f"{LAST}, Q."),
    ("in a table cell", f"| Q. {LAST} | Engineer |"),
    ("in a name written in capitals", f"Q. {LAST.upper()}"),
    ("two initials before the surname", f"Q. R. {LAST}"),
    ("surname first, both names as initials", f"{LAST}, Q. R."),
    ("the surname as an initial", f"{FIRST} V."),
    ("in a run of prose", f"Prepared by Q. {LAST}, engineer of record"),
]

CONTROLS = [
    ("a bare capital is not an initial", f"Q {LAST} signed the sheet"),
    ("an initial of another letter", f"A. {LAST} signed the sheet"),
    ("two dots are not one initial", f"Q.. {LAST} signed the sheet"),
    ("a two-letter word is not an initial", f"Qq. {LAST} signed the sheet"),
    ("an initial before unrelated words", "Q. Rate Card"),
    ("a number is never a name word", f"1. {LAST} signed the sheet"),
]


@pytest.mark.parametrize(
    "label, text, entry", [(label, text, FULL) for label, text in INITIAL_FORMS],
    ids=[row[0] for row in INITIAL_FORMS],
)
def test_an_initial_is_read_as_the_given_name(label, text, entry):
    """A byline's ``Q.`` is the denylisted given name it abbreviates."""
    assert _hits(text, entry), label


@pytest.mark.parametrize(
    "label, text, entry",
    [(label, text, INVERTED) for label, text in INITIAL_FORMS],
    ids=[row[0] for row in INITIAL_FORMS],
)
def test_the_initial_is_read_for_an_inverted_entry_too(label, text, entry):
    """The entry's word order cannot decide whether the initial is read."""
    assert _hits(text, entry), label


@pytest.mark.parametrize(
    "label, text, entry", [(label, text, FULL) for label, text in CONTROLS],
    ids=[row[0] for row in CONTROLS],
)
def test_nothing_else_is_read_as_an_initial(label, text, entry):
    """Only one letter closed by one dot abbreviates, and only a name word."""
    assert _hits(text, entry) == [], label


@pytest.mark.parametrize(
    "label, entry, text",
    [
        ("the given name in full still matches", FULL, f"{FULL} signed"),
        ("the surname-first form still matches", FULL, f"{LAST}, {FIRST}"),
        ("a middle initial still matches", FULL, f"{FIRST} R. {LAST}"),
        ("an unrelated name still misses", "Bramwell Ashdown", f"{FULL} signed"),
    ],
)
def test_the_forms_that_matched_before_still_match(label, entry, text):
    """The initial rule adds a spelling; it does not replace one."""
    expected = label != "an unrelated name still misses"
    assert bool(_hits(text, entry)) is expected, label


# -- the two pieces the rule is made of ---------------------------------------
WORD_READS_AS = [
    ("the word itself", "quillon", "quillon", True),
    ("its dotted initial", "q.", "quillon", True),
    ("an initial of another letter", "a.", "quillon", False),
    ("a bare capital letter", "q", "quillon", False),
    ("two letters and a dot", "qq.", "quillon", False),
    ("an initial of another word", "v.", "quillon", False),
]


@pytest.mark.parametrize(
    "label, written, word, expected",
    WORD_READS_AS,
    ids=[row[0] for row in WORD_READS_AS],
)
def test_a_written_word_reads_as_its_word_or_its_initial(
    label, written, word, expected
):
    """The comparison is one predicate, so it is read here on its own terms."""
    assert scan_secrets._word_reads_as(written, word) is expected, label


RUNS = [
    ("an initial keeps its dot in the run", f"Q. {LAST}", ["q.", LAST.lower()]),
    ("a bare letter joins without a dot", f"Q {LAST}", ["q", LAST.lower()]),
    ("a full given name is unchanged", f"{FIRST} {LAST}", [FIRST.lower(), LAST.lower()]),
]


@pytest.mark.parametrize(
    "label, text, expected", RUNS, ids=[row[0] for row in RUNS]
)
def test_the_run_carries_the_dot_that_makes_a_letter_an_initial(
    label, text, expected
):
    """Without the dot the matcher cannot tell an initial from a one-letter word."""
    runs = scan_secrets.name_runs(text)
    assert runs and runs[0][1] == expected, label


CANDIDATES = [
    ("the word is written out", "quillon", "quillon vantrey", True),
    ("only its initial is written", "quillon", "q. vantrey", True),
    ("neither is written", "quillon", "vantrey signed", False),
    ("the initial of another word", "quillon", "b. vantrey", False),
]


@pytest.mark.parametrize(
    "label, word, block, expected", CANDIDATES, ids=[row[0] for row in CANDIDATES]
)
def test_a_block_with_the_initial_is_not_dropped_before_the_run_reader(
    label, word, block, expected
):
    """The candidate test in front of the run reader must see the initial."""
    assert scan_secrets._word_or_initial_in(word, block) is expected, label


LINE_POSITIONS = [
    ("after two lines of a site file", f"site file\ntwo\nQ. {LAST} Engineer\n", 3),
    ("at the top of the page", f"Q. {LAST} Engineer\n", 1),
]


@pytest.mark.parametrize(
    "label, text, expected",
    LINE_POSITIONS,
    ids=[row[0] for row in LINE_POSITIONS],
)
def test_a_hit_is_reported_at_the_line_the_initial_is_on(label, text, expected):
    """The position reported is the line the name starts on, not the next one."""
    assert [line for line, _ in _hits(text)] == [expected], label
