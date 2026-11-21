"""The inverted listing form: a name written ``Surname, Given``.

A byline, an index entry, a table cell and a client list all write a name with
the surname first and a comma between it and the given name. The run rule read
every closing mark as the end of an entry, so the comma ended the run and the
two halves of the name were never seen together: measured on the branch head
that carries the middle-name rule, five spellings of the inverted form scored
0 of 5, while the two controls on the same lines scored.

The repair is that a comma between two name words separates entries instead of
ending one, and that only a mark written *after* a word can end the entry the
word is in -- an opening bracket or quote written before one closes nothing.
Nothing else moves: a full stop, a question mark, a colon and a semicolon still
close the run, and the existing sentence-break tests still hold that line.

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
    "scan_secrets_inverted", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_inverted"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"
MIDDLE_INITIAL = "A."


def _patterns(*extra: str) -> list:
    return scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: "\n".join((FULL, *extra))}
    )


def _hits(text: str, *extra: str) -> list[tuple[int, int]]:
    return scan_secrets.scan_client_document(text, _patterns(*extra))


# -- the inverted listing form -------------------------------------------------


INVERTED = [
    ("surname then given, after a comma", f"{LAST}, {FIRST}"),
    ("the same set in capitals", f"{LAST.upper()}, {FIRST.upper()}"),
    ("with a middle initial", f"{LAST}, {FIRST} {MIDDLE_INITIAL}"),
    ("in a table row", f"| {LAST}, {FIRST} | engineer |"),
    ("as an index entry", f"{LAST}, {FIRST} ... 12"),
    ("between two other names", f"Card, Rate; {LAST}, {FIRST}"),
]


@pytest.mark.parametrize(
    "label, spelled", INVERTED, ids=[label for label, _ in INVERTED]
)
def test_a_name_written_surname_first_after_a_comma_is_caught(label, spelled):
    """A comma separates entries; it does not end the name on either side."""
    text = f"Reviewed by {spelled}\n"
    assert _hits(text) == [(1, 0)], label


COMMA_IN_SCRIPT = [
    ("latin", ","),
    ("arabic", "\u060c"),
    ("ideographic", "\u3001"),
    ("fullwidth", "\uff0c"),
]


@pytest.mark.parametrize(
    "label, comma", COMMA_IN_SCRIPT, ids=[label for label, _ in COMMA_IN_SCRIPT]
)
def test_the_comma_is_read_as_each_script_writes_it(label, comma):
    """The same mark in another script is the same separator."""
    text = f"Reviewed by {LAST}{comma} {FIRST}\n"
    assert _hits(text) == [(1, 0)], label


# -- what the comma must not do ------------------------------------------------


NOT_A_LISTING = [
    ("a lower-case word after the comma", f"{LAST}, the engineer signed"),
    ("prose after the comma", f"{LAST}, who signed it, agreed"),
    ("a full stop before the given name", f"{LAST}. {FIRST} signed"),
    ("a semicolon before the given name", f"{LAST}; {FIRST} signed"),
    ("a colon before the given name", f"{LAST}: {FIRST} signed"),
    ("a question mark before the given name", f"{LAST}? {FIRST} signed"),
]


@pytest.mark.parametrize(
    "label, text", NOT_A_LISTING, ids=[label for label, _ in NOT_A_LISTING]
)
def test_a_break_that_is_not_a_comma_still_ends_the_run(label, text):
    """One mark was widened; the other closing marks keep their behaviour."""
    assert _hits(f"{text}\n") == [], label


PROSE_WITH_A_COMMA = [
    ("a common noun and a place", "the quilt, the vantage point"),
    ("two ordinary listed nouns", "the site, the crew"),
]


@pytest.mark.parametrize(
    "label, text",
    PROSE_WITH_A_COMMA,
    ids=[label for label, _ in PROSE_WITH_A_COMMA],
)
def test_a_comma_between_prose_words_starts_nothing(label, text):
    """Merging runs across a comma must not manufacture a hit out of prose."""
    assert _hits(f"{text}\n") == [], label


# -- wrapping punctuation is written before the word, not after it -------------


WRAPPED = [
    ("an opening bracket", f"({LAST} {FIRST})"),
    ("an opening quote", f'"{LAST} {FIRST}'),
    ("an opening bracket round the inverted form", f"({LAST}, {FIRST})"),
]


@pytest.mark.parametrize("label, spelled", WRAPPED, ids=[row[0] for row in WRAPPED])
def test_punctuation_written_before_a_word_does_not_end_the_run(label, spelled):
    """A bracket or a quote opens the word; nothing is being closed by it.

    This is the other half of reading the mark that matters: only what is
    written AFTER a word can end the entry the word is in.
    """
    assert _hits(f"Reviewed by {spelled}\n") == [(1, 0)], label


# -- the predicate the run builder reads --------------------------------------


KEEPS_RUN = [
    ("nothing after the word", "Vantrey", "Vantrey", True),
    ("a comma", "Vantrey,", "Vantrey", True),
    ("two commas", "Vantrey,,", "Vantrey", True),
    ("an arabic comma", "Vantrey\u060c", "Vantrey", True),
    ("the dot of an initial", "A.", "A", True),
    ("a full stop after a surname", "Vantrey.", "Vantrey", False),
    ("a semicolon", "Vantrey;", "Vantrey", False),
    ("a closing bracket", "Vantrey)", "Vantrey", False),
    ("a quote", '"Vantrey"', "Vantrey", False),
    ("a question mark", "Vantrey?", "Vantrey", False),
]


@pytest.mark.parametrize(
    "label, shape, core, expected",
    KEEPS_RUN,
    ids=[row[0] for row in KEEPS_RUN],
)
def test_only_a_comma_and_an_initial_keep_the_run_open(label, shape, core, expected):
    """The rule is one predicate, so it is read here on its own terms."""
    assert scan_secrets._keeps_run(shape, core) is expected, label


def test_a_comma_alone_carries_no_word_and_ends_nothing():
    """A token of punctuation between two spaces is a separator, not a word."""
    runs = scan_secrets.name_runs(f"{LAST} , {FIRST} signed")
    assert runs and runs[0][1] == [LAST.lower(), FIRST.lower()]


@pytest.mark.parametrize(
    "label, text, expected",
    [
        (
            "the form after a title line",
            f"site file\n{LAST}, {FIRST}  Engineer  Site",
            2,
        ),
        ("the form deeper in the page", f"one\ntwo\n{LAST}, {FIRST}", 3),
    ],
)
def test_the_inverted_form_is_reported_at_the_line_it_starts_on(label, text, expected):
    """A location, not the text: the first line of the match is the address."""
    assert _hits(f"{text}\n") == [(expected, 0)], label
