"""The denylist entry: how the secret was typed cannot switch the rule off.

The order-independent rule reads an *entry* of the client-pattern denylist and
sequencing its words, so the shape of the entry decides whether a name is
protected in any form but its literal spelling. An entry is typed by a person
from the place the name was found -- a byline, an index entry, a licence line,
a table cell -- and those places write ``Surname, Given`` with the comma, a
closing dot or a quotation mark around it. Measured on the branch head that
carries the inverted-listing rule, an entry written that way scored 0 of 2
order-independent document forms while the plain entry scored 2 of 2: the
protection was silently off for the entry itself.

The repair is that an entry is read the way the document reader reads a
written word -- the same edge punctuation stripped, the entry comma treated as
the separator beside it -- so the set of words is the same however the secret
was typed. What is *not* widened: an entry still has to be two to four
Titlecase words, so a lower-case phrase, a regex and a caps phrase stay
phrases, and the folded literal match is unchanged.

The words are invented and assembled at runtime, so the repository contains no
name and no fixture the real scan would have to chase.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_entry", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_entry"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: entry}
    )
    return scan_secrets.scan_client_document(text, patterns)


# -- how an entry's words are read --------------------------------------------

ENTRY_FORMS = [
    ("surname then given, after a comma", f"{LAST}, {FIRST}"),
    ("no space after the comma", f"{LAST},{FIRST}"),
    ("the Arabic comma", f"{LAST}\u060c {FIRST}"),
    ("the ideographic comma", f"{LAST}\u3001 {FIRST}"),
    ("the fullwidth comma", f"{LAST}\uff0c {FIRST}"),
    ("a closing dot", f"{LAST}, {FIRST}."),
    ("quoted", f'"{LAST}, {FIRST}"'),
    ("a different separator", f"{LAST}\t{FIRST}"),
]


@pytest.mark.parametrize("label,entry", ENTRY_FORMS)
def test_an_entry_written_in_any_listing_form_reads_as_its_words(label, entry):
    pattern = re.compile(entry, re.IGNORECASE)
    assert scan_secrets.name_pattern_words(pattern) == (
        LAST.lower(),
        FIRST.lower(),
    ), label


DOC_FORMS = [
    ("given then surname", f"note: {FIRST} {LAST} signed"),
    ("given then surname in capitals", f"note: {FIRST.upper()} {LAST.upper()}"),
    ("surname then given", f"by {LAST}, {FIRST}"),
    ("in a table row", f"| {FIRST} {LAST} | engineer |"),
    ("as an index entry", f"{LAST}, {FIRST} ... 12"),
]


@pytest.mark.parametrize("label,text", DOC_FORMS)
def test_an_inverted_entry_catches_every_written_form(label, text):
    assert _hits(text, entry=f"{LAST}, {FIRST}"), label


@pytest.mark.parametrize("label,entry", ENTRY_FORMS)
def test_an_entry_in_each_listing_form_catches_the_plain_document(label, entry):
    assert _hits(f"note: {FIRST} {LAST} signed", entry=entry), label
    assert _hits(f"note: {FIRST.upper()} {LAST.upper()}", entry=entry), label


@pytest.mark.parametrize("label,text", DOC_FORMS)
def test_the_plain_entry_keeps_catching_every_written_form(label, text):
    assert _hits(text, entry=FULL), label


# -- what stays a phrase, stated rather than discovered later -----------------

NOT_A_WORD_SET = [
    ("a lower-case phrase", "site plan"),
    ("a regex entry", f"{LAST}|{FIRST}"),
    ("one word", LAST),
    ("a joiner between the words", f"{LAST} & {FIRST}"),
    ("capitalised prose", "CLIENT NAME"),
]


@pytest.mark.parametrize("label,entry", NOT_A_WORD_SET)
def test_an_entry_that_does_not_spell_a_name_has_no_word_set(label, entry):
    assert scan_secrets.name_pattern_words(re.compile(entry, re.IGNORECASE)) is None, label


@pytest.mark.parametrize(
    "label,entry",
    [
        ("caps surname, given", f"{LAST.upper()}, {FIRST.upper()}"),
        ("caps given surname", f"{FIRST.upper()} {LAST.upper()}"),
    ],
)
def test_an_entry_spelled_in_capitals_stays_a_phrase(label, entry):
    """The Titlecase contract is unchanged: only caps *documents* are read."""
    assert scan_secrets.name_pattern_words(re.compile(entry, re.IGNORECASE)) is None, label


# -- no hit is manufactured --------------------------------------------------

CONTROLS = [
    ("unrelated text", "a line about something else entirely"),
    ("prose with a conjunction", f"note: {FIRST} and {LAST} signed"),
    ("the given name alone", f"note: {FIRST} signed the sheet"),
    ("the surname alone", f"note: {LAST} signed the sheet"),
]


@pytest.mark.parametrize("label,text", CONTROLS)
def test_the_inverted_entry_adds_no_hit_to_other_lines(label, text):
    assert _hits(text, entry=f"{LAST}, {FIRST}") == [], label


@pytest.mark.parametrize(
    "label,page",
    [
        ("name on the third line", ["header", "body", f"note: {FIRST} {LAST}"]),
        ("name on the fifth line", ["a", "b", "c", "d", f"note: {LAST}, {FIRST}"]),
    ],
)
def test_a_hit_is_reported_on_the_line_that_carries_it(label, page):
    hits = _hits("\n".join(page), entry=f"{LAST}, {FIRST}")
    assert [line for line, _ in hits] == [len(page)], label
