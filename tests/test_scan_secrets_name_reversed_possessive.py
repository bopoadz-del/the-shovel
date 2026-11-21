"""A surname-first attribution with a possessive: ``Vantrey's crew, Quillon``.

A document attributes work to a person and then prints the entry the way an
index, a byline and a table cell print it -- surname first, the given name
after it. The possessive sits on the surname, and the noun it attributes
(``crew``, ``sheet``, ``report``) stands between the two name words. The run
reader closed the run at that noun, so the surname was left alone in its run
and the given name in the next one: measured on the branch base, 7 of 7 such
spellings missed against both entry orders, and 14 of 14 read now.

The evidence is the clitic itself, never a vocabulary: a word written with a
possessive mark attributes what follows to the name just written, so the run
passes over exactly one lower-case word and resumes at the next name word.
Only one word is passed over, the noun has to keep the entry open -- a comma
or no mark at all -- and a sentence break still ends the entry.

What is *not* widened: a longer word, another person's possessive, two
lower-case words between the name words, a full stop between them, a
possessive of a number, and a surname written with no given name beside it all
stay quiet. A possessive on a written initial is a different rule, read by
``test_scan_secrets_name_initial_possessive.py``.

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
    "scan_secrets_name_reversed_possessive", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_reversed_possessive"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str, str, str]:
    """The invented identities, assembled so no tracked line spells them."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    other_first = "Bell" + "amy"
    s_final_last = "Cor" + "vis"
    return first, last, other_first, s_final_last


FIRST, LAST, OTHER_FIRST, S_FINAL_LAST = _words()
S = "'"
CURLY = "\u2019"
DASH = "\u2013"


def _hits(text: str, entry: str) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


def _orders(first: str, last: str) -> tuple[str, str]:
    """The entry as a list types it and as a document prints it."""
    return f"{first} {last}", f"{last}, {first}"


# -- the surname-first attributions a document writes ------------------------
REVERSED_POSSESSIVES = [
    ("comma attribution", f"{LAST}{S}s crew, {FIRST}", _orders(FIRST, LAST)),
    ("comma attribution, curly mark", f"{LAST}{CURLY}s crew, {FIRST}", _orders(FIRST, LAST)),
    ("comma attribution, en dash", f"{LAST}{S}s crew {DASH} {FIRST}", _orders(FIRST, LAST)),
    ("initial after the noun", f"{LAST}{S}s crew, Q.", _orders(FIRST, LAST)),
    ("plural possessive, comma", f"{LAST}s{S} crew, {FIRST}", _orders(FIRST, LAST)),
    ("plural possessive, initial", f"{LAST}s{S} crew, Q.", _orders(FIRST, LAST)),
    (
        "s-final surname, possessive",
        f"{S_FINAL_LAST}{S}s crew, {OTHER_FIRST}",
        _orders(OTHER_FIRST, S_FINAL_LAST),
    ),
    (
        "s-final surname, plural possessive",
        f"{S_FINAL_LAST}es{S} crew, {OTHER_FIRST}",
        _orders(OTHER_FIRST, S_FINAL_LAST),
    ),
]


@pytest.mark.parametrize(("label", "text", "entries"), REVERSED_POSSESSIVES)
def test_a_reversed_possessive_reads_the_entry(
    label: str, text: str, entries: tuple[str, str]
) -> None:
    reads = [entry for entry in entries if _hits(text, entry)]
    assert len(reads) == 2, label


# -- what the reader must not widen to ---------------------------------------
QUIET = [
    ("a longer word", f"{LAST}field{S}s crew, {FIRST}"),
    ("another person's possessive", f"{S_FINAL_LAST}{S}s crew, {FIRST}"),
    ("two lower-case words between", f"{LAST}{S}s whole crew, {FIRST}"),
    ("a sentence break between", f"{LAST}{S}s crew. {FIRST}"),
    ("no given name beside it", f"{LAST}{S}s crew mobilised"),
    ("a possessive of a number", f"2010{S}s crew, {FIRST}"),
]


@pytest.mark.parametrize(("label", "text"), QUIET)
@pytest.mark.parametrize("order", [0, 1])
def test_the_widening_stays_quiet(label: str, text: str, order: int) -> None:
    assert not _hits(text, _orders(FIRST, LAST)[order]), label


# -- the rule underneath, read on its own ------------------------------------
@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (f"{LAST}{S}s", True),
        (f"{LAST}s{S}", True),
        (f"{LAST}{S}s,", True),
        (f"{LAST}.", False),
        (f"{LAST})", False),
    ],
)
def test_a_possessive_mark_keeps_the_entry_open(shape: str, expected: bool) -> None:
    core = shape.strip("\"'`.,;:!?()[]{}<>*_/\\|~#@$%^&+=")
    assert scan_secrets._keeps_run(shape, core) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"{LAST}{S}s crew, {FIRST} signed", [[LAST.lower(), FIRST.lower()]]),
        (f"{LAST}s{S} crew, {FIRST} signed", [[LAST.lower() + "s", FIRST.lower()]]),
        (f"{LAST}{S}s crew. {FIRST} signed", []),
    ],
)
def test_the_run_resumes_after_the_attributed_noun(text: str, expected: list) -> None:
    runs = [run for _, run in scan_secrets.name_runs(text) if len(run) >= 2]
    assert runs == expected
