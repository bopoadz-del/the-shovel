"""An initial whose dot is followed by the document's own closing mark.

The run reader reads ``Q.`` as an initial because of its dot: the dot is the
document's statement that the letter stands for a name. Measured on ``main``,
that dot was accepted only when nothing at all followed it. ``Vantrey, Q.``
at the end of a line read; the same entry inside an aside, a bold span, a
quotation, a bracket, or followed by the comma or semicolon that ends a
listing entry read 0 -- 3 of 10 forms read in total. The letter lost its dot
to the punctuation written after it and became a bare one-letter word, which
matches nothing.

The repair reads the dot as the letter's whenever it is written right after
the letter; what follows that dot is the document's punctuation. What is
*not* widened: a bare capital letter is still an ordinary one-letter word, a
second dot (an ellipsis written onto the letter) is not an initial's dot, and
the word an initial abbreviates still comes only from the denylist entry.

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
    "scan_secrets_name_initial_closing_mark", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_initial_closing_mark"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quil" + "lon"
    last = "Vant" + "rey"
    other = "Cor" + "vis"
    return first, last, other


FIRST, LAST, OTHER = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


# -- the closing marks a document writes after an initial ---------------------
CLOSED_FORMS = [
    ("an aside's closing bracket", f"approved (by {LAST}, Q.) today"),
    ("a bold span's closing mark", f"**{LAST}, Q.** approved"),
    ("an emphasis span's closing mark", f"_{LAST}, Q._ approved"),
    ("a listing entry's semicolon", f"{LAST}, Q.; {OTHER}, Z."),
    ("a listing entry's comma", f"{LAST}, Q., and the crew"),
    ("a square bracket", f"see [{LAST}, Q.] above"),
    ("a straight quote", f'signed "{LAST}, Q." on the sheet'),
    ("a curly quote", f"signed “{LAST}, Q.” on the sheet"),
    ("a colon", f"{LAST}, Q.: approved"),
]


@pytest.mark.parametrize("entry", [FULL, INVERTED])
@pytest.mark.parametrize("label,text", CLOSED_FORMS)
def test_an_initial_keeps_its_dot_behind_a_closing_mark(label, text, entry):
    assert _hits(text, entry), label


# -- the forms that read before still read ------------------------------------
UNCHANGED_FORMS = [
    ("surname then initial at the end of a line", f"{LAST}, Q."),
    ("surname then initial mid-sentence", f"signed {LAST}, Q. today"),
    ("initial then surname in an aside", f"(Q. {LAST})"),
    ("written out in full in an aside", f"({FULL})"),
]


@pytest.mark.parametrize("entry", [FULL, INVERTED])
@pytest.mark.parametrize("label,text", UNCHANGED_FORMS)
def test_the_forms_that_read_before_still_read(label, text, entry):
    assert _hits(text, entry), label


# -- what is not widened -------------------------------------------------------
NOT_READ = [
    ("another person behind the same mark", f"({OTHER}, Q.)"),
    ("another initial behind the same mark", f"({LAST}, Z.)"),
    ("the surname alone behind a mark", f"({LAST}.)"),
    ("a bare letter with no dot, then a mark", f"({LAST}, Q)"),
    ("an ellipsis written onto the letter", f"({LAST}, Q...)"),
    ("a doubled dot written onto the letter", f"({LAST}, Q..)"),
]


@pytest.mark.parametrize("entry", [FULL, INVERTED])
@pytest.mark.parametrize("label,text", NOT_READ)
def test_nothing_else_is_read_as_an_initial(label, text, entry):
    assert not _hits(text, entry), label


# -- the predicate itself -------------------------------------------------------
@pytest.mark.parametrize(
    "shape,core,expected",
    [
        ("Q.", "Q", True),
        ("Q.)", "Q", True),
        ("(Q.)", "Q", True),
        ("Z.**", "Z", True),
        ("Q.;", "Q", True),
        ("Q", "Q", False),
        ("Q)", "Q", False),
        ("Q...", "Q", False),
        ("Z..)", "Z", False),
        ("QR.)", "QR", False),
        ("q.)", "q", False),
    ],
)
def test_the_dot_is_the_letters_only_right_after_it(shape, core, expected):
    assert scan_secrets._closes_with_initial(shape, core) is expected


@pytest.mark.parametrize(
    "text,expected_line",
    [
        (f"intro\n\n(by {LAST}, Q.)\n", 3),
        (f"one\ntwo\nthree\n**{LAST}, Q.**\n", 4),
    ],
)
def test_a_hit_is_reported_at_the_line_the_initial_is_on(text, expected_line):
    hits = _hits(text)
    assert hits
    assert {line for line, _ in hits} == {expected_line}
