"""A label welded to the name it introduces by a joiner.

A label and the name after it are often written with no space between them --
``by:Q. Surname``, ``and/Surname, Given``, ``owner=Surname, Q.``. The tokeniser
splits only on the separators a name is written with, so it read the label, the
joiner and the first name word as one token. That token is name-shaped in no
case at all, and every spelling but the literal one was lost: the initial, the
surname-first listing, the initial after a surname.

In prose the token is now cut right after a joiner written between a plain word
and a capital, and the two words are read as they would be had a space followed
the joiner. The joiner stays on the left-hand word as a closing mark, so that
word ends its run and no two words are joined across a joiner.

What is *not* widened: source files are read exactly as before; a left-hand side
that is not letters from the token's start (a path, a link destination, an
address) is cut nowhere; a joiner followed by a lower-case letter is not cut;
another person behind a joiner is still not the entry. The words are invented
and assembled at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_joiner_weld", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_joiner_weld"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str, str]:
    """The invented identities, assembled so no tracked line ever spells them."""
    first = "Quil" + "lon"
    last = "Vant" + "rey"
    other = "Cor" + "vis"
    return first, last, other


FIRST, LAST, OTHER = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"
INITIAL = f"{FIRST[0]}. {LAST}"
INITIAL_AFTER = f"{LAST}, {FIRST[0]}."
JOINERS = [":", "/", "=", "|", "+", "&"]


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


def _runs(text: str, prose: bool = True) -> list[list[str]]:
    return [run for _, run in scan_secrets.name_runs(text, prose=prose)]


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize("written", [INITIAL, INVERTED, INITIAL_AFTER])
def test_a_name_welded_to_its_label_is_read(joiner: str, written: str) -> None:
    """The joiner ends the label; the name after it is read in every spelling."""
    assert _hits(f"signed by{joiner}{written} today\n") == [(1, 0)], (joiner, written)


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize("label", ["by", "Owner"])
def test_the_word_after_the_joiner_opens_its_own_run(joiner: str, label: str) -> None:
    """The welded token reads as the two words a space would have made."""
    welded = _runs(f"x {label}{joiner}{INITIAL} y\n")
    spaced = _runs(f"x {label}{joiner} {INITIAL} y\n")
    assert welded == spaced, (joiner, label)
    assert [FIRST[0].lower() + ".", LAST.lower()] in [
        [w.lower() for w in run] for run in welded
    ], (joiner, label)


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize("left, right", [(FIRST, LAST), (LAST, FIRST)])
def test_two_words_are_never_joined_across_a_joiner(
    joiner: str, left: str, right: str
) -> None:
    """The joiner is a closing mark on the left word: it ends that word's run."""
    text = f"see {left}{joiner}{right} here\n"
    assert _runs(text) == [], (joiner, left)
    assert _hits(text) == [], (joiner, left)
    assert _hits(text, INVERTED) == [], (joiner, left)


@pytest.mark.parametrize(
    "token",
    [
        f"https://example.invalid/{FIRST[0]}.",
        f"(https://example.invalid/{LAST})",
        f"9to:{FIRST[0]}.",
        f"./dir/{FIRST[0]}.",
        f"a.b:{LAST},",
    ],
)
def test_a_left_side_that_is_not_a_plain_word_is_cut_nowhere(token: str) -> None:
    """A path, a link destination or an identifier keeps its token whole."""
    assert scan_secrets._welded_word_end(token, 0, len(token)) == len(token), token


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize("text", ["by{j}some word", "ratio{j}value", "a{j}b"])
def test_a_joiner_before_a_lower_case_letter_is_not_cut(joiner: str, text: str) -> None:
    """Only a capital after the joiner can be a name word, so nothing else moves."""
    written = text.format(j=joiner)
    end = written.find(" ") if " " in written else len(written)
    assert scan_secrets._welded_word_end(written, 0, end) == end, written


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize(
    "written",
    [f"{FIRST[0]}. {OTHER}", f"{OTHER}, {FIRST}", f"Z. {LAST}", f"{LAST} alone"],
)
def test_another_person_behind_a_joiner_is_still_not_the_entry(
    joiner: str, written: str
) -> None:
    """Reading through the joiner adds reading, never a new match."""
    assert _hits(f"by{joiner}{written} today\n") == [], (joiner, written)


@pytest.mark.parametrize("joiner", [":", "=", "|"])
@pytest.mark.parametrize("written", [INITIAL, INVERTED])
def test_source_files_are_read_exactly_as_before(joiner: str, written: str) -> None:
    """The cut is prose layout; a source line is read as written."""
    code = f"by{joiner}{written}\n"
    assert _runs(code, prose=False) == [], (joiner, written)
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: FULL})
    assert scan_secrets.scan_client_document(
        code, patterns, allow_line_join=False
    ) == [], (joiner, written)
    assert scan_secrets.scan_client_document(code, patterns) == [(1, 0)], (joiner, written)
