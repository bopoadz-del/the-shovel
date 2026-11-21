"""A written name carries its script and its middle names as presentation.

Two gaps are closed here, and both were measured open on the default branch
before either was built:

* a denylist entry spelled in a non-Latin script got no sequence match at all.
  ``name_pattern_words`` asked an ASCII regular expression whether a word was
  a name word, so a Cyrillic, Greek or Armenian entry -- written with a
  capital at every word exactly as a Latin one is -- was never a name to the
  rule, and only the one spelling the list happened to be typed in was caught;
* a name written out in full carries more than one word between its first and
  last: ``Given Alaric B. Surname`` has a middle name and an initial at once.
  The rule allowed exactly one inserted word, so the fullest -- and most
  formal -- spelling of a name was the one it walked past.

The repairs are the shape of the word read from the characters themselves
rather than from an ASCII range, and a bounded interior gap: up to two
inserted name words, with the first and last word of the window still exact,
so an insertion at either end still does not count and a sentence's worth of
capitalised text between the words still cannot satisfy it.

The words are invented here and assembled at runtime, so the repository
contains no name and no fixture the real scan would have to chase.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_width", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_width"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"

# Two middle words, invented like the name itself and used only to stand in
# the gap a full written name has. Deliberately not name-shaped by accident:
# any capitalised word would do, and these are chosen to be unmistakable.
MIDDLE = ["Ambrose", "Bramwell"]
MIDDLE_INITIALS = ["A.", "B."]


def _patterns(*extra: str) -> list:
    return scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: "\n".join((FULL, *extra))}
    )


def _hits(text: str, *extra: str) -> list[tuple[int, int]]:
    return scan_secrets.scan_client_document(text, _patterns(*extra))


# -- the word shape is a property of the alphabet -----------------------------


@pytest.mark.parametrize(
    "word, expected",
    [
        ("Vantrey", True),
        ("\u0412\u0430\u043d\u0442\u0440\u0435\u0439", True),  # Cyrillic Titlecase
        ("\u0391\u03bb\u03ad\u03be\u03b1\u03bd\u03b4\u03c1\u03bf\u03c2", True),
        ("VANTREY", False),
        ("Vantrey2", False),
        ("Abc1", False),
        ("A", False),
        ("\u4e09\u90ce", False),  # a script with no case has no name word
        ("", False),
    ],
)
def test_a_titlecase_word_is_read_in_any_script(word, expected):
    assert scan_secrets._is_titlecase_word(word) is expected, word


@pytest.mark.parametrize(
    "word, expected",
    [
        ("VANTREY", True),
        ("\u0412\u0410\u041d\u0422\u0420\u0415\u0419", True),
        ("A", True),
        ("Vantrey", False),
        ("AB1", False),
        ("", False),
    ],
)
def test_a_capitalised_word_is_read_in_any_script(word, expected):
    assert scan_secrets._is_capitalised_word(word) is expected, word


@pytest.mark.parametrize(
    "word, expected",
    [
        ("A", True),
        ("\u0410", True),
        ("AB", False),
        ("a", False),
        ("", False),
    ],
)
def test_an_initial_is_one_capital_letter(word, expected):
    assert scan_secrets._is_initial(word) is expected, word


# -- a name written out in full ------------------------------------------------


def _spellings() -> list[tuple[str, str]]:
    """Written forms of one name with two interior words, derived at runtime."""
    return [
        ("two initials", f"{FIRST} {MIDDLE_INITIALS[0]} {MIDDLE_INITIALS[1]} {LAST}"),
        ("two middle names", f"{FIRST} {MIDDLE[0]} {MIDDLE[1]} {LAST}"),
        (
            "surname first, two initials",
            f"{LAST} {MIDDLE_INITIALS[0]} {MIDDLE_INITIALS[1]} {FIRST}",
        ),
        ("surname first, two middle names", f"{LAST} {MIDDLE[0]} {MIDDLE[1]} {FIRST}"),
        (
            "surname first, one middle name and one initial",
            f"{LAST} {MIDDLE[0]} {MIDDLE_INITIALS[1]} {FIRST}",
        ),
    ]


SPELLINGS = _spellings()


@pytest.mark.parametrize(
    "label, spelled", SPELLINGS, ids=[label for label, _ in SPELLINGS]
)
def test_a_name_written_out_with_two_middle_words_is_caught(label, spelled):
    text = f"Filed by {spelled} on site\n"
    assert _hits(text) == [(1, 0)], label


@pytest.mark.parametrize(
    "label, spelled", SPELLINGS, ids=[label for label, _ in SPELLINGS]
)
def test_the_same_full_name_set_in_capitals_is_caught(label, spelled):
    """A title block sets the full name in capitals: the gap is two, not one."""
    text = f"FILE {spelled.upper()} ON SITE\n"
    assert _hits(text) == [(1, 0)], label


@pytest.mark.parametrize(
    "label, text",
    [
        (
            "three middle names",
            f"{FIRST} {MIDDLE[0]} {MIDDLE[1]} Cedric {LAST} signed\n",
        ),
        (
            "surname first, three middle names",
            f"{LAST} {MIDDLE[0]} {MIDDLE[1]} Cedric {FIRST} signed\n",
        ),
        (
            "three initials",
            f"{FIRST} {MIDDLE_INITIALS[0]} {MIDDLE_INITIALS[1]} C. {LAST} signed\n",
        ),
    ],
)
def test_the_interior_gap_is_bounded(label, text):
    """Three interior words are a list of people or a heading, not one name.

    The bound is what keeps the rule an identity match: without it, the first
    and last words of any long capitalised run would be a name, and a heading
    that happens to start and end with the right words would report one.
    """
    assert _hits(text) == [], label


# -- a denylist entry in a non-Latin script ------------------------------------

_CYRILLIC = {
    "Q": "\u041a",
    "u": "\u0443",
    "i": "\u0438",
    "l": "\u043b",
    "o": "\u043e",
    "n": "\u043d",
    "V": "\u0412",
    "a": "\u0430",
    "t": "\u0442",
    "r": "\u0440",
    "e": "\u0435",
    "y": "\u0439",
}
_CYRILLIC_INITIAL = "\u0410"  # the Cyrillic capital A, a middle initial


def _transliterate(word: str) -> str:
    """The invented word written in Cyrillic, letter for letter, at runtime."""
    return "".join(_CYRILLIC[letter] for letter in word)


def _cyrillic_full() -> str:
    return f"{_transliterate(FIRST)} {_transliterate(LAST)}"


CYRILLIC_FULL = _cyrillic_full()


@pytest.mark.parametrize(
    "label, text",
    [
        ("as listed", CYRILLIC_FULL),
        ("surname first", f"{_transliterate(LAST)} {_transliterate(FIRST)}"),
        (
            "with a middle initial",
            f"{_transliterate(FIRST)} {_CYRILLIC_INITIAL}. {_transliterate(LAST)}",
        ),
        (
            "surname first, with a middle initial",
            f"{_transliterate(LAST)} {_CYRILLIC_INITIAL}. {_transliterate(FIRST)}",
        ),
        ("set in capitals", CYRILLIC_FULL.upper()),
    ],
)
def test_a_name_listed_in_another_script_is_caught_however_it_is_written(label, text):
    """Cyrillic, Greek and Armenian spell a name with capitals, as Latin does.

    An entry in such a script is a name entry, and the written forms are the
    same ones: surname first, a middle initial, a title block in capitals.
    Reading the word shape from the characters rather than from the ASCII
    range is what makes those forms reachable at all.
    """
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: CYRILLIC_FULL})
    assert scan_secrets.scan_client_document(
        f"\u0424\u0430\u0439\u043b: {text}\n", patterns
    ) == [(1, 0)], label


@pytest.mark.parametrize(
    "text",
    [
        f"{_transliterate(FIRST[:-1] + 'a')} {_transliterate(LAST)} signed\n",
        f"{_transliterate(LAST)} {_transliterate(FIRST[:-1] + 'a')} signed\n",
    ],
)
def test_another_script_does_not_turn_near_misses_into_hits(text):
    """The same exact rule in another script: a different word is different."""
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: CYRILLIC_FULL})
    assert scan_secrets.scan_client_document(text, patterns) == [], text


@pytest.mark.parametrize(
    "text",
    [
        f"{FIRST} and {MIDDLE[0]} and {MIDDLE[1]} and {LAST} signed\n",
        f"{LAST} and {MIDDLE[0]} and {MIDDLE[1]} and {FIRST} signed\n",
    ],
)
def test_lower_case_words_between_the_name_words_still_break_the_run(text):
    """The interior tolerance reads capitalised words only, never prose."""
    assert _hits(text) == [], text


# -- the limit this change does not close --------------------------------------


def _caseless_words() -> list[tuple[str, str, str]]:
    """Invented two-word names in scripts that do not mark case, from code points."""
    han = ("".join(map(chr, (0x4E09, 0x90CE))), "".join(map(chr, (0x5C71, 0x7530))))
    kana = ("".join(map(chr, (0x304B, 0x308F))), "".join(map(chr, (0x307F, 0x306A))))
    return [("Han", han[0], han[1]), ("Kana", kana[0], kana[1])]


CASELESS = _caseless_words()


@pytest.mark.parametrize(
    "label, first, last", CASELESS, ids=[label for label, _, _ in CASELESS]
)
def test_a_script_without_case_is_matched_literally_and_the_limit_is_recorded(
    label, first, last
):
    """A caseless script has no capital to read as a name word.

    Han and Kana do not mark case, so the sentence-order rule -- which is what
    catches a surname-first attribution -- has nothing to read there. The
    literal comparison still catches the name as listed, across separators and
    presentation, and that is asserted here rather than left for a reader to
    assume: the gap is real, and it is a different mechanism (a script-aware
    word splitter), not this rule widened.
    """
    entry = f"{first} {last}"
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    assert scan_secrets.scan_client_document(f"{entry}\n", patterns) == [(1, 0)], label
    other = f"{last} {first}"
    assert scan_secrets.scan_client_document(f"{other}\n", patterns) == [], label
