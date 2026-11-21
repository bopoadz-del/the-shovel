"""A task-list box or an html tag welded to the name it carries.

Two marks of prose layout are written with no space before the word they carry:
a task-list item's checked box (``- [x]Surname, Given``) and an html tag's
delimiters (``<b>Surname, Given</b>``, ``<a href="u">Q. Surname</a>``). The
tokeniser reads the mark and the first name word as one token, the word is
name-shaped in no case at all, and every spelling except the literal one is
lost -- the surname-first listing, the initial, the initial after a surname.

Each mark keeps its opening character and the rest is blanked to spaces before
any token is built. Every offset stays where the document wrote it; the kept
``[`` or ``<`` still ends the run before the mark, so two list items or two
table cells are never read as one name; and the word after the mark is no
longer welded to it.

What is *not* widened: ``[x]`` that does not open a list item is an index or a
regular expression's class and is left as written; a tag written across a line
break is left alone; an attribute's text stays where it was written; source
files are read exactly as before. The words are invented and assembled at
runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_markup_mark", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_markup_mark"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str, str]:
    """The invented identities, assembled so no tracked line ever spells them."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    other = "Cor" + "vis"
    return first, last, other


FIRST, LAST, OTHER = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"
INITIAL = f"{FIRST[0]}. {LAST}"
INITIAL_AFTER = f"{LAST}, {FIRST[0]}."


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


def _runs(text: str) -> list[list[str]]:
    return [run for _, run in scan_secrets.name_runs(text)]


@pytest.mark.parametrize(
    "box, written",
    [
        ("- [x]", INVERTED),
        ("* [X]", INITIAL),
        ("+ [x]", INITIAL_AFTER),
        ("1. [x]", INVERTED),
        ("  - [X]", INITIAL),
        ("3) [x]", INITIAL_AFTER),
    ],
)
def test_a_checked_box_welded_to_the_name_keeps_it_readable(
    box: str, written: str
) -> None:
    """The box is the item's state, never a letter of the item's text."""
    assert _hits(f"{box}{written} to sign\n") == [(1, 0)], (box, written)


@pytest.mark.parametrize(
    "tag, close, written",
    [
        ("<b>", "</b>", INVERTED),
        ('<a href="https://example.invalid/x">', "</a>", INITIAL),
        ('<span class="n">', "</span>", INVERTED),
        ("<td>", "</td>", INITIAL),
        ("<strong>", "</strong>", INVERTED),
    ],
)
def test_a_tag_welded_to_the_name_keeps_it_readable(
    tag: str, close: str, written: str
) -> None:
    """``<b`` and ``>`` are markup; the text between the tags is what is read."""
    assert _hits(f"by {tag}{written}{close} today\n") == [(1, 0)], (tag, written)


@pytest.mark.parametrize(
    "text",
    [f"- [x]{INVERTED}\n", f"<b>{INITIAL}</b>\n", f'<a href="u">{INVERTED}</a>\n'],
)
def test_blanked_marks_preserve_every_offset(text: str) -> None:
    """One character out for one character in: nothing moves."""
    blanked = scan_secrets._without_markup_marks(text)
    assert len(blanked) == len(text), text
    assert blanked.index(LAST) == text.index(LAST), text


@pytest.mark.parametrize(
    "text, opening",
    [("- [x]x\n", "["), ("<em>x</em>\n", "<"), ('<a href="u">x</a>\n', "<")],
)
def test_the_mark_keeps_only_its_opening_character(text: str, opening: str) -> None:
    """The kept character is the boundary the mark already was."""
    blanked = scan_secrets._without_markup_marks(text)
    removed = {char for char, was in zip(blanked, text) if char != was}
    assert removed == {" "}, text
    assert opening in blanked, text


@pytest.mark.parametrize(
    "text",
    [
        f"- [x] {FIRST}\n- [x] {LAST}\n",
        f"<td>{FIRST}</td><td>{LAST}</td>\n",
        f"* [X]{FIRST}\n* [X]{LAST}\n",
        f"<li>{FIRST}</li>\n<li>{LAST}</li>\n",
    ],
)
def test_two_items_or_cells_are_never_one_name(text: str) -> None:
    """The kept opening character ends the run before the mark, as before."""
    assert _runs(text) == [], text
    assert _hits(text, INVERTED) == [], text


@pytest.mark.parametrize(
    "text",
    [
        f"array[x]{INITIAL}\n",
        f"see [x]{INVERTED} here\n",
        f"value [X]{INITIAL_AFTER}\n",
    ],
)
def test_a_box_that_opens_no_list_item_is_left_as_written(text: str) -> None:
    """Outside a list item's start, ``[x]`` is an index or a class, not a box."""
    assert scan_secrets._without_markup_marks(text) == text, text


@pytest.mark.parametrize(
    "text",
    [f"<a\nhref='u'>{INITIAL}</a>\n", f"<span\nclass='n'>{INVERTED}</span>\n"],
)
def test_a_tag_written_across_a_line_break_is_not_blanked(text: str) -> None:
    """The opening tag's delimiter spans a line; only its closing tag is blanked."""
    blanked = scan_secrets._without_markup_marks(text)
    assert blanked.startswith(text.split("\n")[0]), text


@pytest.mark.parametrize(
    "text",
    [f'<a title="x {OTHER}">y</a>\n', f'<span data-n="{OTHER} z">w</span>\n'],
)
def test_attribute_text_is_kept_where_it_was_written(text: str) -> None:
    """Only delimiters and the element's own name are blanked."""
    assert OTHER in scan_secrets._without_markup_marks(text), text


@pytest.mark.parametrize(
    "written, other",
    [(f"- [x]{FIRST[0]}. {OTHER}\n", FULL), (f"<b>{OTHER}, {FIRST}</b>\n", FULL)],
)
def test_another_person_behind_a_mark_is_still_not_the_entry(
    written: str, other: str
) -> None:
    """Reading through the mark adds reading, never a new match."""
    assert _hits(written, other) == [], written


@pytest.mark.parametrize(
    "code, prose",
    [
        (f'x = "<b>{INVERTED}</b>"\n', f"<b>{INVERTED}</b>\n"),
        (f"tpl = '<td>{INITIAL}</td>'\n", f"- [x]{INITIAL}\n"),
    ],
)
def test_source_files_are_read_exactly_as_before(code: str, prose: str) -> None:
    """The blanking is prose layout; a source line is read as written."""
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: FULL})
    assert scan_secrets.scan_client_document(
        code, patterns, allow_line_join=False
    ) == [], code
    assert scan_secrets.scan_client_document(prose, patterns) == [(1, 0)], prose
