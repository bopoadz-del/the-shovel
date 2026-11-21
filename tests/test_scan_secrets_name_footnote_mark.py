"""A footnote reference welded to the word it marks.

A footnote is written in brackets after the word it annotates -- ``Surname[^1]``
-- and the identifier inside the brackets is a pointer the document resolves at
its foot, never a word the surname spells. Welded to the surname's last word it
makes that word name-shaped in no case at all, so the run reader loses the entry
it exists to read; and where the reference stands before a link's own mark
(``Surname[^1](url)``) the footnote's bracket pair is read as the link's closing
mark, so the surname keeps the identifier instead.

The mark is layout, exactly as the destination, the reference and the mark that
holds a space already are: it is blanked to spaces of its own length before any
token is built, so every offset stays where the document wrote it and the word
before the mark ends where the mark opens. Nothing is joined that the line did
not write -- blanking separates, and the word after the mark is the next token.

What is *not* widened: a mark whose bracket never closes is punctuation and not
a footnote; a mark written across a line break is not a footnote either; a
bracket pair that names a word rather than a footnote is still a bracket pair.
The words are invented and assembled at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_name_footnote_mark", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_name_footnote_mark"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"
INVERTED = f"{LAST}, {FIRST}"
URL = "https://example.invalid/doc"


def _hits(text: str, entry: str = FULL) -> list[tuple[int, int]]:
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: entry})
    return scan_secrets.scan_client_document(text, patterns)


def _runs(text: str) -> list[list[str]]:
    return [run for _, run in scan_secrets.name_runs(text)]


@pytest.mark.parametrize(
    "mark",
    ["[^1]", "[^note]", "[^a b]", "[^1-2]", "[^_x]"],
)
def test_footnote_after_the_surname_keeps_the_entry_readable(mark: str) -> None:
    """The surname arrives welded to a footnote, not to a destination."""
    text = f"See {INVERTED}{mark} crew.\n"
    assert _hits(text) == [(1, 0)], mark


@pytest.mark.parametrize(
    "mark, url",
    [("[^1]", URL), ("[^n]", "https://example.invalid/other"), ("[^a b]", URL)],
)
def test_footnote_before_a_link_mark_is_not_the_link_s_own_mark(
    mark: str, url: str
) -> None:
    """``Surname[^1](url)`` is a footnote and a link, not an identifier."""
    text = f"See [{INVERTED}]({url}){mark} and {INVERTED}{mark}({url}) crew.\n"
    assert _hits(text) == [(1, 0)], mark


@pytest.mark.parametrize(
    "mark, tail",
    [("[^1]", " noted."), ("[^id]", " and more.")],
)
def test_footnote_definition_line_names_its_owner(mark: str, tail: str) -> None:
    """The definition writes the mark first; the name after it is still read."""
    text = f"{mark}: {INVERTED} wrote it.{tail}\n"
    assert _hits(text) == [(1, 0)], mark


@pytest.mark.parametrize(
    "label",
    [f"[{LAST}[^1], {FIRST}]", f"[{LAST}[^n], {FIRST}]"],
)
def test_footnote_inside_a_link_label_is_layout_too(label: str) -> None:
    """A label may carry a footnote of its own before the mark closes it."""
    text = f"See {label}({URL}) crew.\n"
    assert _hits(text) == [(1, 0)], label


@pytest.mark.parametrize(
    "mark",
    ["[^1]", "[^note]"],
)
def test_blanked_mark_preserves_every_offset(mark: str) -> None:
    """The mark becomes spaces of its own length, so no offset moves."""
    text = f"See {INVERTED}{mark} crew at 10:00.\n"
    blanked = scan_secrets._without_footnote_marks(text)
    assert len(blanked) == len(text), mark
    assert blanked.count("[^") == 0, mark


@pytest.mark.parametrize(
    "mark",
    ["[^1]", "[^note]"],
)
def test_run_after_a_footnote_starts_where_the_document_wrote_it(mark: str) -> None:
    """The run's offset is the surname's own place in the block."""
    text = f"note {mark} first.\nthen {INVERTED} crew.\n"
    runs = scan_secrets.name_runs(text)
    assert [run for _, run in runs] == [["vantrey", "quillon"]], mark
    assert runs[0][0] == text.index(LAST, text.index("then")), mark


@pytest.mark.parametrize(
    "text",
    [
        f"see {INVERTED}[^1 crew.\n",
        f"see {INVERTED}[^note crew.\n",
    ],
)
def test_an_unclosed_mark_is_left_where_it_stands(text: str) -> None:
    """A bracket pair that never closes is punctuation, not a footnote."""
    assert scan_secrets._without_footnote_marks(text) == text, text
    assert _hits(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        f"see {INVERTED}[^\n1] crew.\n",
        f"see {INVERTED}[^note\nhere] crew.\n",
    ],
)
def test_a_mark_written_across_a_line_break_is_not_blanked(text: str) -> None:
    """A footnote is written on the line that spells it."""
    assert scan_secrets._without_footnote_marks(text) == text, text


@pytest.mark.parametrize(
    "text",
    [
        "value = grid[^mask]\nother = table[^row]\n",
        "if cells[^index] == 3:\n    pass\n",
    ],
)
def test_a_subscript_named_after_a_word_is_not_a_name(text: str) -> None:
    """Lower-case identifiers keep the bracket; no name run is manufactured."""
    assert _hits(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        f"See [{LAST}][{FIRST}] crew.\n",
        f"See {LAST}[^{FIRST}] crew.\n",
    ],
)
def test_two_bracket_groups_are_still_not_one_name(text: str) -> None:
    """A second bracket group is its own token, never the first one's surname."""
    assert _hits(text) == [], text


@pytest.mark.parametrize(
    "text, label",
    [
        (f"see {INVERTED} crew.\n", "plain"),
        (f"see {INVERTED}[^1]\n", "footnote"),
    ],
)
def test_the_word_before_the_mark_ends_where_the_mark_opens(
    text: str, label: str
) -> None:
    """The mark's own identifier is never read as a word of the name."""
    runs = [run for run in _runs(text) if run[-1:] == ["quillon"]]
    assert len(runs) == 1, label
    assert runs[0] == ["vantrey", "quillon"], label


@pytest.mark.parametrize(
    "code, prose",
    [
        (f'pattern = r"{LAST}[^.]{FIRST}"\n', f"see {LAST}[^1] {FIRST} crew.\n"),
        (f"re.compile(r'{LAST}[^/]{FIRST}')\n", f"see [{LAST}[^n], {FIRST}]({URL}) crew.\n"),
    ],
)
def test_a_regular_expression_class_is_not_a_footnote_mark(
    code: str, prose: str
) -> None:
    """The same bracket shape negates a class in code; only prose carries marks."""
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: FULL})
    assert scan_secrets.scan_client_document(
        code, patterns, allow_line_join=False
    ) == [], code
    assert scan_secrets.scan_client_document(prose, patterns) == [(1, 0)], prose
