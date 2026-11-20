"""A denylisted name has more than one spelling, and the gate must know them.

The env-fed client-pattern denylist used to compare a pattern to the raw line
and to its escape-decoded form. Both comparisons are literal, so a name was
only caught in the exact spacing the list happened to be typed in: hyphenated
it walked past, joined by a non-breaking space it walked past, split by a
zero-width space it walked past, wrapped onto the next line it walked past,
written surname-first it walked past, and given a middle initial it walked
past. Each of those is the same person, and a privacy gate that recognises one
spelling of a name recognises the spelling nobody had to hide.

The tests below are the whole matrix, in both directions: every obfuscation is
caught, and the ordinary prose that shares letters with it is not. The words
are invented here, assembled at runtime, so the repository contains no name
and no fixture that the real scan would have to chase.

`scan_client_text` keeps its literal contract; `scan_client_document` is the
superset the file walk uses.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets_folding", _SCRIPTS / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets_folding"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _words() -> tuple[str, str]:
    """The invented identity, assembled so no tracked line ever spells it."""
    first = "Quillon"[:4] + "lon"
    last = "Vant" + "rey"
    return first, last


FIRST, LAST = _words()
FULL = f"{FIRST} {LAST}"


def _patterns(*extra: str) -> list:
    return scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: "\n".join((FULL, *extra))}
    )


def _hits(text: str, *extra: str) -> list[tuple[int, int]]:
    return scan_secrets.scan_client_document(text, _patterns(*extra))


# â”€â”€ every spelling of one name â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@pytest.mark.parametrize(
    "label, text",
    [
        ("plain", f"{FULL} reviewed this change.\n"),
        ("hyphen", f"{FIRST}-{LAST} reviewed this change.\n"),
        ("en dash", f"{FIRST}\u2013{LAST} reviewed this change.\n"),
        ("underscore", f"{FIRST}_{LAST} reviewed this change.\n"),
        ("no-break space", f"{FIRST}\u00a0{LAST} reviewed this change.\n"),
        ("narrow no-break space", f"{FIRST}\u202f{LAST} reviewed this change.\n"),
        ("ideographic space", f"{FIRST}\u3000{LAST} reviewed this change.\n"),
        ("tab", f"{FIRST}\t{LAST} reviewed this change.\n"),
        ("four spaces", f"{FIRST}    {LAST} reviewed this change.\n"),
        ("zero-width space", f"{FIRST}\u200b{LAST} reviewed this change.\n"),
        ("zero-width joiner", f"{FIRST}\u200d{LAST} reviewed this change.\n"),
        ("soft hyphen inside a word", f"{FIRST} Vant\u00adrey signed.\n"),
        ("byte-order mark", f"{FIRST} {LAST}\ufeff signed.\n"),
        ("wrapped line", f"{FIRST}\n{LAST} reviewed this change.\n"),
        ("surname first", f"{LAST} {FIRST} reviewed this change.\n"),
        ("middle initial", f"{FIRST} A. {LAST} reviewed this change.\n"),
        ("full middle name", f"{FIRST} Ambrose {LAST} reviewed this change.\n"),
        ("accent on the surname", f"{FIRST} Vantr\u00e9y signed.\n"),
        ("decomposed accent", f"{FIRST} Vantre\u0301y signed.\n"),
        ("full-width letters", f"{FULL[0]}\uff55illon {LAST} signed.\n"),
        ("lower case", f"{FULL.lower()} signed.\n"),
    ],
)
def test_one_name_in_any_spelling_is_caught(label, text):
    assert _hits(text) == [(1, 0)], label


def test_the_wrapped_name_is_reported_at_the_line_it_starts_on():
    """A location, not the text: the first line of the match is the address."""
    hits = _hits("intro\n" + f"{FIRST}\n{LAST} signed\n")
    assert hits == [(2, 0)]


def test_a_handle_split_by_a_separator_still_matches():
    handle = "qvant" + "rey" + "_" + "zz"
    patterns = scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: handle}
    )
    hits = scan_secrets.scan_client_document(f"author: {handle}\n", patterns)
    assert hits == [(1, 0)]


# â”€â”€ the other direction: no manufactured hits â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@pytest.mark.parametrize(
    "text",
    [
        "nothing to see here\n",
        "the quilt and the vantage point\n",
        f"a different name entirely, no relation to {FIRST} at all\n",
    ],
)
def test_clean_prose_stays_clean(text):
    assert _hits(text) == []


def test_a_pattern_of_function_words_is_not_matched_out_of_order():
    """Two short common words are prose, not an identity.

    The literal comparison still catches the phrase where it is written as a
    phrase; the order-independent path must not extend a function-word pair
    into every paragraph that happens to use both words.
    """
    patterns = scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: "the of"}
    )
    text = "the\nsilt and gravel\nof the bank\n"
    assert scan_secrets.scan_client_document(text, patterns) == []


def test_a_pattern_with_regex_syntax_is_left_to_the_literal_path():
    patterns = scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: r"qvantrey_\d+"}
    )
    text = "author: qvantrey_ 12\n"
    assert scan_secrets.scan_client_document(text, patterns) == []


def test_an_ordinary_word_near_a_name_word_is_not_a_hit():
    """The folding matches whole words, so a lookalike word is not the name."""
    patterns = scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: LAST}
    )
    text = f"{FIRST} walked past the vantage point\nand the vault beyond\n"
    assert scan_secrets.scan_client_document(text, patterns) == []


def test_one_location_is_reported_once():
    """Both the literal path and the folding path see this line."""
    hits = _hits(f"{FULL} signed\n")
    assert hits == [(1, 0)], "the same line was reported twice"


# â”€â”€ the interface that mattered: the file walk, the exit code, the silence â”€


def test_the_file_walk_uses_the_folding_and_reports_location_only(tmp_path, capsys, monkeypatch):
    fixture = tmp_path / "notes.md"
    fixture.write_text(f"Attribution: {FIRST}\u00a0{LAST}\n", encoding="utf-8")
    monkeypatch.setenv(scan_secrets.ENV_VAR, FULL)
    monkeypatch.setattr(
        "sys.argv", ["scan_secrets.py", "--paths", str(fixture)]
    )

    assert scan_secrets.main() == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "notes.md:1" in combined
    assert "pattern #0" in combined
    for secret in (FIRST, LAST, FULL):
        assert secret not in combined, "the denylisted text was echoed"


def test_a_missing_denylist_still_refuses(monkeypatch, capsys):
    """The folding must not have opened a path around the fail-closed check."""
    monkeypatch.delenv(scan_secrets.ENV_VAR, raising=False)
    monkeypatch.setattr("sys.argv", ["scan_secrets.py", "--paths"])
    assert scan_secrets.main() == 2
    assert "NO SECRET MATERIAL" not in capsys.readouterr().out
