"""A denylisted phrase has more than one spelling, and the gate must know them.

The env-fed client-pattern denylist used to compare a pattern to the raw line
and to its escape-decoded form. Both comparisons are literal, so a name was
only caught in the exact spacing the list happened to be typed in: hyphenated
it walked past, joined by a non-breaking space it walked past, split by a
zero-width space it walked past, wrapped onto the next line it walked past.
Each of those is the same person, and a privacy gate that recognises one
spelling of a name recognises the spelling nobody had to hide.

The repair has to survive the opposite failure at the same time, and this file
records it: a fold that joins *any* two words read every underscored file name
in this repository as the words around it, and a match allowed to ignore word
order and the words between them turned ordinary prose into hits. Seventeen
files that main scans clean went red. So the rules here are: fold presentation,
never word order; fold prose separators, never code's joiners; join lines only
where a human reads a paragraph; and refuse a folded match that sits inside a
longer identifier. What the scanner still does not catch is asserted as a gap
in its own test rather than left for a reader to assume.

The words are invented here and assembled at runtime, so the repository
contains no name and no fixture that the real scan would have to chase.

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


# -- every spelling of one name -------------------------------------------------


@pytest.mark.parametrize(
    "label, text",
    [
        ("plain", f"{FULL} reviewed this change.\n"),
        ("hyphen", f"{FIRST}-{LAST} reviewed this change.\n"),
        ("en dash", f"{FIRST}\u2013{LAST} reviewed this change.\n"),
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
        ("accent on the surname", f"{FIRST} Vantr\u00e9y signed.\n"),
        ("decomposed accent", f"{FIRST} Vantre\u0301y signed.\n"),
        ("full-width letters", f"{FULL[0]}\uff55illon {LAST} signed.\n"),
        ("lower case", f"{FULL.lower()} signed.\n"),
    ],
)
def test_one_name_in_any_spelling_is_caught(label, text):
    assert _hits(text) == [(1, 0)], label


@pytest.mark.parametrize(
    "label, text",
    [
        ("surname first", f"{LAST} {FIRST} reviewed this change.\n"),
        ("middle initial", f"{FIRST} A. {LAST} reviewed this change.\n"),
        ("full middle name", f"{FIRST} Ambrose {LAST} reviewed this change.\n"),
        ("surname first with a middle initial", f"{LAST} A. {FIRST} signed.\n"),
        (
            "surname first, decomposed accent",
            f"{LAST[:6]}\u0301{LAST[6:]} {FIRST} signed.\n",
        ),
        (
            "surname first, soft hyphen inside a word",
            f"{LAST[:4]}\u00ad{LAST[4:]} {FIRST} signed.\n",
        ),
    ],
)
def test_a_name_written_surname_first_or_with_a_middle_name_is_caught(label, text):
    """The orders a written name actually takes, and nothing wider.

    A person is spelled surname-first in an attribution and with a middle name
    or an initial in a legal one. Both are the exact sequence of the
    denylisted words -- reversed, or with one word inserted between them -- so
    they are matched by reading the run of capitalised words, never by
    matching a set of words that happen to appear in the same sentence. An
    entry that is not a written name (not every word capitalised) is not
    reordered at all.
    """
    assert _hits(text) == [(1, 0)], label


def test_an_entry_that_is_not_a_written_name_is_never_reordered():
    """Only a name-shaped entry gets the name rule.

    A lower-case entry is a phrase, not a person: the folded phrase match
    already covers it, and reordering a phrase's words would invent names out
    of ordinary vocabulary -- which is the false positive the withdrawn
    word-set match produced.
    """
    phrase = f"{FIRST.lower()} {LAST.lower()}"
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: phrase})
    assert scan_secrets.scan_client_document(phrase + " signed\n", patterns) == [
        (1, 0)
    ]
    reversed_phrase = f"{LAST.lower()} {FIRST.lower()} signed\n"
    assert scan_secrets.scan_client_document(reversed_phrase, patterns) == []


@pytest.mark.parametrize(
    "text",
    [
        f"{FIRST}_{LAST} reviewed this change.\n",
        f"see docs/{FIRST.lower()}_{LAST.lower()}.md\n",
    ],
)
def test_an_underscore_between_the_words_is_an_identifier_not_a_name(text):
    """Underscore is code's joiner: a run of name words never spans one."""
    assert _hits(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        f"{LAST}. {FIRST} signed.\n",
        f"{FIRST} and {LAST} signed.\n",
        f"{LAST} owns it; {FIRST} reviewed it.\n",
    ],
)
def test_a_name_run_does_not_bridge_a_sentence_or_a_conjunction(text):
    """Two capitalised words are not a name when a break sits between them.

    The full stop of one sentence, a lower-case conjunction, and a semicolon
    all end the run. Without that, the rule would read one sentence's last
    word and the next sentence's first as a surname and a given name.
    """
    assert _hits(text) == [], text


@pytest.mark.parametrize(
    "label, prefix, expected",
    [
        ("name on the second line", "one\n", 2),
        ("name deeper in the page", "one\ntwo\n\nthree\n", 5),
    ],
)
def test_a_reordered_name_is_reported_at_the_line_it_starts_on(
    label, prefix, expected
):
    hits = _hits(prefix + f"{LAST} {FIRST} signed\n")
    assert hits == [(expected, 0)], label


@pytest.mark.parametrize(
    "label, prefix, expected",
    [
        ("wrapped after the first word", "intro\n", 2),
        ("wrapped deeper in the page", "one\ntwo\n\nthree\n", 5),
    ],
)
def test_the_wrapped_name_is_reported_at_the_line_it_starts_on(
    label, prefix, expected
):
    """A location, not the text: the first line of the match is the address."""
    hits = _hits(prefix + f"{FIRST}\n{LAST} signed\n")
    assert hits == [(expected, 0)], label


@pytest.mark.parametrize(
    "spelling",
    ["_".join(["qvant" + "rey", "zz"]), "-".join(["qvant" + "rey", "zz"])],
)
def test_a_handle_split_by_a_separator_still_matches(spelling):
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: spelling})
    hits = scan_secrets.scan_client_document(f"author: {spelling}\n", patterns)
    assert hits == [(1, 0)], spelling


# -- the other direction: no manufactured hits ---------------------------------


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


@pytest.mark.parametrize(
    "phrase, text",
    [
        ("the of", "the\nsilt and gravel\nof the bank\n"),
        ("a in", "a\nspade and a barrow\nin the trench\n"),
    ],
)
def test_a_pattern_of_function_words_is_not_matched_across_lines(phrase, text):
    """Two short common words are prose, not an identity.

    The literal comparison catches the phrase where it is written as one line;
    joining consecutive lines must not extend a function-word pair into every
    paragraph that happens to use both words.
    """
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: phrase})
    assert scan_secrets.scan_client_document(text, patterns) == []


@pytest.mark.parametrize(
    "phrase, text",
    [
        (r"qvantrey_\d+", "author: qvantrey_ 12\n"),
        (r"qvantrey[0-9]+", "author: qvantrey 12\n"),
    ],
)
def test_a_pattern_with_regex_syntax_is_left_to_the_literal_path(phrase, text):
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: phrase})
    assert scan_secrets.scan_client_document(text, patterns) == []


@pytest.mark.parametrize(
    "text",
    [
        f"{FIRST} walked past the vantage point\nand the vault beyond\n",
        f"the vault and the vantage, said {FIRST}\n",
    ],
)
def test_an_ordinary_word_near_a_name_word_is_not_a_hit(text):
    """The folding matches whole words, so a lookalike word is not the name."""
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: LAST})
    assert scan_secrets.scan_client_document(text, patterns) == []


@pytest.mark.parametrize(
    "template",
    [
        "see build_{slug}.py for the totals\n",
        "docs/{slug}/notes.md\n",
        "docs/{slug}.md\n",
        "the {slug}: a column of totals\n",
    ],
)
def test_a_phrase_inside_an_identifier_is_not_a_hit(template):
    """An identifier is one word to a reader, and code naming is not content.

    Without this rule the folded comparison reported ordinary files of this
    repository -- script names, test fixtures -- as client content, on a
    denylist that main scans clean.
    """
    text = template.format(slug=f"{FIRST.lower()}_{LAST.lower()}")
    assert _hits(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        "run scan_secrets before the build\n",
        "see the scan_secrets module\n",
    ],
)
def test_an_underscore_never_joins_two_words_of_a_name(text):
    """Underscore is code's joiner, not prose's.

    A denylist entry written as two words must not match an identifier that
    joins those two words with an underscore: folding it as a separator made
    every underscored identifier in the tree read as the words around it, which
    is how a denylist of ordinary product vocabulary turned into hundreds of
    hits on this repository's own files.
    """
    # The phrase is assembled, never spelled: writing a denylist entry into a
    # tracked line would make the scan find its own fixture, which is exactly
    # the noise this rule exists to remove.
    phrase = " ".join(("scan", "secrets"))
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: phrase})
    assert scan_secrets.scan_client_document(text, patterns) == []


@pytest.mark.parametrize(
    "text",
    [
        f"Signed by {FULL}.\n",
        f"Signed by {FULL}, reviewed.\n",
    ],
)
def test_a_phrase_ending_a_sentence_is_still_a_hit(text):
    """The dot of a full stop is not the dot of a file extension."""
    assert _hits(text) == [(1, 0)], text


@pytest.mark.parametrize(
    "label, spelling, expected",
    [
        ("exact spelling in a path", lambda handle: handle, [(1, 0)]),
        (
            "folded-only spelling in a path",
            lambda handle: "".join(
                chr(ord(c) - ord("a") + ord("\uff41")) if c.islower() else c
                for c in handle
            ),
            [],
        ),
    ],
)
def test_a_handle_inside_an_identifier_does_not_fire(label, spelling, expected):
    """A literal hit is never suppressed; a folded-only one inside a path is.

    The denylist named the exact string, so finding it in a path is a real
    finding and stays one. The folded path is a judgement: a full-width
    spelling sitting inside a file name is the fold having read an identifier
    as two words, not a person's handle in prose.
    """
    handle = "_".join(["qvant" + "rey", "zz"])
    patterns = scan_secrets.load_client_patterns({scan_secrets.ENV_VAR: handle})
    text = f"tests/{spelling(handle)}.py\n"
    assert scan_secrets.scan_client_document(text, patterns) == expected, label


@pytest.mark.parametrize(
    "text",
    [
        f"{FULL} signed\n",
        f"reviewed by {FULL}\n",
    ],
)
def test_one_location_is_reported_once(text):
    """Both the literal path and the folding path see this line."""
    hits = _hits(text)
    assert hits == [(1, 0)], "the same line was reported twice"


# -- the interface that mattered: the file walk, the exit code, the silence ----


@pytest.mark.parametrize(
    "label, body",
    [
        ("no-break space", f"Attribution: {FIRST}\u00a0{LAST}\n"),
        ("zero-width space", f"Attribution: {FIRST}\u200b{LAST}\n"),
    ],
)
def test_the_file_walk_uses_the_folding_and_reports_location_only(
    label, body, tmp_path, capsys, monkeypatch
):
    fixture = tmp_path / "notes.md"
    fixture.write_text(body, encoding="utf-8")
    monkeypatch.setenv(scan_secrets.ENV_VAR, FULL)
    monkeypatch.setattr("sys.argv", ["scan_secrets.py", "--paths", str(fixture)])

    assert scan_secrets.main() == 1, label
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "notes.md:1" in combined
    assert "pattern #0" in combined
    for secret in (FIRST, LAST, FULL):
        assert secret not in combined, "the denylisted text was echoed"


@pytest.mark.parametrize(
    "argv",
    [
        ["scan_secrets.py", "--paths"],
        ["scan_secrets.py"],
    ],
)
def test_a_missing_denylist_still_refuses(argv, monkeypatch, capsys):
    """The folding must not have opened a path around the fail-closed check."""
    monkeypatch.delenv(scan_secrets.ENV_VAR, raising=False)
    monkeypatch.setattr("sys.argv", argv)
    assert scan_secrets.main() == 2
    assert "NO SECRET MATERIAL" not in capsys.readouterr().out
