"""An oversize markdown table is chunked as row groups that keep its header.

``chunk_markdown`` (the curated knowledge-note chunker) kept a table whole up
to ``max_table_chars`` and char-chunked anything larger, so every chunk after
the first held data rows with no column-header row; the index ceiling
(``CHUNK_MAX_CHARS``) then re-cut tables between the ceiling and the cap the
same way. Every input here is built in the test; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import bound_chunks
from app.core.doc_index import _markdown_table_chunks, chunk_markdown


def _table(heading: str, cols: list[str], rows: int, cell: str) -> tuple[str, str, list[str]]:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    data = [
        "| " + " | ".join(f"{c} {cell} {r:03d}" for c in cols) + " |"
        for r in range(rows)
    ]
    body = "\n".join([header, sep, *data])
    text = (f"{heading}\n\n" if heading else "") + "Intro sentence for the note.\n\n" + body + "\n"
    return text, header, data


INPUTS = {
    "rates": _table("## Supply rates", ["Item", "Unit", "Rate", "Source"], 60, "alpha"),
    "specs": _table("## Mix limits", ["Grade", "Cement", "Slump", "Cover", "Note"], 45, "bravo"),
}


def _table_chunks(chunks: list[str]) -> list[str]:
    return [c for c in chunks if c.count("|") >= 4]


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_every_table_chunk_carries_its_header(name):
    text, header, _ = INPUTS[name]
    chunks = bound_chunks(chunk_markdown(text))
    tables = _table_chunks(chunks)
    assert len(tables) > 1
    assert all(header in c for c in tables)


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_every_data_row_is_whole_in_one_chunk(name):
    text, _, data = INPUTS[name]
    chunks = bound_chunks(chunk_markdown(text))
    for row in data:
        assert sum(row in c.split("\n") for c in chunks) == 1, row


@pytest.mark.parametrize("ceiling", [900, 1300])
def test_groups_fit_a_lower_index_ceiling(monkeypatch, ceiling):
    monkeypatch.setenv("CHUNK_MAX_CHARS", str(ceiling))
    text, header, data = INPUTS["rates"]
    chunks = chunk_markdown(text)
    tables = _table_chunks(chunks)
    assert all(len(c) <= ceiling for c in tables)
    assert all(header in c for c in tables)
    assert sum(c.count("\n| ") for c in tables) - len(tables) == len(data)


@pytest.mark.parametrize("rows", [3, 8])
def test_small_table_stays_one_chunk(rows):
    text, header, data = _table("## Small", ["A", "B", "C"], rows, "x")
    tables = _table_chunks(chunk_markdown(text))
    assert len(tables) == 1
    assert tables[0].startswith("## Small\n" + header)
    assert all(row in tables[0] for row in data)


@pytest.mark.parametrize("heading", ["## Fixings schedule", "### Coating systems"])
def test_each_group_opens_on_heading_header_and_separator(heading):
    text, header, _ = _table(heading, ["Ref", "Size", "Qty"], 120, "y")
    lines = text.split("\n")
    sep = lines[lines.index(header) + 1]
    tables = _table_chunks(chunk_markdown(text))
    assert len(tables) > 1
    assert all(c.split("\n")[:3] == [heading, header, sep] for c in tables)


@pytest.mark.parametrize("cols", [["P", "Q"], ["P", "Q", "R", "S"]])
def test_headingless_group_opens_on_header(cols):
    _, header, data = _table("", cols, 50, "z")
    sep = "|" + "|".join("---" for _ in cols) + "|"
    groups = _markdown_table_chunks("", [header, sep, *data], 600)
    assert len(groups) > 1
    assert all(g.split("\n")[:2] == [header, sep] for g in groups)
    assert all(len(g) <= 600 for g in groups)


@pytest.mark.parametrize("lines", [["| a | b |"], ["| a | b |", "|---|---|"]])
def test_no_data_row_gives_no_groups(lines):
    assert _markdown_table_chunks("## H", lines, 2000) == []


@pytest.mark.parametrize("cap", [60, 100])
def test_lead_over_half_the_cap_falls_back(cap):
    header = "| " + " | ".join(f"column heading {i}" for i in range(4)) + " |"
    rows = [f"| r{i} | 1 | 2 | 3 |" for i in range(10)]
    assert _markdown_table_chunks("## Heading", [header, "|---|---|---|---|", *rows], cap) == []
