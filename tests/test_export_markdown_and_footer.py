"""A client-facing export must not ship raw markdown, and a footer must be one.

THE INCIDENT (H1/H2, gate battery live ``13b2bf7``, 2026-08-31). The exported
docx was inspected by walking its ZIP central directory rather than trusting
the content-type header, and two things came back:

* ``word/document.xml`` contained, literally::

      The Accepted Contract Amount, **excluding VAT**, is:**SAR 1,754,504,456.25**

  Asterisks as characters. The exporter's own docstring said it did not parse
  markdown, so this was a known gap that had been shipping.

* **No ``word/footer*.xml`` part existed at all.** What the code called a
  footer was ``doc.add_paragraph(...)`` at the end of the body -- which does
  not repeat on the page and is not a footer. The requirement "footer = live
  URL, never localhost" was unmet: not violated, absent.

H2 found the same markers leaking into spreadsheet cells
(``**Total activities:** 204``).

These are bugs against a stated requirement, so they land now (owner's ruling
R4). Multi-message A1–A9 report export (H1b) lives in
``tests/test_answer_report_export.py``; this file stays the single-message
markdown + footer contract.

Every assertion below reads the produced file, never a return value.
"""

import zipfile

import pytest

from app.lib.markdown_docx import (
    iter_inline,
    split_blocks,
    strip_inline_markdown,
)
from app.routers.exports import (
    _public_base_url_for,
    _render_message_docx,
    _render_message_xlsx,
)

# The H1 answer's opening line, verbatim from the evidence pack.
H1_LINE = ("The Accepted Contract Amount, **excluding VAT**, "
           "is:**SAR 1,754,504,456.25**")

H1_ANSWER = f"""{H1_LINE}

## Breakdown

| Item | Qty | Rate | Amount |
|---|---|---|---|
| Site clearance | 158 ha | 186,328.00 | **29,439,824.00** |

- Contract Data 1.1.1
- Exact to the halala

The identifier `message_index` and duration_overrides_applied are unchanged.
"""


class _Req:
    def __init__(self, headers):
        self.headers = headers


def _docx(text, base_url="https://theshovel.ai"):
    path = _render_message_docx("PRJ2 Infra", text, "conv1234abcd", -1, base_url)
    return zipfile.ZipFile(path)


# --------------------------------------------------------------------------
# The markdown bug
# --------------------------------------------------------------------------

def test_no_literal_asterisks_reach_the_document_body():
    z = _docx(H1_ANSWER)
    doc = z.read("word/document.xml").decode()
    assert "**" not in doc
    assert "is:**SAR" not in doc


def test_bold_becomes_real_bold_runs():
    doc = _docx(H1_ANSWER).read("word/document.xml").decode()
    assert "<w:b/>" in doc, "no bold run property emitted"
    assert doc.count("<w:b/>") >= 3


def test_the_figure_survives_intact():
    doc = _docx(H1_ANSWER).read("word/document.xml").decode()
    assert "SAR 1,754,504,456.25" in doc
    assert "29,439,824.00" in doc


def test_a_pipe_table_becomes_a_word_table_not_a_row_of_pipes():
    doc = _docx(H1_ANSWER).read("word/document.xml").decode()
    assert "<w:tbl>" in doc
    assert "|---|" not in doc


def test_headings_become_headings():
    doc = _docx("## Breakdown\n\nbody text\n").read("word/document.xml").decode()
    assert "Heading" in doc
    assert "## Breakdown" not in doc


def test_snake_case_identifiers_are_never_italicised_or_eaten():
    """Underscore emphasis is deliberately unsupported: construction answers
    are full of snake_case, and turning the middle of an identifier into
    italics makes the document less true, not more finished."""
    text = "Fields: message_index, duration_overrides_applied, generate_wbs.\n"
    doc = _docx(text).read("word/document.xml").decode()
    for ident in ("message_index", "duration_overrides_applied", "generate_wbs"):
        assert ident in doc, ident


@pytest.mark.parametrize("raw,plain", [
    ("**bold**", "bold"),
    ("*italic*", "italic"),
    ("`code`", "code"),
    ("a **b** c *d* e", "a b c d e"),
    ("2 * 3 * 4", "2 * 3 * 4"),
    ("no markers here", "no markers here"),
    ("snake_case_name", "snake_case_name"),
])
def test_inline_stripping(raw, plain):
    assert strip_inline_markdown(raw) == plain


def test_bold_wins_over_italic_on_a_double_delimiter():
    styles = [s for s, _ in iter_inline("**x**") if s]
    assert styles == ["bold"]


def test_block_parser_separates_tables_from_prose():
    blocks = split_blocks(H1_ANSWER)
    kinds = [b["kind"] for b in blocks]
    assert "table" in kinds
    assert "heading" in kinds
    assert kinds.count("bullet") == 2
    table = next(b for b in blocks if b["kind"] == "table")
    assert table["rows"][0] == ["Item", "Qty", "Rate", "Amount"]
    assert len(table["rows"]) == 2, "separator row must be dropped"


def test_fenced_code_survives_verbatim():
    blocks = split_blocks("text\n\n```\nline one\n  line two\n```\n\nafter")
    code = next(b for b in blocks if b["kind"] == "code")
    assert code["text"] == "line one\n  line two"


# --------------------------------------------------------------------------
# The footer bug
# --------------------------------------------------------------------------

def test_a_real_footer_part_exists():
    """A trailing body paragraph is not a footer. python-docx only writes a
    footer part when section.footer is written, which is exactly why H1's ZIP
    walk found none."""
    names = _docx(H1_ANSWER).namelist()
    assert any(n.startswith("word/footer") and n.endswith(".xml") for n in names), names


def test_the_footer_carries_the_live_url():
    z = _docx(H1_ANSWER)
    part = next(n for n in z.namelist() if n.startswith("word/footer"))
    footer = z.read(part).decode()
    assert "theshovel.ai" in footer
    assert "Generated by The Shovel" in footer


def test_the_footer_never_carries_localhost():
    """#332's class: a dead or private host stamped into a client deliverable."""
    z = _docx(H1_ANSWER, base_url="")
    part = next(n for n in z.namelist() if n.startswith("word/footer"))
    footer = z.read(part).decode()
    assert "localhost" not in footer
    assert "Generated by The Shovel" in footer
    body = z.read("word/document.xml").decode()
    assert "localhost" not in body


@pytest.mark.parametrize("host", [
    "localhost", "localhost:8000", "127.0.0.1:8000", "0.0.0.0:80",
    "10.0.0.4", "192.168.1.9", "172.16.0.3", "box.local",
])
def test_private_hosts_are_never_promoted_to_a_public_url(host):
    assert _public_base_url_for(_Req({"host": host})) == ""


def test_a_public_host_from_the_request_is_used_when_no_env_is_set():
    """H1 found NO url in the export: neither env var was set on the live
    service and the code correctly refused to invent one. The request knows
    the host the client actually reached, including the custom domain that
    replaced the render slug."""
    assert _public_base_url_for(_Req({
        "host": "theshovel.ai", "x-forwarded-proto": "https",
    })) == "https://theshovel.ai"


def test_forwarded_host_wins_and_a_list_is_reduced_to_the_first_entry():
    assert _public_base_url_for(_Req({
        "x-forwarded-host": "theshovel.ai, internal.svc",
        "host": "internal.svc",
        "x-forwarded-proto": "https, http",
    })) == "https://theshovel.ai"


def test_env_still_wins_over_the_request(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://configured.example")
    assert _public_base_url_for(_Req({"host": "theshovel.ai"})) == "https://configured.example"


def test_no_request_and_no_env_yields_no_url():
    assert _public_base_url_for(None) == ""


def test_a_broken_request_object_never_breaks_the_export():
    class Exploding:
        @property
        def headers(self):
            raise RuntimeError("boom")
    assert _public_base_url_for(Exploding()) == ""


# --------------------------------------------------------------------------
# The same leak in the spreadsheet (H2)
# --------------------------------------------------------------------------

def _xlsx_cells(text):
    import openpyxl
    wb = openpyxl.load_workbook(_render_message_xlsx("PRJ2 Infra", text, "conv1234abcd"))
    return [c.value for ws in wb.worksheets for row in ws.iter_rows()
            for c in row if c.value is not None]


def test_markdown_markers_never_reach_a_spreadsheet_cell():
    cells = _xlsx_cells("**Total activities:** 204\n\n| A | **B** |\n|---|---|\n| 1 | 2 |\n")
    assert not any("**" in str(c) for c in cells), cells
    assert "Total activities: 204" in cells
    assert "B" in cells


def test_the_spreadsheet_still_carries_its_figures():
    cells = _xlsx_cells("| Item | Amount |\n|---|---|\n| Site clearance | **29,439,824.00** |\n")
    assert "29,439,824.00" in cells


# --------------------------------------------------------------------------
# Nothing here may break an export
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "no markdown at all",
                                  "| broken table\n", "```\nunclosed fence\n",
                                  "***", "**", "*"])
def test_degenerate_input_still_produces_a_valid_docx(text):
    z = _docx(text)
    assert "word/document.xml" in z.namelist()
    assert z.testzip() is None


def test_bold_in_plain_prose_alone_produces_a_bold_run():
    """Deliberately no table and no heading: both emit their own bold, so a
    count over the whole document proves nothing about inline emphasis. This
    is the H1 line by itself."""
    doc = _docx(H1_LINE + "\n").read("word/document.xml").decode()
    assert "<w:tbl>" not in doc
    assert "<w:b/>" in doc, "inline bold produced no bold run"
    assert doc.count("<w:b/>") >= 2, "both bold spans should be bold"
    assert "**" not in doc


def test_italic_in_plain_prose_alone_produces_an_italic_run():
    doc = _docx("The amount is *indicative* only.\n").read("word/document.xml").decode()
    assert "<w:i/>" in doc
    assert "*indicative*" not in doc


def test_a_template_without_list_styles_still_exports():
    """add_paragraph(style=...) raises KeyError on a base template that lacks
    the style. An export must degrade to an unstyled paragraph, not 500."""
    from app.lib.markdown_docx import _styled_paragraph

    class _NoStyles:
        def __init__(self):
            self.calls = []

        def add_paragraph(self, style=None):
            self.calls.append(style)
            if style is not None:
                raise KeyError(style)
            return "unstyled"

    doc = _NoStyles()
    assert _styled_paragraph(doc, "List Bullet") == "unstyled"
    assert doc.calls == ["List Bullet", None]
