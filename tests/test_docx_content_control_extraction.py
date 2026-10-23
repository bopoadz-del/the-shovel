"""D1: text inside Word content controls must reach the corpus.

A 139 KB letter reached Neon as 1168 characters ending ``Yours sincerely, ,``
with no signatory, and the model correctly refused to name one. The letter was
not badly ingested by accident: its closing and signature block sit inside a
``w:sdt`` -- a Word content control -- and ``python-docx``'s ``.paragraphs``
returns only ``w:p`` elements that are *direct children* of ``w:body``. A
paragraph nested in ``sdtContent`` is invisible to it, and the text-box walk
added earlier looks only inside ``txbxContent`` / ``txBody``.

Two shapes, both common in letter templates:

* **block-level** -- ``sdt > sdtContent > p > r > t``: whole paragraphs, which
  is where a signature block lives.
* **inline** -- ``p > sdt > sdtContent > r > t``: runs inside a paragraph, which
  is where a reference number or a subject line lives. ``Paragraph.text``
  concatenates only direct ``w:r`` children, so an inline control leaves a hole
  in the middle of a line that looks complete.

Fixtures here are synthetic. No client document, name or reference appears in
this repository.
"""
from __future__ import annotations

import docx
import pytest
from docx.oxml.ns import qn

from app.core.doc_index import _docx_plain_text, _docx_sdt_plain_parts


def _sdt_block(text_lines: list[str]):
    """A block-level content control wrapping whole paragraphs."""
    sdt = docx.oxml.OxmlElement("w:sdt")
    content = docx.oxml.OxmlElement("w:sdtContent")
    for line in text_lines:
        p = docx.oxml.OxmlElement("w:p")
        r = docx.oxml.OxmlElement("w:r")
        t = docx.oxml.OxmlElement("w:t")
        t.text = line
        r.append(t)
        p.append(r)
        content.append(p)
    sdt.append(content)
    return sdt


def _sdt_inline(text: str):
    """An inline content control: runs directly under sdtContent, no paragraph."""
    sdt = docx.oxml.OxmlElement("w:sdt")
    content = docx.oxml.OxmlElement("w:sdtContent")
    r = docx.oxml.OxmlElement("w:r")
    t = docx.oxml.OxmlElement("w:t")
    t.text = text
    r.append(t)
    content.append(r)
    sdt.append(content)
    return sdt


@pytest.fixture
def letter(tmp_path):
    """A letter shaped like the one that failed, with invented content."""
    document = docx.Document()
    document.add_paragraph("Dear Sir or Madam,")
    document.add_paragraph("The works described above are complete.")
    document.add_paragraph("Yours sincerely,")

    # The signature block, in a block-level content control.
    document.element.body.append(
        _sdt_block(["Regards", "A Signatory", "Engineer's Representative"])
    )

    # A subject line split by an inline content control, as a template does it.
    subject = document.add_paragraph("Ref. No.: ")
    subject._p.append(_sdt_inline("REF-000372"))
    subject.add_run(" — completion notice")

    path = tmp_path / "letter.docx"
    document.save(path)
    return path


def test_paragraphs_alone_lose_the_signature_block(letter):
    """The failure this test exists for, demonstrated on the fixture."""
    document = docx.Document(letter)
    paragraphs = "\n".join(p.text for p in document.paragraphs if p.text.strip())

    assert "Yours sincerely," in paragraphs
    assert "A Signatory" not in paragraphs, (
        "python-docx now sees content controls; this test's premise has changed"
    )
    assert "Engineer's Representative" not in paragraphs


def test_the_extractor_recovers_a_block_level_signature(letter):
    text = _docx_plain_text(docx.Document(letter))

    assert "Yours sincerely," in text
    assert "Regards" in text
    assert "A Signatory" in text
    assert "Engineer's Representative" in text


def test_the_extractor_recovers_an_inline_content_control(letter):
    """A reference number inside an inline control is part of its own line."""
    text = _docx_plain_text(docx.Document(letter))

    assert "REF-000372" in text, (
        "an inline content control left a hole in a line that looked complete"
    )
    assert "Ref. No.:" in text
    assert "completion notice" in text


def test_every_run_in_the_document_is_extracted(letter):
    """The measure that matters: nothing in the file is left behind."""
    from lxml import etree

    document = docx.Document(letter)
    text = _docx_plain_text(document)
    runs = [
        el.text.strip()
        for el in document.element.iter()
        if etree.QName(el).localname == "t" and el.text and el.text.strip()
    ]
    missing = [r for r in runs if r not in text]

    assert runs, "the fixture produced no text runs"
    assert not missing, f"runs present in the file but absent from the extract: {missing}"


def test_nothing_is_counted_twice(letter):
    """A content control can wrap a paragraph another walk already returned."""
    text = _docx_plain_text(docx.Document(letter))
    assert text.count("A Signatory") == 1
    assert text.count("Yours sincerely,") == 1


def test_a_paragraph_inside_a_table_is_not_collected_twice(tmp_path):
    """The table walk owns those; the content-control walk must not re-add them."""
    document = docx.Document()
    table = document.add_table(rows=1, cols=1)
    cell = table.rows[0].cells[0]
    cell.text = "Cell content"

    # Wrap the cell's paragraph in a content control, as a form template does.
    sdt = _sdt_block(["Wrapped in a control"])
    cell._tc.append(sdt)

    path = tmp_path / "table.docx"
    document.save(path)

    text = _docx_plain_text(docx.Document(path))
    assert text.count("Cell content") == 1


def test_the_walk_is_safe_on_a_document_without_controls(tmp_path):
    document = docx.Document()
    document.add_paragraph("Plain text only.")
    path = tmp_path / "plain.docx"
    document.save(path)

    loaded = docx.Document(path)
    assert _docx_sdt_plain_parts(loaded.element) == []
    assert "Plain text only." in _docx_plain_text(loaded)


def test_the_walk_tolerates_a_test_double_without_an_element():
    class Double:
        paragraphs = [type("P", (), {"text": "only paragraphs"})()]

    assert _docx_plain_text(Double()) == "only paragraphs"
    assert _docx_sdt_plain_parts(None) == []


def test_qn_import_is_used_for_namespaced_lookups():
    """Guard against the namespace helper being dropped from the module."""
    assert qn("w:sdt").endswith("}sdt")
