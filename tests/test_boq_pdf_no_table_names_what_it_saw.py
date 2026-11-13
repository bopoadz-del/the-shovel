"""A BOQ PDF with no usable table: the result names what the parser saw.

S-G1 (advisor 04:22Z, 13 Nov). The no-table branch used to say "likely
scanned" / "may be a pure-image scan" whatever the file was. The parser
cannot know how a file was produced, so that wording put a guess in front of
the answer model. The result now carries the counts the parser measured
(pages read, raw tables found, pages with a text layer) and its sentence is
built from those counts only.

Every test writes its own PDF; each claim is checked on two different inputs
so a constant cannot pass.
"""
import asyncio
import os
import tempfile

import pytest

pytest.importorskip("pdfplumber")

from app.blocks import boq_processor as bp


def _pdf_bytes(pages):
    """A minimal valid PDF. ``pages`` is a list of lists of text lines; an
    empty list is a page with no text layer at all."""
    objs = []  # body of objects 1..n
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    page_ids = [3 + 2 * i for i in range(n_pages)]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, lines in enumerate(pages):
        content_id = page_ids[i] + 1
        objs.append(
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
             f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
             f"/Contents {content_id} 0 R >>").encode()
        )
        ops = []
        for j, line in enumerate(lines):
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"BT /F1 11 Tf 60 {780 - 16 * j} Td ({safe}) Tj ET")
        stream = "\n".join(ops).encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


@pytest.fixture
def make_pdf():
    made = []

    def make(pages):
        fd, path = tempfile.mkstemp(suffix="_BOQ.pdf")
        with os.fdopen(fd, "wb") as f:
            f.write(_pdf_bytes(pages))
        made.append(path)
        return path

    yield make
    for p in made:
        try:
            os.remove(p)
        except OSError:
            pass


def _parse(path):
    # The in-process page parser: the subject here is the result's wording,
    # not the isolation/budget wrapper (covered elsewhere).
    return bp.BOQProcessorBlock()._parse_pdf_pages(path, {}, float("inf"))


_GUESSES = ("scanned", "scan ", "image", "ocr")


@pytest.mark.parametrize(
    "pages, read, with_text",
    [
        ([["Item 1.1 Excavation m3 120"], ["Item 1.2 Backfill m3 80"], []], 3, 2),
        ([["Bill No. 4 - Drainage"], [], [], ["Item 4.7 Manhole nr 6"], ["Total"]], 5, 3),
    ],
)
def test_text_without_a_table_reports_counts_not_a_guess(make_pdf, pages, read, with_text):
    out = _parse(make_pdf(pages))
    assert out["status"] == "partial"
    assert out["pages_read"] == read
    assert out["tables_seen"] == 0
    assert out["pages_with_text"] == with_text
    assert out["page_count"] == with_text  # unchanged meaning: pages returned
    note = out["note"]
    assert f"read {read} pages" in note
    assert f"{with_text} pages of {read} carry a text layer" in note
    assert "found no table" in note
    assert not any(g in note.lower() for g in _GUESSES), note


@pytest.mark.parametrize("n_pages", [1, 4])
def test_no_text_layer_error_names_the_observation(make_pdf, n_pages):
    out = _parse(make_pdf([[] for _ in range(n_pages)]))
    assert out["status"] == "error"
    assert out["pages_read"] == n_pages
    assert out["pages_with_text"] == 0
    assert out["tables_seen"] == 0
    err = out["error"]
    assert ("read 1 page," if n_pages == 1 else f"read {n_pages} pages,") in err
    assert "no page carries a text layer" in err
    assert not any(g in err.lower() for g in _GUESSES), err


@pytest.mark.parametrize(
    "read, tables, text, expect",
    [
        (7, 2, 7, "read 7 pages, found 2 tables, none with a header row"),
        (1, 1, 1, "read 1 page, found 1 table, none with a header row"),
        (9, 0, 1, "1 page of 9 carries a text layer"),
        (2, 0, 0, "found no table; no page carries a text layer"),
    ],
)
def test_observation_sentence_follows_the_counts(read, tables, text, expect):
    s = bp._no_table_observation(read, tables, text)
    assert expect in s
    assert not any(g in s.lower() for g in _GUESSES), s


@pytest.mark.parametrize("pages", [
    [["Preliminaries"], ["Item 2.3 Formwork m2 340"]],
    [["Section 4"], ["Item 4.1 Blockwork m2 210"], ["Item 4.2 Plaster m2 95"]],
])
def test_block_entrypoint_carries_the_counts(make_pdf, pages):
    """Through the block's public ``process`` (isolation wrapper included)
    the counts survive to the caller."""
    path = make_pdf(pages)
    out = asyncio.run(bp.BOQProcessorBlock().process({"file_path": path}, {}))
    inner = out.get("result", out) if isinstance(out, dict) else out
    found = inner if "pages_read" in inner else next(
        (v for v in inner.values() if isinstance(v, dict) and "pages_read" in v), inner
    )
    assert found.get("pages_read") == len(pages), out
    assert found.get("pages_with_text") == len(pages), out
