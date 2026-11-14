"""P-B3: a rendered bill page read through the real OCR engine.

The geometry tests (tests/test_bill_table_geometry.py) feed hand-built word
boxes, and the tesseract adapter is tested with a stubbed ``image_to_data``.
Neither proves that the boxes tesseract really emits for a scanned page
group into the same lines and columns. This file closes that gap end to end:

    text drawn on a white page image
      -> wrapped in an image-only PDF (no text layer, like a scan)
      -> rasterised back with PyMuPDF, as the indexer does for OCR
      -> app.core.bill_table_geometry.ocr_page_words (real tesseract)
      -> extract_bill_rows

Every page is drawn here from cell texts; no document, bill or fixture file.
Each decision (rows read, column order, qty x rate check, a page without a
header continuing the last columns) is exercised with at least two inputs.

Tesseract is installed by every CI test job (``.github/workflows/test.yml``,
"Install system deps for OCR + PDF", which fails the job if apt cannot
install it), so this test runs in CI rather than being skipped there.
"""

from __future__ import annotations

import io

import pytest

from app.core import bill_table_geometry as g

FONT_PX = 30
LINE_PX = 58
MARGIN_TOP = 80
PAGE_W = 2200


def _font():
    from PIL import ImageFont

    return ImageFont.load_default(size=FONT_PX)


def _draw_page(columns, lines, header=True):
    """Draw a bill page. ``columns`` = [(header label, x px)]; each line is a
    list of cell texts (one per column) or a str label at the left margin."""
    from PIL import Image, ImageDraw

    height = MARGIN_TOP * 2 + LINE_PX * (len(lines) + 1)
    img = Image.new("L", (PAGE_W, height), 255)
    draw = ImageDraw.Draw(img)
    font = _font()
    y = MARGIN_TOP
    if header:
        for label, x in columns:
            draw.text((x, y), label, fill=0, font=font)
        y += LINE_PX
    for line in lines:
        if isinstance(line, str):
            draw.text((columns[0][1], y), line, fill=0, font=font)
        else:
            for (_label, x), text in zip(columns, line):
                if text:
                    draw.text((x, y), text, fill=0, font=font)
        y += LINE_PX
    return img


def _scan_and_ocr(images):
    """Image-only PDF from ``images``, rasterised and OCR'd page by page."""
    import fitz
    from PIL import Image

    doc = fitz.open()
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page = doc.new_page(width=img.width * 0.5, height=img.height * 0.5)
        page.insert_image(page.rect, stream=buf.getvalue())
    pdf = fitz.open("pdf", doc.tobytes())
    words = []
    for number, page in enumerate(pdf, start=1):
        assert page.get_text().strip() == ""  # no text layer: a scan
        pix = page.get_pixmap(dpi=144)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        words += g.ocr_page_words(image, number)
    return words


def _read(rows):
    out = []
    for r in rows:
        f = r.fields
        out.append(
            (
                f.get("description", ""),
                g.parse_number(f.get("quantity", "")),
                g.parse_number(f.get("rate", "")),
                g.parse_number(f.get("total", "")),
                r.check,
            )
        )
    return out


COLS_A = [
    ("Item", 60),
    ("Description", 220),
    ("Unit", 1000),
    ("Qty", 1180),
    ("Rate", 1420),
    ("Amount", 1720),
]
COLS_B = [
    ("Ref", 60),
    ("Amount", 220),
    ("Rate", 560),
    ("Quantity", 820),
    ("Unit", 1100),
    ("Description", 1280),
]

MISMATCH = "qty x rate != amount"


@pytest.mark.parametrize(
    "columns, lines, expected",
    [
        (
            COLS_A,
            [
                ["A1", "Supply pipe", "m", "120", "45.50", "5,460.00"],
                ["A2", "Gate valve", "nr", "4", "1,200.00", "4,800.00"],
                ["A3", "Thrust block", "nr", "6", "300.00", "1,900.00"],
            ],
            [
                ("Supply pipe", 120.0, 45.5, 5460.0, ""),
                ("Gate valve", 4.0, 1200.0, 4800.0, ""),
                ("Thrust block", 6.0, 300.0, 1900.0, MISMATCH),
            ],
        ),
        (
            COLS_B,
            [
                ["E1", "2,000.00", "10.00", "200", "m3", "Excavate trench"],
                ["E2", "750.00", "2.50", "300", "m2", "Trim and compact"],
                ["E3", "640.00", "8.00", "90", "m3", "Imported fill"],
                ["E4", "7.00", "7.00", "1", "m", "Marker post"],
            ],
            [
                ("Excavate trench", 200.0, 10.0, 2000.0, ""),
                ("Trim and compact", 300.0, 2.5, 750.0, ""),
                ("Imported fill", 90.0, 8.0, 640.0, MISMATCH),
                ("Marker post", 1.0, 7.0, 7.0, ""),
            ],
        ),
    ],
    ids=["description-first", "figures-first"],
)
def test_rendered_page_reads_rows_through_tesseract(columns, lines, expected):
    words = _scan_and_ocr([_draw_page(columns, lines)])
    rows = g.extract_bill_rows(words)
    assert _read(rows) == expected
    # One-character units ("m") survive the engine as well as figures do.
    unit_at = [label for label, _x in columns].index("Unit")
    assert [r.fields.get("unit") for r in rows] == [line[unit_at] for line in lines]


@pytest.mark.parametrize(
    "columns, first, second, expected",
    [
        (
            COLS_A,
            [["P1", "Manhole cover", "nr", "3", "150.00", "450.00"]],
            [["P2", "Pipe bedding", "m", "40", "12.00", "480.00"]],
            [
                ("Manhole cover", 3.0, 150.0, 450.0, "", 1),
                ("Pipe bedding", 40.0, 12.0, 480.0, "", 2),
            ],
        ),
        (
            COLS_B,
            [["R1", "900.00", "30.00", "30", "m2", "Formwork"]],
            [
                ["R2", "1,500.00", "25.00", "60", "m", "Kerb laying"],
                ["R3", "220.00", "11.00", "20", "nr", "Road gully"],
            ],
            [
                ("Formwork", 30.0, 30.0, 900.0, "", 1),
                ("Kerb laying", 60.0, 25.0, 1500.0, "", 2),
                ("Road gully", 20.0, 11.0, 220.0, "", 2),
            ],
        ),
    ],
    ids=["two-pages-one-row-each", "two-pages-two-rows-on-second"],
)
def test_rendered_page_without_header_continues_columns(columns, first, second, expected):
    words = _scan_and_ocr([_draw_page(columns, first), _draw_page(columns, second, header=False)])
    rows = g.extract_bill_rows(words)
    assert [(*t, r.source_page) for t, r in zip(_read(rows), rows)] == expected


@pytest.mark.parametrize(
    "columns, lines",
    [
        (COLS_A, [["A1", "Supply pipe", "m", "120", "45.50", "5,460.00"]]),
        (COLS_B, [["E1", "2,000.00", "10.00", "200", "m3", "Excavate trench"]]),
    ],
    ids=["layout-a", "layout-b"],
)
def test_rendered_page_without_any_header_gives_no_rows(columns, lines):
    words = _scan_and_ocr([_draw_page(columns, lines, header=False)])
    assert words  # the engine did read the page
    assert g.extract_bill_rows(words) == []
