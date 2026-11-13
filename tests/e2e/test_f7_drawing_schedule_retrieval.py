"""F7 — a drawing's schedule must be RETRIEVABLE, not merely extractable.

This closes the limitation F6 left open and that docs/KNOWN_INCOMPLETE.md recorded:
`_process_drawing` recovered bar schedules, but the RAG ingest path is a
different pipeline. `doc_index._extract_with_meta` pulls the PDF's raw text
layer, so a schedule reached the corpus as a flat run of words with its
row/column structure gone — "how many 16mm bars are on this sheet" was
unanswerable even though the figure was printed on the page.

The distinction this file exists to enforce: proving a helper RETURNS chunks
is not proving those chunks are retrievable. So these go through the real
`index_document` and then ask `retrieve_with_filter` for the value, the same
shape as F3. A test that stopped at the helper would have passed against the
old code path too, since the helper is new either way.
"""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is the drawing reader")

from app.core import doc_index
from app.core.rag import vector_store as vs


TITLE_BLOCK = """PROJECT: SAMPLE WORKS RESIDENTIAL TOWER
DRAWING TITLE: TYPICAL FLOOR SLAB REINFORCEMENT
DRAWING NO: QZ-100-DWG-0001
REV: C2
SCALE: 1:50"""


def _drawing_pdf(path, *, with_schedule=True, with_text=True):
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)
    if with_text:
        for i, line in enumerate(TITLE_BLOCK.splitlines()):
            page.insert_text(fitz.Point(500, 400 + i * 14), line, fontsize=8)
    if with_schedule:
        xs, ys = [40, 170, 300, 430], [140, 175, 210, 245]
        for y in ys:
            page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y))
        for x in xs:
            page.draw_line(fitz.Point(x, ys[0]), fitz.Point(x, ys[-1]))
        cells = [
            ["BAR MARK", "DIAMETER", "QTY"],
            ["B1", "16", "48"],
            ["B2", "20", "24"],
        ]
        for r, row in enumerate(cells):
            for c, value in enumerate(row):
                page.insert_text(fitz.Point(xs[c] + 6, ys[r] + 22), value, fontsize=8)
    doc.save(str(path))
    doc.close()
    return str(path)


# ── the gate, checked against corpus-shaped filenames ─────────────────────
# Invented names that follow the naming conventions seen in real corpora:
# hyphenated document codes (upper and lower case), a plural "DWGs" in a
# free-text name, and an underscore-separated "drawing_" stem.

CORPUS_SHAPED_DRAWINGS = [
    "QZ-SWK-100-0000-ABC-DWG-TM-200-0000005-A.pdf",
    "qz-swk-101-0000-abc-dwg-sg-200-0000076-04.pdf",
    "Sample Works conditional IFC DWGs - rev b.pdf",
    "drawing_qz_0100.pdf",
]
CORPUS_SHAPED_NON_DRAWINGS = [
    "QZ-SWK-100-0000-ABC-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf",
    "QZ-SWK-100-0000-ABC-SPC-IF-000013-B_Specification.pdf",
    "QZ-100-0000-ABC-PLN-DE-000006-A_Post-appointment BIM Execution Plan.pdf",
    "SW-2030-001 - Volume 1 - Conditions of Contract.pdf",
    "XYZ-301 Request for Information.pdf",
]


@pytest.mark.parametrize("name", CORPUS_SHAPED_DRAWINGS)
def test_the_gate_matches_corpus_shaped_drawing_filenames(name):
    """One name per drawing-naming convention seen in the live corpus. A gate
    that is correct in principle and inert on the shapes real documents
    actually use is the same failure as no gate at all."""
    assert doc_index._looks_like_drawing(name, ".pdf"), f"{name} not recognised"


@pytest.mark.parametrize("name", CORPUS_SHAPED_NON_DRAWINGS)
def test_the_gate_ignores_the_other_document_types(name):
    """Same conventions, other document types. Table detection on every PDF would add
    real cost to a bulk re-index, so the gate has to stay narrow."""
    assert not doc_index._looks_like_drawing(name, ".pdf"), f"{name} wrongly matched"


def test_non_pdfs_are_never_opened(tmp_path):
    assert not doc_index._looks_like_drawing("drawing.xlsx", ".xlsx")
    assert not doc_index._looks_like_drawing("drawing.dwg", ".dwg")


# ── the chunks themselves ─────────────────────────────────────────────────

def test_the_schedule_becomes_chunks_carrying_its_values(tmp_path):
    path = _drawing_pdf(tmp_path / "QZ-SWK-100-ABC-DWG-TM-200-0000005-A.pdf")

    chunks = doc_index._drawing_chunks_for_document(
        path, "QZ-SWK-100-ABC-DWG-TM-200-0000005-A.pdf", ".pdf", "p1"
    )

    assert chunks, "a drawing with a ruled bar schedule produced no chunks"
    blob = "\n".join(chunks)
    assert "B1" in blob and "48" in blob, f"schedule values missing:\n{blob}"
    assert "bar" in blob.lower(), "no schedule vocabulary for retrieval to match on"


def test_chunks_name_the_sheet_they_came_from(tmp_path):
    """A project holds many sheets; a bar schedule on one is not the bar
    schedule on another."""
    path = _drawing_pdf(tmp_path / "QZ-100-DWG-0001.pdf")

    chunks = doc_index._drawing_chunks_for_document(
        path, "QZ-100-DWG-0001.pdf", ".pdf", "p1"
    )

    assert any("QZ-100-DWG-0001" in c for c in chunks), (
        f"no chunk attributes itself to the drawing number:\n{chunks}"
    )


@pytest.mark.parametrize("fname", ["blank-DWG.pdf", "QZ-200-DWG-0007.pdf"])
def test_a_sheet_with_no_text_layer_records_an_attribute_not_a_chunk(tmp_path, fname):
    """No schedule and no text layer: the fact goes on the document row as
    counts. A stored warning sentence would be indexed like content."""
    path = _drawing_pdf(tmp_path / fname, with_schedule=False, with_text=False)
    attrs: dict = {}

    chunks = doc_index._drawing_chunks_for_document(path, fname, ".pdf", "p1", attrs)

    assert chunks == []
    assert attrs[doc_index.DRAWING_SCHEDULE_META_KEY] == {
        "read": False,
        "reason": "no_table_grid_no_text_layer",
        "pages_read": 1,
        "pages_with_text": 0,
        "tables_read": 0,
    }


@pytest.mark.parametrize("fname", ["notes-only-DWG.pdf", "QZ-300-DWG-0012.pdf"])
def test_a_readable_sheet_with_no_schedule_records_its_text_layer(tmp_path, fname):
    """Text but no table grid: the attribute says the text layer was there,
    and no chunk claims anything about the sheet."""
    path = _drawing_pdf(tmp_path / fname, with_schedule=False)
    attrs: dict = {}

    chunks = doc_index._drawing_chunks_for_document(path, fname, ".pdf", "p1", attrs)

    assert not any("GUARD" in c for c in chunks), chunks
    attr = attrs[doc_index.DRAWING_SCHEDULE_META_KEY]
    assert attr["read"] is False
    assert attr["reason"] == "no_table_grid_text_layer_present"
    assert (attr["pages_read"], attr["pages_with_text"], attr["tables_read"]) == (1, 1, 0)


@pytest.mark.parametrize("fname", ["sched-DWG.pdf", "QZ-400-DWG-0021.pdf"])
def test_a_sheet_whose_schedule_was_read_counts_its_tables(tmp_path, fname):
    path = _drawing_pdf(tmp_path / fname)
    attrs: dict = {}

    chunks = doc_index._drawing_chunks_for_document(path, fname, ".pdf", "p1", attrs)

    attr = attrs[doc_index.DRAWING_SCHEDULE_META_KEY]
    assert attr["read"] is True and attr["reason"] == "tables_read"
    table_chunks = [c for c in chunks if c.startswith("DRAWING TABLE")]
    assert attr["tables_read"] == len(table_chunks) >= 1


@pytest.mark.parametrize("stored", [
    {"read": False, "reason": "no_table_grid_no_text_layer",
     "pages_read": 3, "pages_with_text": 0, "tables_read": 0},
    {"read": True, "reason": "tables_read",
     "pages_read": 1, "pages_with_text": 1, "tables_read": 2},
])
def test_drawing_schedule_status_reads_the_attribute_back(stored):
    doc = {"metadata": {doc_index.DRAWING_SCHEDULE_META_KEY: stored}}

    assert doc_index.drawing_schedule_status(doc) == stored
    assert doc_index.drawing_schedule_status({"metadata": {}}) is None
    assert doc_index.drawing_schedule_status(None) is None


def test_an_unreadable_file_never_breaks_indexing(tmp_path):
    """The primary text path must survive a drawing parse failure."""
    bad = tmp_path / "corrupt-DWG.pdf"
    bad.write_text("not a pdf", encoding="utf-8")

    assert doc_index._drawing_chunks_for_document(
        str(bad), "corrupt-DWG.pdf", ".pdf", "p1"
    ) == []


# ── the one that proves the point: end to end into retrieval ──────────────

@pytest.fixture
def indexed_drawing(monkeypatch, tmp_path):
    """Ingest a synthetic drawing through the real `index_document` path."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    vs.reset_store_cache()
    from app.core.rag.embeddings import reset_embedder_cache
    reset_embedder_cache()

    from app.core import projects as projects_store
    from app.core import users as users_store

    filename = "QZ-SWK-100-0000-ABC-DWG-TM-200-0000005-A.pdf"
    pdf_path = _drawing_pdf(tmp_path / filename)

    # projects.user_id is a real FK — the system user has to exist first.
    users_store.init_db()
    project = projects_store.create_project(
        name="Sample Works Infra", user_id=users_store.SYSTEM_USER_ID
    )
    document = projects_store.add_document(
        project_id=project["id"],
        original_name=filename,
        file_path=pdf_path,
        size=len(open(pdf_path, "rb").read()),
    )
    result = doc_index.index_document(project["id"], document["id"])
    return project["id"], result


def test_the_schedule_ROW_is_retrievable_not_just_its_loose_words(indexed_drawing):
    """THE assertion this whole change exists for, and it took two attempts
    to write honestly.

    The first version asserted that "B1" and "48" were retrievable. That
    passed with the wiring DISABLED — PyMuPDF's raw text layer already
    contains both, as loose words. It proved nothing.

    The loose value was never the problem. The problem was that nothing tied
    it to anything: "48" was in the corpus with no indication it is the
    QUANTITY, for bar mark B1, of 16mm diameter, on this sheet. So this
    asserts on the ROW as a unit — which only the structured table chunk can
    produce — and on the schedule's classification.
    """
    project_id, result = indexed_drawing
    assert result.get("indexed"), f"the drawing did not index: {result}"

    from app.core.rag.retriever import retrieve_with_filter
    chunks, _ = retrieve_with_filter(
        "bar mark B1 diameter quantity reinforcement bending schedule",
        project_id, k=5,
    )

    assert chunks, "nothing retrieved for a bar-schedule question"
    blob = "\n".join(c.text for c in chunks)

    assert "B1 | 16 | 48" in blob, (
        "the schedule ROW did not survive into retrieval — only its loose "
        f"words did, which is the state this change exists to fix:\n{blob[:800]}"
    )
    assert "bar bending schedule" in blob.lower(), (
        f"the table reached retrieval unclassified:\n{blob[:800]}"
    )


def test_the_sheet_identity_is_retrievable_as_a_title_block(indexed_drawing):
    """Same discipline as above: `QZ-100-DWG-0001` appears in the raw text layer
    too, so asserting on the bare number would pass without the wiring.

    This asserts the TITLE BLOCK chunk — the one that states the number is a
    drawing number and pairs it with the revision and scale, which is what
    lets an answer attribute a schedule to a sheet.
    """
    project_id, _ = indexed_drawing

    from app.core.rag.retriever import retrieve_with_filter
    chunks, _ = retrieve_with_filter(
        "drawing title block sheet identity drawing number revision scale",
        project_id, k=5,
    )

    blob = "\n".join(c.text for c in chunks)
    assert "DRAWING TITLE BLOCK" in blob, (
        f"no title-block chunk reached retrieval:\n{blob[:800]}"
    )
    assert "drawing number: qz-100-dwg-0001" in blob.lower(), (
        f"the sheet identity is not attributed:\n{blob[:800]}"
    )
