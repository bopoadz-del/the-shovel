"""S11: pptx / xlsx / pdf extraction census is queryable and honest."""
from __future__ import annotations

import importlib

import pytest

from app.core import ingest_status as ist
from app.core.ingest_reconcile import (
    OFFICE_KIND_MIME,
    office_extraction_census,
    office_kind,
    render_office_census_markdown,
    tally_office_docs,
)


def test_office_kind_uses_extension_and_mime():
    assert office_kind({"original_name": "deck.pptx"}) == ".pptx"
    assert office_kind({"original_name": "bill.xlsx"}) == ".xlsx"
    assert office_kind({"original_name": "spec.pdf"}) == ".pdf"
    assert office_kind({"original_name": "letter.docx"}) is None
    assert office_kind({
        "original_name": "NOC Tracker",
        "metadata": {
            "mimeType": OFFICE_KIND_MIME[".pptx"],
        },
    }) == ".pptx"


def test_tally_flags_silent_indexed_zero_and_thin_single_chunk():
    rows = [
        {
            "original_name": "rich.pdf",
            "ingest_status": ist.INDEXED,
            "chunk_count": 8,
            "extractor_version": ist.EXTRACTOR_VERSION,
            "has_file": True,
        },
        {
            "original_name": "lie.pdf",
            "ingest_status": ist.INDEXED,
            "chunk_count": 0,
            "extractor_version": None,
            "has_file": True,
        },
        {
            "original_name": "one.pptx",
            "ingest_status": ist.INDEXED,
            "chunk_count": 1,
            "extractor_version": ist.EXTRACTOR_VERSION,
            "has_file": True,
        },
        {
            "original_name": "sparse.xlsx",
            "ingest_status": ist.TEXT_SPARSE,
            "chunk_count": 2,
            "extractor_version": "(old)",
            "has_file": False,
        },
        {
            "original_name": "ghost.xlsx",
            "ingest_status": ist.UNVERIFIED,
            "chunk_count": 0,
            "has_file": False,
        },
        {
            "original_name": "ignore.docx",
            "ingest_status": ist.INDEXED,
            "chunk_count": 0,
            "has_file": False,
        },
    ]
    report = tally_office_docs(rows)
    pdf = report["by_kind"][".pdf"]
    pptx = report["by_kind"][".pptx"]
    xlsx = report["by_kind"][".xlsx"]
    assert report["documents_total"] == 5
    assert pdf["by_status"][ist.INDEXED] == 2
    assert pdf["indexed_zero_chunk"] == 1
    assert report["indexed_zero_chunk"] == 1
    assert pptx["single_chunk"] == 1
    assert pptx["thin"] == 1
    assert pptx["thin_rate"] == 1.0
    assert xlsx["by_status"][ist.TEXT_SPARSE] == 1
    assert xlsx["by_status"][ist.UNVERIFIED] == 1
    assert xlsx["missing_source"] == 2
    assert xlsx["thin"] == 1
    assert pdf["by_extractor_version"][ist.EXTRACTOR_VERSION] == 1
    assert pdf["by_extractor_version"]["(none)"] == 1
    md = render_office_census_markdown(report)
    assert "indexed_zero_chunk=1" in md
    assert ".pptx" in md
    assert "letter.docx" not in md


def _reload_projects(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    pm = importlib.reload(projects)
    pm._initialized = False
    pm.init_db()
    return pm


def test_office_census_from_ledger_and_script(monkeypatch, tmp_path):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("Office", user_id="system")
    proj = pm.list_projects("system")[0]
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"PK")
    live = pm.add_document(
        proj["id"], "deck.pptx", file_path=str(src), size=2,
    )
    pm.stamp_document_index(
        live["id"], chunk_count=3, ingest_status=ist.INDEXED,
    )
    missing = pm.add_document(proj["id"], "gone.xlsx", size=0)
    pm.stamp_document_index(
        missing["id"], chunk_count=0, ingest_status=ist.UNVERIFIED,
        stamp_extractor_version=False,
    )
    report = office_extraction_census(project_id=proj["id"])
    assert report["by_kind"][".pptx"]["documents"] == 1
    assert report["by_kind"][".pptx"]["by_status"][ist.INDEXED] == 1
    assert report["by_kind"][".xlsx"]["missing_source"] == 1
    assert report["by_kind"][".pdf"]["documents"] == 0

    from scripts.office_extraction_census import main

    out = tmp_path / "OFFICE_EXTRACTION_CENSUS.md"
    assert main(["--project", proj["id"], "--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Office extraction census" in text
    assert "gone.xlsx" not in text


def test_pptx_tables_are_extracted(tmp_path):
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "NOC Tracker"
    table = slide.shapes.add_table(
        2, 2,
        pptx.util.Inches(1), pptx.util.Inches(2),
        pptx.util.Inches(6), pptx.util.Inches(2),
    ).table
    table.cell(0, 0).text = "Ref"
    table.cell(0, 1).text = "Status"
    table.cell(1, 0).text = "NOC-014"
    table.cell(1, 1).text = "Approved"
    path = tmp_path / "noc.pptx"
    prs.save(str(path))

    from app.core.doc_index import _extract_pptx

    out = _extract_pptx(str(path))
    assert "NOC Tracker" in out
    assert "NOC-014" in out and "Approved" in out
    assert "Ref | Status" in out or ("Ref" in out and "Status" in out)


def test_corrupt_pptx_names_extract_failed(tmp_path):
    from app.core.doc_index import _extract_with_meta_impl

    path = tmp_path / "not-really.pptx"
    # Zip local-file magic so the quarantine gate keeps it. The package is
    # still not a presentation, which is what this census names.
    path.write_bytes(b"PK\x03\x04this is not a presentation")
    text, meta = _extract_with_meta_impl(str(path), "not-really.pptx")
    assert text == ""
    assert meta.get("extract_failed")
    assert meta.get("extract_failed_detail")


def test_index_corrupt_pptx_stamps_extract_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("PptxFail", user_id="system")
    proj = pm.list_projects("system")[0]
    path = tmp_path / "broken.pptx"
    path.write_bytes(b"PK\x03\x04not a valid package")
    doc = pm.add_document(
        proj["id"], "broken.pptx", file_path=str(path), size=path.stat().st_size,
    )
    from app.core import doc_index
    importlib.reload(doc_index)

    result = doc_index.index_document(proj["id"], doc["id"])
    assert result["error"] == "ZERO_CHUNK"
    assert "PackageNotFoundError" in (result.get("extract_error") or "")
    stamped = pm.get_document(doc["id"])
    assert stamped["ingest_status"] == ist.EXTRACT_FAILED
