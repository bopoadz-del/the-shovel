"""The BOQ verified-total fact is a document attribute, never chunk text.

A stored "no verified total" warning sentence is embedded and FTS-indexed like
content: it ranks for every "total"/"bill" ask and the retrieval layer cannot
tell it apart from what the document says. These tests build their own inputs
(a typed priced bill and an image-only PDF) and run the real ingest path.
"""

from __future__ import annotations

import importlib

import pytest

from app.core import file_crypto
from app.core import projects as projects_mod


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    return tmp_path


def _doc_index():
    from app.core import doc_index
    importlib.reload(doc_index)
    from app.core.rag import vector_store as _vs
    _vs.reset_store_cache()
    return doc_index


def _add(pid, tmp_path, name, payload):
    path = str(tmp_path / name)
    file_crypto.write_document(path, payload)
    return projects_mod.add_document(pid, name, file_path=path, size=len(payload))


def _image_only_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(10, 10, 290, 190), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def _no_warning_prose(chunks):
    blob = "\n".join(chunks).lower()
    assert "guard" not in blob, blob[:300]
    assert "do not state" not in blob, blob[:300]
    assert "scanned/image" not in blob, blob[:300]


@pytest.mark.parametrize(
    "rows",
    [
        [("Excavation in trench", 120, "m3", 25, 3000), ("Backfill", 80, "m3", 12, 960)],
        [("Supply 150mm uPVC pipe", 300, "LM", 40, 12000),
         ("Manhole type A", 4, "nr", 2500, 10000),
         ("Bedding", 30, "m3", 50, 1500)],
    ],
)
def test_priced_bill_has_rows_no_guard_chunk_and_attribute(fresh_db, tmp_path, rows):
    doc_index = _doc_index()
    csv = "Description,Quantity,Unit,Rate,Amount\n" + "".join(
        f"{d},{q},{u},{r},{a}\n" for d, q, u, r, a in rows
    )
    pid = projects_mod.create_project("Priced bill")["id"]
    doc = _add(pid, tmp_path, "Bill of Quantities.csv", csv.encode())

    result = doc_index.index_document(pid, doc["id"])

    chunks = result.get("chunks") or []
    line_chunks = [c for c in chunks if c.startswith("BOQ line item")]
    assert len(line_chunks) == len(rows), chunks
    _no_warning_prose(chunks)
    status = doc_index.boq_total_status(projects_mod.get_document(doc["id"]))
    assert status == {
        "verified": False, "priced_rows": len(rows), "reason": "not_cross_checked",
    }


@pytest.mark.parametrize("name", ["Bill of Quantities (Priced).pdf", "Demolition BOQ.pdf"])
def test_image_only_pdf_has_no_rows_no_guard_chunk_and_unverified(fresh_db, tmp_path, name):
    doc_index = _doc_index()
    pid = projects_mod.create_project("Scanned bill")["id"]
    doc = _add(pid, tmp_path, name, _image_only_pdf())

    result = doc_index.index_document(pid, doc["id"])

    chunks = result.get("chunks") or []
    assert not [c for c in chunks if c.startswith("BOQ line item")], chunks
    _no_warning_prose(chunks)
    status = doc_index.boq_total_status(projects_mod.get_document(doc["id"]))
    assert status is not None and status["verified"] is False
    assert status["reason"] in ("parse_failed", "no_priced_rows", "parse_error"), status


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"status": "error", "error": "x"}, {"verified": False, "reason": "parse_failed", "detail": "x"}),
        ({"status": "success", "line_items": []}, {"verified": False, "priced_rows": 0, "reason": "no_priced_rows"}),
        ({"status": "success", "line_items": [{}], "pages_skipped": 3},
         {"verified": False, "priced_rows": 1, "reason": "pages_skipped", "pages_skipped": 3}),
        ({"status": "success", "line_items": [{}, {}]},
         {"verified": False, "priced_rows": 2, "reason": "not_cross_checked"}),
    ],
)
def test_attribute_names_what_the_parser_saw(result, expected):
    doc_index = _doc_index()
    assert doc_index._boq_total_attribute(result) == expected


def test_non_boq_document_gets_no_attribute(fresh_db, tmp_path):
    doc_index = _doc_index()
    pid = projects_mod.create_project("Notes")["id"]
    doc = _add(pid, tmp_path, "site_notes.txt", b"Pour scheduled Tuesday.")
    doc_index.index_document(pid, doc["id"])
    assert doc_index.boq_total_status(projects_mod.get_document(doc["id"])) is None
    assert doc_index.boq_total_status(None) is None
