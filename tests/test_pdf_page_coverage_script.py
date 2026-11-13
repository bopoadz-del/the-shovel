"""P-B1 step 1: the read-only page-coverage diagnosis script.

Every input is built here: per-page char lists and small PDFs written with
PyMuPDF. No document id or corpus value from production appears.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pdf_page_coverage.py"
_SRC = _SCRIPT.read_text(encoding="utf-8")


def _load():
    spec = importlib.util.spec_from_file_location("pdf_page_coverage", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pdf(tmp_path: Path, name: str, pages: list[str], size=(595, 842)) -> str:
    import fitz
    doc = fitz.open()
    for body in pages:
        page = doc.new_page(width=size[0], height=size[1])
        if body:
            page.insert_text((40, 60), body, fontsize=9)
    out = tmp_path / name
    doc.save(str(out))
    doc.close()
    return str(out)


# --- SQL is read-only -------------------------------------------------------

def test_sql_is_a_single_select_with_a_bound_prefix():
    mod = _load()
    assert mod._DOC_SQL.lstrip().upper().startswith("SELECT")
    assert ":p" in mod._DOC_SQL
    for verb in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"):
        assert not re.search(rf"\b{verb}\b", mod._DOC_SQL.upper())
    # No OCR entry point is reachable from the script.
    assert "_ocr_pdf_page" not in _SRC
    assert "extract_document_text" not in _SRC


# --- id parsing --------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("abc123, def456", ["abc123", "def456"]),
    ("  f00d\nf00d beef-1 ", ["f00d", "beef-1"]),
])
def test_parse_ids_splits_and_dedupes(raw, expected):
    assert _load().parse_ids(raw) == expected


@pytest.mark.parametrize("raw", ["abc%", "x' OR '1'='1"])
def test_parse_ids_rejects_like_wildcards_and_quotes(raw):
    with pytest.raises(ValueError):
        _load().parse_ids(raw)


@pytest.mark.parametrize("raw", ["", " , "])
def test_main_refuses_without_ids(monkeypatch, raw):
    mod = _load()
    monkeypatch.setenv("COVERAGE_DOC_IDS", raw)
    assert mod.main() == 2


# --- coverage ------------------------------------------------------------------

@pytest.mark.parametrize("chars, stops_at, last_text", [
    ([500, 400, 0, 0, 3], 3, 2),          # tail is image-only: text stops at page 3
    ([500, 0, 0, 900, 40], None, 5),      # last page has text: nothing stops
])
def test_text_stops_at_is_the_start_of_the_trailing_image_only_run(chars, stops_at, last_text):
    cov = _load().coverage(chars, threshold=30, ocr_budget=10)
    assert cov["text_stops_at"] == stops_at
    assert cov["last_text_page"] == last_text


@pytest.mark.parametrize("chars, budget, exhausted_at, past", [
    ([0, 100, 0, 0, 100, 0], 2, 4, 2),    # image-only pages 1,3,4,6; the 3rd (page 4) is past a budget of 2
    ([0, 100, 0, 0, 100, 0], 4, None, 0),  # budget covers all four
    ([0, 0, 0], 0, 1, 3),                  # OCR off: the first image-only page is already past it
])
def test_ocr_budget_exhaustion_page(chars, budget, exhausted_at, past):
    cov = _load().coverage(chars, threshold=30, ocr_budget=budget)
    assert cov["ocr_budget_exhausted_at"] == exhausted_at
    assert cov["image_only_past_budget"] == past


@pytest.mark.parametrize("threshold, text_pages, runs", [
    (30, 2, [(2, 3), (5, 5)]),
    (10, 3, [(2, 2), (5, 5)]),
])
def test_threshold_decides_image_only_and_runs(threshold, text_pages, runs):
    cov = _load().coverage([100, 5, 20, 100, 0], threshold=threshold, ocr_budget=0)
    assert cov["text_pages"] == text_pages
    assert cov["image_only_pages"] == 5 - text_pages
    assert cov["image_only_runs"] == runs


@pytest.mark.parametrize("n, step, rows", [(15, 10, 2), (370, 15, 25)])
def test_chars_by_range_is_bounded_and_sums_to_total(n, step, rows):
    mod = _load()
    chars = [i % 7 * 10 for i in range(n)]
    cov = mod.coverage(chars, threshold=30, ocr_budget=0)
    assert mod.range_size(n) == step
    assert len(cov["chars_by_range"]) == rows
    assert sum(r["chars"] for r in cov["chars_by_range"]) == sum(chars) == cov["chars_total"]
    assert cov["chars_by_range"][-1]["pages"][1] == n


# --- reading a real PDF (no OCR) ---------------------------------------------

@pytest.mark.parametrize("bodies, stops_at", [
    (["Bill 1 item rate amount " * 4, "Bill 2 item rate amount " * 4, "", ""], 3),
    (["", "Bill 1 item rate amount " * 4, "", "Bill 3 item rate amount " * 4], None),
])
def test_read_page_chars_counts_text_layer_per_page(tmp_path, bodies, stops_at):
    mod = _load()
    path = _pdf(tmp_path, "doc.pdf", bodies)
    chars, max_side, error = mod.read_page_chars(path)
    assert error is None
    assert len(chars) == len(bodies)
    assert [c >= 30 for c in chars] == [bool(b) for b in bodies]
    assert max_side == 842
    assert mod.coverage(chars, 30, 0)["text_stops_at"] == stops_at


@pytest.mark.parametrize("size, probe_runs", [((595, 842), True), ((2384, 1684), False)])
def test_gates_report_the_table_probe_decision(tmp_path, monkeypatch, size, probe_runs):
    monkeypatch.delenv("PDF_TABLES_MAX_PAGE_PT", raising=False)
    monkeypatch.delenv("PDF_TABLES_MAX_PAGES", raising=False)
    mod = _load()
    path = _pdf(tmp_path, "sheet.pdf", ["Bill 1 item " * 5], size=size)
    gates = mod.ingest_gates(path, "sheet.pdf")
    assert gates["table_probe_runs"] is probe_runs
    assert gates["skipped_too_large"] is False


@pytest.mark.parametrize("exists", [True, False])
def test_diagnose_names_a_missing_file_instead_of_failing(tmp_path, exists):
    mod = _load()
    path = _pdf(tmp_path, "a.pdf", ["Bill 1 item rate amount " * 4, ""]) if exists \
        else str(tmp_path / "gone.pdf")
    row = ("d1", "p1", "a.pdf", 10, path, "INDEXED", 3, None, None)
    out = "\n".join(mod.diagnose(row))
    assert ("not on this filesystem" in out) is (not exists)
    assert ("text_stops_at=2" in out) is exists


@pytest.mark.parametrize("matches, code", [(0, 1), (2, 1), (1, 0)])
def test_main_needs_exactly_one_document_per_id(monkeypatch, tmp_path, matches, code):
    mod = _load()
    path = _pdf(tmp_path, "b.pdf", ["Bill 1 item rate amount " * 4])
    rows = [(f"d{i}", "p1", "b.pdf", 10, path, "INDEXED", 1, None, None) for i in range(matches)]
    seen = []

    class _Res:
        def all(self):
            return rows

    class _S:
        def execute(self, stmt, params):
            seen.append(params)
            return _Res()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    from app.core import db
    monkeypatch.setattr(db, "SessionLocal", lambda: _S())
    monkeypatch.setenv("COVERAGE_DOC_IDS", "abc")
    assert mod.main() == code
    assert seen == [{"p": "abc%"}]
