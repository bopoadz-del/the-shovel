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
    row = ("d1", "p1", "a.pdf", 10, path, "INDEXED", 3, None, None, None, None)
    out = "\n".join(mod.diagnose(row))
    assert ("not on this filesystem" in out) is (not exists)
    assert ("text_stops_at=2" in out) is exists


@pytest.mark.parametrize("matches, code", [(0, 1), (2, 1), (1, 0)])
def test_main_needs_exactly_one_document_per_id(monkeypatch, tmp_path, matches, code):
    mod = _load()
    path = _pdf(tmp_path, "b.pdf", ["Bill 1 item rate amount " * 4])
    rows = [(f"d{i}", "p1", "b.pdf", 10, path, "INDEXED", 1, None, None, None, None)
            for i in range(matches)]
    seen = []

    class _Res:
        def __init__(self, out):
            self._out = out

        def all(self):
            return self._out

    class _S:
        def execute(self, stmt, params):
            seen.append(params)
            return _Res(rows if "p" in params else [(1, 2, 40)])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    from app.core import db
    monkeypatch.setattr(db, "SessionLocal", lambda: _S())
    monkeypatch.setenv("COVERAGE_DOC_IDS", "abc")
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    assert mod.main() == code
    # the prefix select, then one stored-chunk read per printed document
    assert seen == [{"p": "abc%"}] + ([{"pr": "p1", "d": "d0"}] if matches == 1 else [])


# --- the report is a public log: ids, counts and page numbers only -------------

@pytest.mark.parametrize("value", ["Signed Set", "x"])
def test_main_refuses_the_withdrawn_drive_path_selector_without_echoing_it(monkeypatch, capsys, value):
    mod = _load()
    monkeypatch.setenv("COVERAGE_DRIVE_PATH", value)
    monkeypatch.setenv("COVERAGE_DOC_IDS", "abc")
    assert mod.main() == 2
    captured = capsys.readouterr()
    assert value not in captured.out + captured.err


@pytest.mark.parametrize("n, code", [(21, 2), (20, 1)])
def test_main_caps_the_number_of_documents(monkeypatch, n, code):
    mod = _load()
    seen: list = []

    class _S:
        def execute(self, stmt, params):
            seen.append(params)
            return _NoRows()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    from app.core import db
    monkeypatch.setattr(db, "SessionLocal", lambda: _S())
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("COVERAGE_DOC_IDS", " ".join(f"id{i}" for i in range(n)))
    assert mod.main() == code
    assert len(seen) == (0 if code == 2 else n)


class _NoRows:
    def all(self):
        return []


def _session_with(monkeypatch, doc_rows, chunk_rows):
    class _Res:
        def __init__(self, out):
            self._out = out

        def all(self):
            return self._out

    class _S:
        def execute(self, stmt, params):
            return _Res(doc_rows if "p" in params else chunk_rows)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    from app.core import db
    monkeypatch.setattr(db, "SessionLocal", lambda: _S())


# Each input is a client-looking name, Drive path, project id and free-text
# reason / metadata built here; none may reach the printed report.
@pytest.mark.parametrize("name, folder, project, exists", [
    ("Harbour Tower Vol 3 Drawings.pdf", "Client Alpha/Tender Pack FINAL", "proj-alpha-7731", True),
    ("Riverside_Mall_BOQ_rev2.pdf", "Zeta Holdings/Signed", "proj-zeta-0042", False),
])
def test_rendered_report_carries_no_file_name_path_or_project(
        monkeypatch, tmp_path, capsys, name, folder, project, exists):
    import json

    mod = _load()
    sub = tmp_path / folder
    sub.mkdir(parents=True)
    path = _pdf(sub, name, ["Bill 1 item rate amount " * 4, ""]) if exists else str(sub / name)
    stem = name.rsplit(".", 1)[0]
    meta = {"drive_path": f"{folder}/{name}",
            "extract_failed_detail": f"FileNotFoundError: {path}",
            "ingest_attempts": {"n": 2, "last": stem},
            "ocr_skip_reason": "size"}
    reason = f"pdf_extract_failed|could not open {name}"
    row = ("d0c1d", project, name, 10, path, "EXTRACT_FAILED", 0, 2, None, reason,
           json.dumps(meta))
    _session_with(monkeypatch, [row], [(1, 1, 40)])
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("COVERAGE_DOC_IDS", "d0c1d")
    assert mod.main() == 0
    out = capsys.readouterr().out
    low = out.lower()
    for secret in (name, stem, folder, project, path, str(tmp_path), "Client", "Zeta"):
        assert secret.lower() not in low, secret
    # What it does print: the id, codes and counts.
    assert "== doc d0c1d" in out
    assert "reason: pdf_extract_failed|<text len=" in out
    assert "meta.ocr_skip_reason: size" in out
    assert "meta.ingest_attempts: {n=2} keys=2" in out
    assert ("text_stops_at=2" in out) is exists
    assert ("file not on this filesystem" in out) is (not exists)


@pytest.mark.parametrize("line, kept", [
    ("open failed at /data/u/Bay Bridge Spec.pdf", "open failed at "),
    ("pages=3 chunks=12 span=3", "pages=3 chunks=12 span=3"),
])
def test_withhold_replaces_the_rows_own_name_and_path(line, kept):
    mod = _load()
    row = ("d1", "proj-xyz1", "Bay Bridge Spec.pdf", 1, "/data/u/Bay Bridge Spec.pdf",
           "INDEXED", 1, 1, None, None, None)
    out = mod.withhold([line], row)[0]
    assert "Bay Bridge" not in out
    assert out.startswith(kept)


@pytest.mark.parametrize("reason, shown", [
    ("pdf_text_sparse|terminal", "pdf_text_sparse|terminal"),
    ("ocr_failed|Tesseract choked on Alpha.pdf", "ocr_failed|<text len=29>"),
])
def test_safe_reason_keeps_codes_and_hides_free_text(reason, shown):
    assert _load().safe_reason(reason) == shown


# --- failure evidence ----------------------------------------------------------

@pytest.mark.parametrize("current, phrase", [
    (True, "IS the current extractor"),
    (False, "no longer closes the row"),
])
def test_terminal_meaning_reads_the_attempts_mark(monkeypatch, current, phrase):
    from app.core import ingest_status as st

    mod = _load()
    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "1")
    version = st.EXTRACTOR_VERSION if current else "older-extractor"
    reason = f"pdf_extract_failed|{st.ATTEMPTS_MARK}{version}:1:{st.TERMINAL}"
    meaning = mod.terminal_meaning(reason)
    assert "closed after 1 attempt(s)" in meaning
    assert "INGEST_MAX_ATTEMPTS now 1" in meaning
    assert phrase in meaning


@pytest.mark.parametrize("reason", [None, "pdf_text_sparse|terminal"])
def test_terminal_meaning_is_none_without_a_mark(reason):
    assert _load().terminal_meaning(reason) is None


@pytest.mark.parametrize("meta, shown, hidden", [
    ({"drive_path": "a/b.pdf", "extract_failed_detail": "MemoryError at a/b.pdf", "chunk_pages": [1]},
     ["meta.extract_failed_detail: <text len=22>"], "a/b.pdf"),
    ({"ingest_attempts": {"n": 1}, "ocr_skip_reason": "size", "sheet_names": ["x"]},
     ["meta.ingest_attempts: {n=1} keys=1", "meta.ocr_skip_reason: size"], "sheet_names"),
])
def test_failure_evidence_prints_failure_keys_only(meta, shown, hidden):
    out = "\n".join(_load().failure_evidence("r1", meta))
    assert "reason: r1" in out
    assert "drive_path" not in out
    for s in shown:
        assert s in out
    assert hidden not in out


@pytest.mark.parametrize("cap_mb, gated", [("0.000001", True), ("100000", False)])
def test_diagnose_names_the_whole_pdf_size_gate(tmp_path, monkeypatch, cap_mb, gated):
    monkeypatch.setenv("PDF_MAX_SIZE_MB", cap_mb)
    mod = _load()
    path = _pdf(tmp_path, "vol.pdf", ["Bill 9 item rate amount " * 4, ""])
    row = ("d1", "p1", "vol.pdf", 10, path, "EXTRACT_FAILED", 0, None, None, None, None)
    out = "\n".join(mod.diagnose(row))
    assert ("SIZE GATE:" in out) is gated
    # The text layer is still read and reported either way.
    assert "text_pages=1" in out


@pytest.mark.parametrize("name, is_pdf", [("v.docx", False), ("v.pdf", True)])
def test_diagnose_skips_the_page_read_for_non_pdf(tmp_path, name, is_pdf):
    mod = _load()
    path = _pdf(tmp_path, name, ["Section 1 text " * 6])
    row = ("d1", "p1", name, 10, path, "INDEXED", 1, None, None, None, None)
    out = "\n".join(mod.diagnose(row))
    assert ("not a PDF: page read skipped" in out) is (not is_pdf)
    assert ("pages=1 " in out) is is_pdf


@pytest.mark.parametrize("timeout, mem", [("900", "1024"), ("3600", "2048")])
def test_budgets_follow_the_product_env(monkeypatch, timeout, mem):
    monkeypatch.setenv("DOC_INDEX_FILE_TIMEOUT_S", timeout)
    monkeypatch.setenv("DOC_INDEX_FILE_MEM_MB", mem)
    b = _load().budgets()
    assert b["file_timeout_s"] == float(timeout)
    assert b["file_mem_mb"] == int(mem)
    from app.core import ingest_status as st
    assert b["extractor_version"] == st.EXTRACTOR_VERSION


# --- the index's own page coverage (stored chunks by page locator) --------------

@pytest.mark.parametrize("rows, pif, with_chunk, missing_runs, span_from", [
    # page 3 of 5 carries no chunk: named as a gap, span read from the row
    ([(1, 2, 900), (2, 1, 400), (4, 1, 50), (5, 3, 1200)], 5, 4, [(3, 3)], "pages_in_file"),
    # every page carries a chunk; pages_in_file unknown -> span = top page
    ([(1, 1, 10), (2, 1, 10), (3, 1, 10)], None, 3, [], "highest_chunk_page"),
])
def test_stored_page_coverage_names_pages_without_a_chunk(rows, pif, with_chunk, missing_runs, span_from):
    cov = _load().stored_page_coverage(rows, pif)
    assert cov["pages_with_chunk"] == with_chunk
    assert cov["pages_without_chunk_runs"] == missing_runs
    assert cov["span_from"] == span_from


@pytest.mark.parametrize("pif, span, tail_gap", [
    (8, 8, [(5, 8)]),       # the file is longer than the last chunked page
    (None, 4, []),          # unknown length: no invented tail gap
])
def test_stored_span_uses_the_file_length_when_recorded(pif, span, tail_gap):
    cov = _load().stored_page_coverage([(1, 1, 5), (2, 1, 5), (3, 1, 5), (4, 1, 5)], pif)
    assert cov["span"] == span
    assert cov["pages_without_chunk_runs"] == tail_gap


@pytest.mark.parametrize("rows, no_page, total", [
    ([(None, 4, 300), (1, 1, 20)], 4, 5),
    ([(1, 1, 20), (2, 2, 30)], 0, 3),
])
def test_chunks_without_a_page_locator_are_counted_not_dropped(rows, no_page, total):
    cov = _load().stored_page_coverage(rows, None)
    assert cov["chunks_without_page"] == no_page
    assert cov["chunks"] == total


@pytest.mark.parametrize("n_pages", [9, 400])
def test_stored_by_range_is_bounded_and_sums_to_total(n_pages):
    mod = _load()
    rows = [(p, 1, p) for p in range(1, n_pages + 1)]
    cov = mod.stored_page_coverage(rows, n_pages)
    assert len(cov["by_range"]) <= mod._MAX_RANGE_ROWS
    assert sum(r["chars"] for r in cov["by_range"]) == sum(range(1, n_pages + 1))
    assert sum(r["pages_with_chunk"] for r in cov["by_range"]) == n_pages


@pytest.mark.parametrize("gaps, page, named", [
    # a gap far past the first 40 document-wide runs is still named by its range row
    (set(range(2, 600, 4)), 302, True),
    # the same page carrying a chunk is not named
    (set(range(2, 600, 4)) - {302}, 302, False),
])
def test_each_range_row_names_its_own_pages_without_a_chunk(gaps, page, named):
    mod = _load()
    rows = [(p, 1, 10) for p in range(1, 591) if p not in gaps]
    cov = mod.stored_page_coverage(rows, 590)
    assert len(cov["pages_without_chunk_runs"]) > 40
    row = next(r for r in cov["by_range"] if r["pages"][0] <= page <= r["pages"][1])
    assert ((page, page) in row["pages_without_chunk_runs"]) is named
    line = next(x for x in mod.render_stored(cov)
                if x.startswith(f"   {row['pages'][0]}-{row['pages'][1]} |"))
    assert (str(page) in line.split(" | ")[-1].split(", ")) is named
    assert sum(len(list(range(a, b + 1)))
               for r in cov["by_range"] for a, b in r["pages_without_chunk_runs"]
               ) == cov["pages_without_chunk"]


@pytest.mark.parametrize("n_runs, shown, more", [(3, 3, ""), (20, 12, " (+8 more runs)")])
def test_range_row_runs_are_bounded(n_runs, shown, more):
    mod = _load()
    text = mod._render_runs([(p, p) for p in range(1, 2 * n_runs, 2)], mod._MAX_RUNS_PER_RANGE)
    assert text.endswith(more)
    assert len(text.removesuffix(more).split(", ")) == shown


@pytest.mark.parametrize("namespace, table", [("v2", "chunks_v2"), ("x9", "chunks_x9")])
def test_stored_sql_is_a_select_on_the_namespaced_table_with_a_bound_id(monkeypatch, namespace, table):
    mod = _load()
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", namespace)
    sql = mod._stored_pages_sql(mod.stored_chunk_table())
    assert sql.lstrip().upper().startswith("SELECT")
    assert f"FROM {table} " in sql and "project_id = :pr AND doc_id = :d" in sql
    for verb in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER"):
        assert not re.search(r"\b" + verb + r"\b", sql.upper())


@pytest.mark.parametrize("fail", [True, False])
def test_stored_lines_print_a_failed_read_instead_of_raising(monkeypatch, fail):
    mod = _load()

    class _S:
        def execute(self, stmt, params):
            if fail:
                raise RuntimeError("no such table")
            return _Res2()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Res2:
        def all(self):
            return [(1, 1, 10), (3, 1, 10)]

    from app.core import db
    monkeypatch.setattr(db, "SessionLocal", lambda: _S())
    row = ("d1", "p1", "a.pdf", 10, "/gone", "INDEXED", 2, 3, None, None, None)
    out = "\n".join(mod.stored_lines(row))
    assert ("READ FAILED" in out) is fail
    assert ("pages_without_chunk_runs: 2" in out) is (not fail)


# --- #409 follow-up: a failing id is named by class; only marked lines publish --

class _Boom:
    """A session whose read raises with a message quoting a client path."""

    def __init__(self, exc):
        self._exc = exc

    def execute(self, stmt, params):
        raise self._exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("exc, cls", [
    (RuntimeError("could not read /data/Client Alpha/Harbour Tower.pdf"), "RuntimeError"),
    (KeyError("Riverside_Mall_BOQ_rev2"), "KeyError"),
])
def test_main_loop_names_a_failure_by_class_and_goes_on(monkeypatch, capsys, exc, cls):
    mod = _load()
    from app.core import db
    monkeypatch.setattr(db, "SessionLocal", lambda: _Boom(exc))
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("COVERAGE_DOC_IDS", "abc def")
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert out.count(f"failed ({cls})") == 2        # both ids reported, loop went on
    assert "done: ids=2 failed=2" in out
    for secret in ("Client Alpha", "Harbour", "Riverside"):
        assert secret not in out


@pytest.mark.parametrize("ids, code", [
    ("abc", 1),   # the run path: budgets, a skipped id, the tally (stdout)
    ("", 2),      # the refusal path: no id given (stderr)
])
def test_every_printed_line_carries_the_mark(monkeypatch, capsys, ids, code):
    mod = _load()
    _session_with(monkeypatch, [], [])
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("COVERAGE_DOC_IDS", ids)
    assert mod.main() == code
    cap = capsys.readouterr()
    lines = [ln for ln in (cap.out + cap.err).splitlines() if ln]
    assert lines and all(ln.startswith(mod.MARK) for ln in lines)


def summary_writes_unmarked(run: str, mark: str) -> list[str]:
    """Lines that write to the step summary without selecting the mark."""
    return [ln.strip() for ln in run.splitlines()
            if "GITHUB_STEP_SUMMARY" in ln and f"^{mark}" not in ln]


@pytest.mark.parametrize("run, bad", [
    ("aws logs get-log-events --output text | tee -a \"$GITHUB_STEP_SUMMARY\"", 1),
    ("sed -n 's/^PPC| //p' stream.txt | tee -a \"$GITHUB_STEP_SUMMARY\"", 0),
])
def test_summary_checker_flags_the_whole_stream(run, bad):
    assert len(summary_writes_unmarked(run, "PPC| ")) == bad


_WHOLE_STREAM_WF = """
jobs:
  report:
    steps:
      - run: |
          aws logs get-log-events --log-stream-name s --output text | tee -a "$GITHUB_STEP_SUMMARY"
"""


@pytest.mark.parametrize("source, clean", [
    ("committed", True),        # the real workflow file
    ("whole-stream", False),    # the #409 shape: the whole container stream teed
])
def test_workflow_publishes_only_the_scripts_marked_lines(source, clean):
    import yaml

    mod = _load()
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pdf-page-coverage.yml"
    body = wf.read_text(encoding="utf-8") if source == "committed" else _WHOLE_STREAM_WF
    doc = yaml.safe_load(body)
    run = "\n".join(s["run"] for job in doc["jobs"].values() for s in job["steps"]
                    if isinstance(s.get("run"), str))
    assert "GITHUB_STEP_SUMMARY" in run
    assert (summary_writes_unmarked(run, mod.MARK) == []) is clean
    assert ("--output text | tee" not in run) is clean


@pytest.mark.parametrize("broken", [True, False])
def test_unreadable_budgets_are_named_by_class_and_the_run_goes_on(monkeypatch, capsys, broken):
    mod = _load()
    if broken:
        def _raise():
            raise OSError("/srv/Client Alpha/config")
        monkeypatch.setattr(mod, "budgets", _raise)
    _session_with(monkeypatch, [], [])
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("COVERAGE_DOC_IDS", "abc")
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert ("budgets: unreadable (OSError)" in out) is broken
    assert "Client Alpha" not in out and "done: ids=1 failed=1" in out


# --- pageless-chunk count by document id (one project) ------------------------

def test_pageless_sql_is_one_select_bound_to_the_project():
    sql = _load()._pageless_sql("rag_chunks_x")
    assert sql.lstrip().upper().startswith("SELECT")
    assert ":pr" in sql and "GROUP BY doc_id" in sql
    assert not re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER)\b", sql, re.IGNORECASE)


@pytest.mark.parametrize("rows, n_hit, pageless, top", [
    ([("dA", 0, 5), ("dB", 3, 9), ("dC", 7, 7)], 2, 10, [("dC", 7, 7), ("dB", 3, 9)]),
    ([("dZ", 2, 4), ("dY", 2, 2), ("dX", None, 1)], 2, 4, [("dY", 2, 2), ("dZ", 2, 4)]),
])
def test_pageless_summary_counts_and_orders_by_count_then_id(rows, n_hit, pageless, top):
    summ = _load().pageless_summary(rows)
    assert summ["documents"] == len(rows)
    assert summ["documents_with_pageless"] == n_hit
    assert summ["pageless_chunks"] == pageless
    assert summ["top"] == top


@pytest.mark.parametrize("n, cap, listed, more", [(5, 3, 3, 2), (2, 3, 2, 0)])
def test_pageless_listing_is_capped_and_says_how_many_more(n, cap, listed, more):
    mod = _load()
    summ = mod.pageless_summary([(f"d{i}", i + 1, 10) for i in range(n)], cap=cap)
    lines = mod.render_pageless("anc", summ)
    assert len([ln for ln in lines if re.match(r"^   d\d+ \|", ln)]) == listed
    assert any(f"+{more} more" in ln for ln in lines) == (more > 0)


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


@pytest.mark.parametrize("flag, pageless_read", [("1", True), ("", False)])
def test_main_pageless_mode_reads_the_anchor_project_and_prints_no_project(
    monkeypatch, capsys, flag, pageless_read,
):
    mod = _load()
    pid, seen = "proj-secret-0001", []
    doc = ("d0c1d", pid, "client_file.pdf", 10, "/data/x/client_file.pdf",
           "indexed", 3, 2, 0, None, None)

    class _S:
        def execute(self, stmt, params):
            seen.append(params)
            if "p" in params:
                return _Rows([doc])
            return _Rows([("d0c1d", 2, 3), ("d0c2e", 0, 4)])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    from app.core import db
    monkeypatch.setattr(db, "SessionLocal", lambda: _S())
    monkeypatch.setattr(mod, "report", lambda row: ["per-page report"])
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("COVERAGE_DOC_IDS", "d0c1d")
    monkeypatch.setenv("COVERAGE_PAGELESS_SUMMARY", flag)
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert ({"pr": pid} in seen) == pageless_read
    assert ("pageless_chunks=2" in out) == pageless_read
    assert ("per-page report" in out) != pageless_read
    assert pid not in out and "client_file" not in out


@pytest.mark.parametrize("flag", ["1", ""])
def test_pageless_mode_keeps_the_ids_required_guard(monkeypatch, flag):
    mod = _load()
    monkeypatch.delenv("COVERAGE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("COVERAGE_DOC_IDS", "")
    monkeypatch.setenv("COVERAGE_PAGELESS_SUMMARY", flag)
    assert mod.main() == 2
