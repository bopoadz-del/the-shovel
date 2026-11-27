"""P-B3: an image-only bill gets its own derived spreadsheet row, linked to the
source and indexed through the spreadsheet row-chunk path -- planned first,
executed only against the plan's hash, never inside a web request.

Every input is built here: synthetic OCR word boxes drawn in two column
orders (one with a section split across pages), and in-memory stand-ins for
the documents table. The derived workbook is real and is read back through
``doc_index._spreadsheet_row_chunks``. Each decision has at least two inputs.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from app.core import bill_classify as bc
from app.core import bill_derived as bd
from app.core import bill_table_geometry as g
from app.core import byte_store, doc_index, projects

H = 20.0
CW = 9.0
LINE = 34.0


def _cell(text: str, x: float, y: float, page: int) -> list[g.Word]:
    out, cur = [], x
    for tok in text.split():
        w = CW * len(tok)
        out.append(g.Word(tok, page, cur, y, w, H))
        cur += w + CW
    return out


def _page(columns, lines, page: int) -> list[g.Word]:
    """A page: header line, then section labels (str) and rows (list)."""
    words: list[g.Word] = []
    y = 100.0
    for label, x in columns:
        words += _cell(label, x, y, page)
    y += LINE
    for line in lines:
        if isinstance(line, str):
            words += _cell(line, columns[1][1], y, page)
        else:
            for (_label, x), text in zip(columns, line):
                if text:
                    words += _cell(text, x, y, page)
        y += LINE
    return words


COLS_A = [("Item", 40), ("Description", 120), ("Unit", 600), ("Qty", 700), ("Rate", 820), ("Amount", 960)]
COLS_B = [("Ref", 40), ("Description", 160), ("Amount SAR", 520), ("Unit Rate SAR", 700), ("Quantity", 880), ("Unit", 1000)]


def _row_a(i: int, q: float, r: float, off: float = 1.0) -> list[str]:
    return [f"A{i}", f"Work item {i}", "m", f"{q:g}", f"{r:.2f}", f"{q * r * off:,.2f}"]


def _row_b(i: int, q: float, r: float, off: float = 1.0) -> list[str]:
    return [f"E{i}", f"Earth task {i}", f"{q * r * off:,.2f}", f"{r:.2f}", f"{q:g}", "m3"]


def _bill_a() -> tuple[list[g.Word], dict[str, tuple[int, float]], int]:
    """Two sections in COLS_A order; the second runs across pages 2 and 3.
    One row's amount is wrong (kept and flagged)."""
    s1 = [_row_a(i, 10 + i, 5 + i) for i in range(4)]
    s2p2 = [_row_a(10 + i, 3 + i, 12 + i) for i in range(3)]
    s2p3 = [_row_a(20 + i, 7 + i, 9 + i, off=1.10 if i == 0 else 1.0) for i in range(2)]
    words = (
        _page(COLS_A, ["Section One Preliminaries", *s1], page=1)
        + _page(COLS_A, ["Section Two Concrete", *s2p2], page=2)
        + _page(COLS_A, s2p3, page=3)
    )
    def total(rows):
        return sum(g.parse_number(r[5]) for r in rows)
    want = {
        "Section One Preliminaries": (4, total(s1)),
        "Section Two Concrete": (5, total(s2p2) + total(s2p3)),
    }
    return words, want, 1


def _bill_b() -> tuple[list[g.Word], dict[str, tuple[int, float]], int]:
    """One section in COLS_B order (amount before rate before quantity)."""
    s = [_row_b(i, 200 + i, 2.5 + i) for i in range(6)]
    words = _page(COLS_B, ["Earthworks Bill", *s], page=1)
    return words, {"Earthworks Bill": (6, sum(g.parse_number(r[2]) for r in s))}, 0


class _Docs:
    """In-memory stand-in for the documents table (only what the module uses)."""

    def __init__(self, source: dict[str, Any], others: list[dict[str, Any]] | None = None,
                 pages: int = 3):
        self.rows = {source["id"]: source, **{o["id"]: o for o in others or []}}
        self.added: list[dict[str, Any]] = []
        self.meta_updates: list[tuple[str, dict]] = []
        self.indexed: list[tuple[tuple, dict]] = []
        self.pages = pages

    def install(self, monkeypatch, tmp_path, *, real_bytes: bool = False) -> None:
        monkeypatch.setattr(projects, "get_document", lambda i: self.rows.get(i))
        monkeypatch.setattr(projects, "list_documents", lambda pid, **k: list(self.rows.values()))
        monkeypatch.setattr(projects, "add_document", self._add)
        monkeypatch.setattr(
            projects, "update_document_metadata",
            lambda i, m: self.meta_updates.append((i, m)),
        )
        if not real_bytes:
            monkeypatch.setattr(projects, "materialize_document_path", lambda d: d.get("file_path"))
        monkeypatch.setattr(byte_store, "allocate_path", lambda name: str(tmp_path / name))
        def write(path, data):
            with open(path, "wb") as fh:
                fh.write(data)
        monkeypatch.setattr(byte_store, "write_bytes", write)
        monkeypatch.setattr(bd, "_page_count", lambda path: self.pages)

        def fake_index(*a, **k):
            self.indexed.append((a, k))
            return {"status": "ok", "total_chunks": 0}
        monkeypatch.setattr(doc_index, "index_document", fake_index)

    def _add(self, project_id, original_name, **kw):
        row = {"id": f"d{len(self.rows)}", "project_id": project_id,
               "original_name": original_name, **kw}
        self.rows[row["id"]] = row
        self.added.append(row)
        return row


def _source(i: str = "src1", name: str = "scan.pdf") -> dict[str, Any]:
    return {"id": i, "original_name": name, "file_path": f"/store/{i}.pdf", "metadata": {}}


def _BILL(path: str) -> bc.BillVerdict:
    return bc.BillVerdict(True, bc.BILL_PRICED)


def _by_page(words: list[g.Word], seen: list[int] | None = None,
             source: str = bc.SOURCE_OCR):
    """A page_source stand-in that returns only the asked page's boxes."""
    def page_source(path, indices):
        (index,) = indices
        if seen is not None:
            seen.append(index)
        page = tuple(w for w in words if w.page == index + 1)
        return [bc.PageRead(index, source, page)]
    return page_source


def _plan_then_derive(words, **kw) -> tuple[dict, dict]:
    plan = bd.plan_bill_sheet("p1", "src1", classify=_BILL, page_source=_by_page(words), **kw)
    out = bd.derive_bill_sheet(
        "p1", "src1", plan_hash=plan["plan_hash"], classify=_BILL,
        page_source=_by_page(words), **kw,
    )
    return plan, out


def _amount_sum(chunks: list[str]) -> float:
    total = 0.0
    for c in chunks:
        for part in c.split("; "):
            if part.startswith("Amount:"):
                total += g.parse_number(part.split(":", 1)[1].strip()) or 0.0
    return total


@pytest.mark.parametrize("build", [_bill_a, _bill_b])
def test_bill_gets_derived_row_whose_sheets_match_the_drawn_tables(tmp_path, monkeypatch, build):
    import openpyxl

    words, want, want_flagged = build()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)

    plan, out = _plan_then_derive(words)

    assert out["derived"] is True
    assert out["rows"] == sum(n for n, _ in want.values())
    assert out["rows_flagged"] == want_flagged
    assert out["sheets"] == len(want)
    (row,) = docs.added
    assert row["metadata"]["derived_from"] == "src1"
    assert row["metadata"]["derivation"] == bd.DERIVATION
    assert row["metadata"]["plan_hash"] == plan["plan_hash"]
    assert row["reingest_of"] is None
    assert row["original_name"].endswith(".xlsx")
    # The source PDF row is linked to the derived row, not hidden.
    assert docs.meta_updates == [("src1", {"derived_document": row["id"]})]
    assert docs.indexed == [(("p1", row["id"]), {"profile": doc_index.PROFILE_CORPUS_BUILD})]

    # The stored workbook: one sheet per section, row counts and sums as drawn.
    wb = openpyxl.load_workbook(row["file_path"], data_only=True)
    assert len(wb.sheetnames) == len(want)
    for ws in wb.worksheets:
        values = [list(r) for r in ws.iter_rows(values_only=True)]
        section = values[0][0]
        header = values[1]
        amount_col = header.index("Amount")
        body = values[2:]
        n, total = want[section]
        assert len(body) == n
        assert sum(g.parse_number(str(r[amount_col])) for r in body) == pytest.approx(total)
        assert "Source Page" in header and "Check" in header

    # Read back through the existing spreadsheet row-chunk path.
    chunks = doc_index._spreadsheet_row_chunks(row["file_path"], row["original_name"], ".xlsx")
    assert len(chunks) == out["rows"]
    assert _amount_sum(chunks) == pytest.approx(sum(t for _, t in want.values()))
    for section in want:
        assert sum(f"Bill: {section}" in c for c in chunks) == want[section][0]


# -- plan before write --------------------------------------------------------


@pytest.mark.parametrize("build, want_sections", [
    (_bill_a, [("Section One Preliminaries", 4, 0, [1, 1]),
               ("Section Two Concrete", 5, 1, [2, 3])]),
    (_bill_b, [("Earthworks Bill", 6, 0, [1, 1])]),
])
def test_plan_reports_sections_and_writes_nothing(tmp_path, monkeypatch, build, want_sections):
    words, _, want_flagged = build()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)

    plan = bd.plan_bill_sheet("p1", "src1", classify=_BILL, page_source=_by_page(words))

    assert plan["derived"] is False
    assert plan["pages"] == [0, 3] and plan["pages_in_file"] == 3
    assert plan["unreconciled"] == want_flagged
    got = [(s["path"][-1], s["rows"], s["unreconciled"], s["pages"]) for s in plan["sections"]]
    assert got == want_sections
    assert plan["rows"] == sum(n for _, n, _, _ in want_sections)
    assert len(plan["plan_hash"]) == 64
    assert docs.added == [] and docs.indexed == [] and docs.meta_updates == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad_hash", ["0" * 64, "stale"])
def test_execute_refuses_a_hash_that_is_not_this_plan(tmp_path, monkeypatch, bad_hash):
    words, _, _ = _bill_a()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    plan = bd.plan_bill_sheet("p1", "src1", classify=_BILL, page_source=_by_page(words))

    out = bd.derive_bill_sheet(
        "p1", "src1", plan_hash=bad_hash, classify=_BILL, page_source=_by_page(words),
    )

    assert out == {"derived": False, "reason": bd.SKIP_PLAN_MISMATCH,
                   "plan_hash": plan["plan_hash"]}
    assert docs.added == [] and docs.indexed == []


def _drop_last_page(words):
    return [x for x in words if x.page != 3]


def _reread_a_ref(words):
    return [g.Word("9" + x.text, x.page, x.left, x.top, x.width, x.height)
            if x.text.startswith("A2") else x for x in words]


@pytest.mark.parametrize("rows_changed", [_drop_last_page, _reread_a_ref])
def test_execute_refuses_when_the_rows_moved_after_the_plan(tmp_path, monkeypatch, rows_changed):
    words, _, _ = _bill_a()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    plan = bd.plan_bill_sheet("p1", "src1", classify=_BILL, page_source=_by_page(words))

    out = bd.derive_bill_sheet(
        "p1", "src1", plan_hash=plan["plan_hash"], classify=_BILL,
        page_source=_by_page(rows_changed(words)),
    )
    assert out["reason"] == bd.SKIP_PLAN_MISMATCH and docs.added == []


@pytest.mark.parametrize("pages", [(0, 2), (1, 3)])
def test_partial_plan_counts_its_range_and_cannot_execute(tmp_path, monkeypatch, pages):
    words, _, _ = _bill_a()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    seen: list[int] = []

    plan = bd.plan_bill_sheet("p1", "src1", pages=pages, classify=_BILL,
                              page_source=_by_page(words, seen))
    assert seen == list(range(*pages))
    assert plan["pages"] == list(pages)
    in_range = set(range(pages[0] + 1, pages[1] + 1))
    assert {p for s in plan["sections"] for p in s["pages"]} <= in_range

    full = bd.plan_bill_sheet("p1", "src1", classify=_BILL, page_source=_by_page(words))
    assert plan["plan_hash"] != full["plan_hash"]
    out = bd.derive_bill_sheet("p1", "src1", plan_hash=plan["plan_hash"],
                               classify=_BILL, page_source=_by_page(words))
    assert out["reason"] == bd.SKIP_PLAN_MISMATCH and docs.added == []


# -- page by page, resumable --------------------------------------------------


@pytest.mark.parametrize("first_range", [(0, 1), (0, 2)])
def test_work_dir_resumes_and_ocrs_each_page_once(tmp_path, monkeypatch, first_range):
    words, _, _ = _bill_a()
    store = tmp_path / "store"
    store.mkdir()
    docs = _Docs(_source())
    docs.install(monkeypatch, store)
    work = str(tmp_path / "work")
    seen: list[int] = []

    bd.plan_bill_sheet("p1", "src1", pages=first_range, classify=_BILL,
                       page_source=_by_page(words, seen), work_dir=work)
    whole = bd.plan_bill_sheet("p1", "src1", classify=_BILL,
                               page_source=_by_page(words, seen), work_dir=work)
    assert seen == [0, 1, 2]  # each page OCR'd once across both runs

    def no_ocr(path, indices):
        raise AssertionError("re-OCR'd a page whose boxes were kept")
    out = bd.derive_bill_sheet("p1", "src1", plan_hash=whole["plan_hash"],
                               classify=_BILL, page_source=no_ocr, work_dir=work)
    assert out["derived"] is True and out["rows"] == whole["rows"]


@pytest.mark.parametrize("pages", [1, 3])
def test_pages_are_ocrd_one_at_a_time(tmp_path, monkeypatch, pages):
    docs = _Docs(_source(), pages=pages)
    docs.install(monkeypatch, tmp_path)
    calls: list[list[int]] = []

    def page_source(path, indices):
        calls.append(list(indices))
        return [bc.PageRead(i, bc.SOURCE_OCR, ()) for i in indices]
    bd.plan_bill_sheet("p1", "src1", classify=_BILL, page_source=page_source)
    assert calls == [[i] for i in range(pages)]


# -- never in a web request ---------------------------------------------------


@pytest.mark.parametrize("step", ["plan", "derive"])
def test_refused_inside_a_web_request(tmp_path, monkeypatch, step):
    from app.infra import monitoring

    words, _, _ = _bill_b()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)

    def no_classify(path):
        raise AssertionError("classified inside a web request")
    token = monitoring.web_request_ctx.set(True)
    try:
        with pytest.raises(bd.WebRequestRefused):
            if step == "plan":
                bd.plan_bill_sheet("p1", "src1", classify=no_classify)
            else:
                bd.derive_bill_sheet("p1", "src1", plan_hash="x", classify=no_classify)
    finally:
        monitoring.web_request_ctx.reset(token)
    assert docs.added == []
    # The same call outside a request runs.
    plan = bd.plan_bill_sheet("p1", "src1", classify=_BILL, page_source=_by_page(words))
    assert plan["rows"] == 6


@pytest.mark.parametrize("path", ["/livez", "/v1/admin/debug/project-reindex"])
def test_middleware_marks_the_request_and_clears_it(path):
    import asyncio

    from starlette.requests import Request
    from starlette.responses import Response

    from app.infra import monitoring

    seen: list[bool] = []

    async def call_next(request):
        seen.append(monitoring.in_web_request())
        return Response("ok")

    scope = {"type": "http", "method": "GET", "path": path, "headers": [],
             "query_string": b"", "server": ("t", 80), "scheme": "http"}
    assert monitoring.in_web_request() is False
    asyncio.run(monitoring.observability_middleware(Request(scope), call_next))
    assert seen == [True]
    assert monitoring.in_web_request() is False


@pytest.mark.parametrize("exc", [RuntimeError("handler died"), ValueError("bad input")])
def test_middleware_reraises_and_still_clears_the_marker(exc):
    import asyncio

    from starlette.requests import Request

    from app.infra import monitoring

    seen: list[bool] = []

    async def call_next(request):
        seen.append(monitoring.in_web_request())
        raise exc

    scope = {"type": "http", "method": "POST", "path": "/v1/x", "headers": [],
             "query_string": b"", "server": ("t", 80), "scheme": "http"}
    with pytest.raises(type(exc), match=str(exc)):
        asyncio.run(monitoring.observability_middleware(Request(scope), call_next))
    assert seen == [True]
    assert monitoring.in_web_request() is False


# -- bytes --------------------------------------------------------------------


@pytest.mark.parametrize("where", ["local", "drive"])
def test_bytes_come_from_local_file_or_drive_pointer(tmp_path, monkeypatch, where):
    payload = b"%PDF-1.4 drawn " + where.encode()
    src = {"id": "src1", "original_name": "scan.pdf", "size": len(payload), "metadata": {}}
    if where == "local":
        local = tmp_path / "local.pdf"
        local.write_bytes(payload)
        src["file_path"] = str(local)
    else:
        src["file_path"] = None
        src["metadata"] = {"drive_file_id": "drv1"}
    docs = _Docs(src)
    docs.install(monkeypatch, tmp_path, real_bytes=True)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    fetched: list[str] = []

    def fake_fetch(doc):
        fetched.append(doc["id"])
        return payload, "drive", None
    monkeypatch.setattr(projects, "_fetch_remote_document_bytes", fake_fetch)
    read: list[bytes] = []

    def page_source(path, indices):
        with open(path, "rb") as fh:
            read.append(fh.read())
        return [bc.PageRead(i, bc.SOURCE_OCR, ()) for i in indices]

    out = bd.plan_bill_sheet("p1", "src1", page_source=page_source)
    assert out["derived"] is False and out["reason"] == bc.NOT_BILL_NO_HEADER
    assert read == [payload] * docs.pages
    assert fetched == ([] if where == "local" else ["src1"])


@pytest.mark.parametrize("fetch", [
    lambda doc: (None, None, "missing"),
    lambda doc: (b"", "drive", None),
])
def test_no_readable_bytes_only_when_nothing_materialises(tmp_path, monkeypatch, fetch):
    src = {"id": "src1", "original_name": "scan.pdf", "file_path": None, "size": 5,
           "metadata": {"drive_file_id": "drv1"}}
    docs = _Docs(src)
    docs.install(monkeypatch, tmp_path, real_bytes=True)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(projects, "_fetch_remote_document_bytes", fetch)

    def no_classify(path):
        raise AssertionError("classified without bytes")
    out = bd.plan_bill_sheet("p1", "src1", classify=no_classify)
    assert out == {"derived": False, "reason": bd.SKIP_NO_BYTES}


# -- skips --------------------------------------------------------------------


@pytest.mark.parametrize("verdict", [
    bc.BillVerdict(False, bc.NOT_BILL_FEW_ROWS),
    bc.BillVerdict(False, bc.NOT_BILL_NO_HEADER),
])
def test_not_a_bill_gets_no_derived_row(tmp_path, monkeypatch, verdict):
    words, _, _ = _bill_b()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    judged: list[list[int]] = []

    def classify(reads):
        judged.append([r.index for r in reads])
        return verdict

    out = bd.derive_bill_sheet("p1", "src1", plan_hash="x",
                               classify=classify, page_source=_by_page(words))

    # The verdict is taken over every page read; a refusal reports those
    # pages' sources and writes nothing.
    assert judged == [list(range(docs.pages))]
    assert out["derived"] is False and out["reason"] == verdict.reason
    assert out["pages_without_text_layer"] == docs.pages
    assert docs.added == [] and docs.indexed == [] and docs.meta_updates == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("source, reason", [
    (_source(name="sheet.xlsx"), bd.SKIP_NOT_PDF),
    ({**_source(), "metadata": {"derived_from": "other"}}, bd.SKIP_IS_DERIVED),
    ({**_source(), "file_path": None}, bd.SKIP_NO_BYTES),
])
def test_skips_before_classifying(tmp_path, monkeypatch, source, reason):
    docs = _Docs(source)
    docs.install(monkeypatch, tmp_path)

    def no_classify(path):
        raise AssertionError("classified a document that cannot have a derived sheet")

    out = bd.plan_bill_sheet("p1", "src1", classify=no_classify)
    assert out == {"derived": False, "reason": reason}
    assert docs.added == []


@pytest.mark.parametrize("words", [
    [],
    _page(COLS_A, ["Only a section label"], page=1),
])
def test_bill_with_no_extractable_rows_gets_no_row(tmp_path, monkeypatch, words):
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    out = bd.derive_bill_sheet("p1", "src1", plan_hash="x", classify=_BILL,
                               page_source=_by_page(words))
    assert out["derived"] is False and out["reason"] == bd.SKIP_NO_ROWS
    assert docs.added == []


@pytest.mark.parametrize("prior_visible, want_reingest", [(True, "old9"), (False, None)])
def test_rerun_supersedes_the_visible_previous_derived_row(
    tmp_path, monkeypatch, prior_visible, want_reingest,
):
    words, _, _ = _bill_b()
    prior = {"id": "old9", "original_name": "scan.bill.xlsx", "file_path": "x",
             "metadata": {"derived_from": "src1"}, "retrieval_visible": prior_visible}
    unrelated = {"id": "u1", "original_name": "o.bill.xlsx", "file_path": "y",
                 "metadata": {"derived_from": "another"}, "retrieval_visible": True}
    docs = _Docs(_source(), [prior, unrelated])
    docs.install(monkeypatch, tmp_path)

    _plan, out = _plan_then_derive(words)

    assert docs.added[0]["reingest_of"] == want_reingest
    assert out["superseded"] == want_reingest


@pytest.mark.parametrize("profile", [doc_index.PROFILE_CORPUS_BUILD, doc_index.PROFILE_USER_UPLOAD])
def test_index_document_never_derives(monkeypatch, profile):
    monkeypatch.setattr(
        doc_index, "_index_document",
        lambda *a, **k: {"status": "ok", "filename": "scan.pdf", "total_chunks": 4},
    )
    monkeypatch.setattr(doc_index, "_record_pages_without_text", lambda *a: None)

    def boom(*a, **k):
        raise AssertionError("index_document ran the derivation")
    monkeypatch.setattr(bd, "plan_bill_sheet", boom)
    monkeypatch.setattr(bd, "derive_bill_sheet", boom)
    result = doc_index.index_document("p1", "d1", profile=profile)
    assert result["total_chunks"] == 4 and "derived_bill" not in result


@pytest.mark.parametrize("source_name, want_name", [
    ("nested/dir/scan.pdf", "scan.bill.xlsx"),
    ("a/b/../Vol 2.PDF", "Vol 2.bill.xlsx"),
])
def test_derived_name_stays_inside_the_store(tmp_path, monkeypatch, source_name, want_name):
    words, _, _ = _bill_b()
    docs = _Docs(_source(name=source_name))
    docs.install(monkeypatch, tmp_path)
    _plan_then_derive(words)
    (row,) = docs.added
    assert os.path.dirname(row["file_path"]) == str(tmp_path)
    assert row["original_name"] == want_name


@pytest.mark.parametrize("pages", [1, 4])
def test_page_count_reads_the_pdf(tmp_path, pages):
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    path = tmp_path / f"drawn_{pages}.pdf"
    doc.save(str(path))
    doc.close()
    assert bd._page_count(str(path)) == pages


# -- rows from a model's read of the pages (bill_page_read) -------------------


def _blank_pages():
    """A page_source with no words: the rows come from the model reader only."""
    def page_source(path, indices):
        (index,) = indices
        return [bc.PageRead(index, bc.SOURCE_OCR, ())]
    return page_source


def _line(ref: str, q: float, r: float, amount: float | None = None) -> dict[str, Any]:
    return {"type": "item", "ref": ref, "description": f"work {ref}", "unit": "m3",
            "quantity": q, "rate": r, "amount": q * r if amount is None else amount}


_READS_A = {
    0: [{"type": "heading", "description": "Part one"}, _line("A", 10, 5), _line("B", 3, 7)],
    1: [_line("C", 2, 50), {"type": "collection", "description": "to collection",
                            "amount": 50 + 21 + 100}],
    2: [{"type": "heading", "description": "Part two"}, _line("D", 4, 4, amount=17)],
}
_READS_B = {
    0: [_line("1", 1, 1000)],
    1: [{"type": "heading", "description": "Sub"}, _line("2", 6, 6), _line("3", 8, 0.5)],
    2: [_line("4", 9, 9)],
}


def _model_reader(reads: dict[int, Any], seen: list[int] | None = None):
    from app.core import bill_page_read as bpr

    def read_page(page_read):
        if seen is not None:
            seen.append(page_read.index)
        return reads.get(page_read.index)
    return bpr.rows_of_model_reads(read_page)


@pytest.mark.parametrize("reads, want_rows, want_flagged", [
    (_READS_A, 5, 1),
    (_READS_B, 4, 0),
])
def test_model_reads_become_the_derived_copy(tmp_path, monkeypatch, reads, want_rows,
                                             want_flagged):
    import openpyxl

    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    seen: list[int] = []
    kw = {"classify": _BILL, "page_source": _blank_pages()}
    plan = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(reads, seen), **kw)
    assert plan["rows"] == want_rows
    assert plan["unreconciled"] == want_flagged
    assert plan["pages_not_read"] == []
    assert docs.added == []
    out = bd.derive_bill_sheet("p1", "src1", plan_hash=plan["plan_hash"],
                               rows_of_reads=_model_reader(reads, seen), **kw)
    assert out["derived"] is True
    assert (out["rows"], out["rows_flagged"]) == (want_rows, want_flagged)
    assert sorted(seen) == [0, 0, 1, 1, 2, 2]  # each page read once per step
    (row,) = docs.added
    assert row["metadata"]["derived_from"] == "src1"

    want_amounts = [ln["amount"] for p in sorted(reads) for ln in reads[p]
                    if ln["type"] != "heading"]
    wb = openpyxl.load_workbook(row["file_path"], data_only=True)
    cells = [c for ws in wb.worksheets for r in ws.iter_rows(values_only=True) for c in r]
    assert cells.count("item") == sum(ln["type"] == "item" for p in reads for ln in reads[p])
    chunks = doc_index._spreadsheet_row_chunks(row["file_path"], row["original_name"], ".xlsx")
    assert len(chunks) == want_rows
    assert _amount_sum(chunks) == pytest.approx(sum(want_amounts))


@pytest.mark.parametrize("unread_page, bad_read", [(1, None), (2, "not json lines")])
def test_page_not_read_is_planned_and_never_written(tmp_path, monkeypatch, unread_page,
                                                    bad_read):
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    reads = {**_READS_B, unread_page: bad_read}
    kw = {"classify": _BILL, "page_source": _blank_pages()}
    plan = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(reads), **kw)
    assert plan["pages_not_read"] == [unread_page + 1]
    assert plan["rows"] == 4 - sum(ln["type"] != "heading" for ln in _READS_B[unread_page])
    out = bd.derive_bill_sheet("p1", "src1", plan_hash=plan["plan_hash"],
                               rows_of_reads=_model_reader(reads), **kw)
    assert out == {"derived": False, "reason": bd.SKIP_PAGES_NOT_READ,
                   "pages_not_read": [unread_page + 1]}
    assert docs.added == [] and docs.indexed == []


@pytest.mark.parametrize("unread_page", [0, 2])
def test_plan_hash_changes_when_a_page_not_read_is_read_later(tmp_path, monkeypatch,
                                                               unread_page):
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    kw = {"classify": _BILL, "page_source": _blank_pages()}
    partial = {**_READS_A, unread_page: None}
    plan = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(partial), **kw)
    full = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(_READS_A), **kw)
    assert plan["plan_hash"] != full["plan_hash"]
    out = bd.derive_bill_sheet("p1", "src1", plan_hash=plan["plan_hash"],
                               rows_of_reads=_model_reader(_READS_A), **kw)
    assert out["reason"] == bd.SKIP_PLAN_MISMATCH
    assert docs.added == []


@pytest.mark.parametrize("build", [_bill_a, _bill_b])
def test_geometry_reader_reports_no_page_not_read(tmp_path, monkeypatch, build):
    words, _want, _flagged = build()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    plan, out = _plan_then_derive(words)
    assert plan["pages_not_read"] == []
    assert out["derived"] is True


@pytest.mark.parametrize("bad_read", [None, []])
def test_page_not_read_is_in_the_plan_hash_even_when_the_rows_are_the_same(
        tmp_path, monkeypatch, bad_read):
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    kw = {"classify": _BILL, "page_source": _blank_pages()}
    unread = {**_READS_B, 2: bad_read}
    read_no_rows = {**_READS_B, 2: [{"type": "heading", "description": "end"}]}
    a = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(unread), **kw)
    b = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(read_no_rows), **kw)
    assert a["rows"] == b["rows"] == 3
    assert (a["pages_not_read"], b["pages_not_read"]) == ([3], [])
    assert a["plan_hash"] != b["plan_hash"]


@pytest.mark.parametrize("printed, want_state", [(171, "RECONCILED"), (170, "FAILED")])
def test_bill_states_are_planned_hashed_and_stored_with_the_copy(tmp_path, monkeypatch,
                                                                printed, want_state):
    import functools

    from app.core import bill_page_read as bpr

    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    reads = {0: [_line("A", 10, 5), _line("B", 3, 7)], 1: [_line("C", 2, 50)],
             2: [_line("D", 0, 9)]}
    kw = {"classify": _BILL, "page_source": _blank_pages()}
    states = functools.partial(bpr.bill_states, tolerance=0,
                               spans=[bpr.BillSpan("all", 1, 3, printed)])
    plan = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(reads),
                              states_of_rows=states, **kw)
    (bill,) = plan["bill_states"]["bills"]
    assert (bill["state"], bill["items_sum"]) == (want_state, 171)
    bare = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(reads), **kw)
    assert bare["bill_states"] is None
    assert bare["plan_hash"] != plan["plan_hash"]  # the states are part of the plan
    out = bd.derive_bill_sheet("p1", "src1", plan_hash=plan["plan_hash"],
                               rows_of_reads=_model_reader(reads), states_of_rows=states, **kw)
    assert out["derived"] is True
    (row,) = docs.added
    assert row["metadata"]["bill_states"]["bills"][0]["state"] == want_state


@pytest.mark.parametrize("reads", [_READS_A, _READS_B])
def test_copy_without_states_stores_no_states(tmp_path, monkeypatch, reads):
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    kw = {"classify": _BILL, "page_source": _blank_pages()}
    plan = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(reads), **kw)
    bd.derive_bill_sheet("p1", "src1", plan_hash=plan["plan_hash"],
                         rows_of_reads=_model_reader(reads), **kw)
    (row,) = docs.added
    assert "bill_states" not in row["metadata"]


@pytest.mark.parametrize("unread_index, want", [
    (1, ["RECONCILED", "NOT READ"]),
    (0, ["NOT READ", "RECONCILED"]),
])
def test_planned_states_see_the_pages_not_read(tmp_path, monkeypatch, unread_index, want):
    import functools

    from app.core import bill_page_read as bpr

    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    reads = {0: [_line("A", 10, 5)], 1: [_line("B", 2, 50)], 2: [_line("C", 1, 1)]}
    reads[unread_index * 2] = None  # page 1 or page 3 not read
    spans = [bpr.BillSpan("first", 1, 1, 50), bpr.BillSpan("rest", 2, 3, 101)]
    plan = bd.plan_bill_sheet("p1", "src1", rows_of_reads=_model_reader(reads),
                              states_of_rows=functools.partial(bpr.bill_states, spans=spans,
                                                               tolerance=0),
                              classify=_BILL, page_source=_blank_pages())
    assert [b["state"] for b in plan["bill_states"]["bills"]] == want
