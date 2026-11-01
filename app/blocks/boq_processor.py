"""BOQ Processor Block - Parse Excel/CSV/PDF Bills of Quantities into structured line items"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
# The app's own database and crypto modules load with the block (the server
# imports both at startup anyway). Their first import -- SQLAlchemy and the
# cryptography extension -- holds the GIL even in a worker thread, and in a
# cold process froze the event loop up to 0.41 s during the first parse (CI
# test-timing). The parse libraries (pandas, pdfplumber) stay lazy: the boot
# check keeps them out of startup.
from app.core import doc_index as _doc_index  # noqa: F401
from app.core import file_crypto as _file_crypto  # noqa: F401
from app.core.universal_base import UniversalBlock
from app.lib.boq_units import reconcile_unit

_logger = logging.getLogger(__name__)


def _boq_pdf_max_mb(filename: str = "") -> float:
    """Parse-refuse ceiling, aligned with the OCR plaintext gate.

    A live scanned priced BOQ is ~26.9 MB plaintext / ~28 MB
    recorded — above the old hardcoded ``BOQ_PDF_MAX_MB=20`` and already
    inside the 32 MB OCR gate. Leftover dashboard ``BOQ_PDF_MAX_MB=20``
    or ``PDF_OCR_MAX_SIZE_MB=25`` must not re-refuse it. BOQ-named files
    use ``PDF_OCR_BOQ_MAX_SIZE_MB`` (default ``PDF_MAX_SIZE_MB``). Ops
    may raise the cap via those env vars; they cannot lower it below
    the 32 MB OCR default.
    """
    from app.core.doc_index import (
        _DEFAULT_PDF_OCR_MAX_SIZE_MB,
        pdf_ocr_max_size_mb,
    )

    max_mb = max(pdf_ocr_max_size_mb(filename), _DEFAULT_PDF_OCR_MAX_SIZE_MB)
    raw = (os.getenv("BOQ_PDF_MAX_MB") or "").strip()
    if not raw:
        return max_mb
    try:
        env_max = float(raw)
    except ValueError:
        return max_mb
    return env_max if env_max > max_mb else max_mb


# A chat turn cannot afford to table-parse a whole contract volume. Measured on
# the live priced bill (370 pages): ~3 s per page even with pages released, so
# ~19 minutes -- and 43 MB per page retained without release, which OOM-killed
# the 2 GB web instance 75 s in (21 Sep 2026). Past these budgets the block
# refuses and points at what does work: the indexed text, a specific item, or
# the .xlsx.
_DEFAULT_BOQ_PDF_MAX_PAGES = 60
_DEFAULT_BOQ_PDF_PARSE_SECONDS = 90.0


def _env_number(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _boq_pdf_max_pages() -> int:
    return int(_env_number("BOQ_PDF_MAX_PAGES", _DEFAULT_BOQ_PDF_MAX_PAGES))


def _boq_pdf_parse_seconds() -> float:
    return _env_number("BOQ_PDF_PARSE_SECONDS", _DEFAULT_BOQ_PDF_PARSE_SECONDS)


def _parse_clock() -> float:
    """The clock the PDF parse budget is measured on.

    One named seam, so a test can drive the budget with a fake clock instead
    of real elapsed time (patching ``time.monotonic`` itself would also move
    asyncio's own timers).
    """
    return time.monotonic()


class _ParseBudgetExceeded(Exception):
    pass


def _resolve_via_project(project_id: str, raw: str) -> str:
    """Map a bare filename to the project's stored absolute file_path.

    Layered fallback for callers that hand us a filename instead of a path
    (the LLM typically does this -- it knows the original document name from
    the user, not the disk location). Returns the input unchanged if no
    matching document is found or the project store isn't reachable.
    """
    if not raw or not project_id or not isinstance(raw, str):
        return raw
    try:
        if os.path.isabs(raw) and os.path.exists(raw):
            return raw
    except (TypeError, ValueError):
        return raw
    try:
        from app.core import projects as _projects
        docs = _projects.list_documents(project_id) or []
    except Exception:
        return raw

    needle = os.path.basename(str(raw)).strip().lower()
    if not needle:
        return raw

    # Pass 1: exact (case-insensitive) match.
    for doc in docs:
        on = (doc.get("original_name") or "").strip().lower()
        if on and on == needle:
            fp = doc.get("file_path") or ""
            if fp and os.path.exists(fp):
                return fp

    # Pass 2: substring match either direction (LLM truncations / rewordings).
    for doc in docs:
        on = (doc.get("original_name") or "").strip().lower()
        if on and (needle in on or on in needle):
            fp = doc.get("file_path") or ""
            if fp and os.path.exists(fp):
                return fp

    return raw


def _parse_pdf_in_child(file_path: str, params: Dict) -> Dict:
    return BOQProcessorBlock()._parse_pdf_sync(file_path, params)


def _parse_pdf_isolated(file_path: str, params: Dict) -> Dict:
    """The PDF table parse in the platform's isolated child process
    (app.core.extract_isolated), in-process where fork is unavailable.

    Its parse libraries' first import (numpy / pandas, pdfplumber) holds the
    GIL even in a worker thread; in the child it holds the child's GIL, so the
    server's event loop is never frozen by it (CI test-timing: 0.41 s after
    #1001). The child also caps the parse's memory. Its wall-clock limit is the
    parse's own budget plus a margin, so the budget -- not the kill -- ends a
    long parse; a child that fails is a named error, never partial totals."""
    from app.core import extract_isolated

    fallback = {"status": "error",
                "error": "The PDF could not be parsed on the server. Upload the .xlsx or .csv version of this BOQ."}
    result, diag = extract_isolated.run_isolated(
        _parse_pdf_in_child, (file_path, params), fallback=fallback, label="boq pdf parse",
        timeout_s=_boq_pdf_parse_seconds() + 30.0)
    if diag:
        _logger.warning("boq pdf parse in the isolated child failed: %s", diag)
        return {**fallback, "parse_diag": diag}
    return result


class BOQProcessorBlock(UniversalBlock):
    name = "boq_processor"
    version = "1.1.0"
    description = "Parse Excel/CSV/PDF Bills of Quantities into structured quantities and cost breakdown"
    layer = 3
    tags = ["domain", "construction", "boq", "quantities", "excel", "pdf"]
    requires = []

    default_config = {
        "currency": "",
        "include_zero_qty": False,
    }

    ui_schema = {
        "input": {
            "type": "file",
            "accept": [".xlsx", ".xls", ".csv", ".pdf"],
            "placeholder": "Upload BOQ spreadsheet (.xlsx, .csv) or BOQ PDF...",
        },
        "output": {
            "type": "table",
            "fields": [
                {"name": "item_count", "type": "number", "label": "Line Items"},
                {"name": "total_cost", "type": "number", "unit": "USD", "label": "Total Cost"},
                {"name": "line_items", "type": "list", "label": "Line Items"},
                {"name": "cost_breakdown", "type": "json", "label": "Cost Breakdown"},
            ],
        },
        "quick_actions": [
            {"icon": "", "label": "Parse BOQ", "prompt": "Parse and summarize this Bill of Quantities"},
            {"icon": "", "label": "Cost Summary", "prompt": "Give me a cost breakdown by trade/division"},
        ],
    }

    # Common BOQ column name aliases
    _COL_MAP = {
        "description": ["description", "item_description", "work_item", "item", "activity", "desc", "name"],
        "quantity": ["quantity", "qty", "no", "number", "count"],
        "unit": ["unit", "uom", "u/m", "unit_of_measure", "measure"],
        "rate": ["rate", "unit_cost", "unit_price", "price", "unit_rate", "cost_per_unit", "cost/unit"],
        "total": ["total", "total_cost", "amount", "line_total", "extended_price", "cost", "value"],
        "section": ["section", "division", "trade", "category", "csi_div", "package", "work_package"],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        file_path = data.get("file_path") or params.get("file_path") or data.get("text") or data.get("input") or (input_data if isinstance(input_data, str) else "")

        # Inline CSV/BOQ pasted in the chat message (no uploaded file).
        inline_boq_csv = False
        raw_text = data.get("text") or data.get("content") or data.get("lines") or params.get("text") or params.get("content") or ""
        if isinstance(input_data, dict):
            raw_text = raw_text or input_data.get("text") or input_data.get("content") or ""
        if (not file_path or ("\n" in str(file_path) and "," in str(file_path))) and raw_text and "," in str(raw_text) and "\n" in str(raw_text):
            import tempfile
            inline_boq_csv = True
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
            # strip leading instruction lines; keep CSV-looking rows
            body_lines = []
            for ln in str(raw_text if raw_text else file_path).splitlines():
                if "," in ln and not ln.strip().lower().startswith("use boq"):
                    body_lines.append(ln)
            tmp.write("\n".join(body_lines) + "\n")
            tmp.close()
            file_path = tmp.name
        elif isinstance(file_path, str) and "\n" in file_path and "," in file_path:
            import tempfile
            inline_boq_csv = True
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
            body_lines = [ln for ln in file_path.splitlines() if "," in ln]
            tmp.write("\n".join(body_lines) + "\n")
            tmp.close()
            file_path = tmp.name
        if not file_path:
            return {"status": "error", "error": "No file_path provided. Requires an .xlsx, .csv, or .pdf BOQ file path, or inline CSV lines."}

        # LLM callers typically pass a bare filename ("Demolition BOQ.pdf") rather
        # than the stored absolute path. Try to resolve that against the project's
        # uploaded documents before failing with "File not found".
        if not os.path.exists(str(file_path)):
            project_id = (
                params.get("project_id")
                or (data.get("project_id") if isinstance(data, dict) else None)
            )
            if project_id:
                resolved = _resolve_via_project(str(project_id), str(file_path))
                if resolved and resolved != file_path and os.path.exists(resolved):
                    file_path = resolved

        if not os.path.exists(str(file_path)):
            return {"status": "error", "error": f"File not found: {file_path}"}

        ext = os.path.splitext(file_path)[1].lower()
        # Prefer the caller-supplied name (LLM often passes the original
        # "…BOQ… (Priced).pdf") so the parse cap can use the BOQ OCR
        # ceiling. After resolve, file_path may be a UUID on disk.
        source_name = os.path.basename(
            str(data.get("file_path") or params.get("file_path") or file_path)
        )
        try:
            # Decrypting, measuring and parsing are one blocking job, and so
            # are the first imports they trigger (file_crypto, doc_index,
            # pandas, pdfplumber): all of it runs in one worker thread, never
            # on the event loop (LANE5 #984/#992). Imports stay lazy, so
            # nothing heavy loads at boot.
            return await asyncio.to_thread(self._process_file, file_path, ext, params, source_name)
        except ImportError as e:
            return {
                "status": "error",
                "error": f"Missing dependency: {e}. Run: pip install pandas openpyxl pdfplumber",
            }
        except Exception as e:
            return {"status": "error", "error": f"Parse error: {e}"}

    # Every parser is synchronous, CPU-bound work. The web service runs ONE
    # uvicorn worker, so parsing on the event loop freezes health checks and
    # every other user for the whole parse (Render then kills the instance as
    # unhealthy). The async wrappers hand the work to a thread.

    async def _parse_csv(self, file_path: str, params: Dict) -> Dict:
        return await asyncio.to_thread(self._parse_csv_sync, file_path, params)

    async def _parse_excel(self, file_path: str, params: Dict) -> Dict:
        return await asyncio.to_thread(self._parse_excel_sync, file_path, params)

    def _parse_csv_sync(self, file_path: str, params: Dict) -> Dict:
        import pandas as pd
        df = pd.read_csv(file_path)
        return self._process_dataframe(df, params)

    # How many leading rows to scan for the real column header. Contract
    # volumes open with project/employer/bill-number banner rows; 60 covers
    # every real bill seen while keeping the scan trivial.
    _HEADER_SCAN_ROWS = 60

    def _detect_header_row(self, df_raw) -> int:
        """Row index of the actual column header, or 0.

        A header row is the first row whose cells resolve to at least two
        DISTINCT canonical columns, one of which is `description`. Matching
        through _resolve_columns keeps this in lockstep with the alias map --
        a new alias automatically improves detection too.
        """
        limit = min(self._HEADER_SCAN_ROWS, len(df_raw))
        for i in range(limit):
            cells = [str(c).strip() for c in df_raw.iloc[i].tolist()]
            resolved = self._resolve_columns(cells)
            if "description" in resolved and len(resolved) >= 2:
                return i
        return 0

    def _parse_excel_sync(self, file_path: str, params: Dict) -> Dict:
        import pandas as pd
        sheet = params.get("sheet_name", 0)
        # Engine by format, not one-size: openpyxl reads only zip-based .xlsx,
        # so the old hardcoded engine="openpyxl" made every legacy BIFF .xls
        # die with "File is not a zip file" -- while the block's own ui_schema
        # advertises ".xls". Real contract-volume BOQs are routinely issued in
        # .xls (found live: 12,971-row Vol III bill). xlrd is the BIFF engine.
        engine = "xlrd" if str(file_path).lower().endswith(".xls") else "openpyxl"
        # Real bills open with banner rows (project, employer, bill number),
        # so the column header is rarely row 0. Read headerless, find the
        # header by alias match, then promote it. Found live: a 12,971-row
        # bill parsed to ZERO items because row 0 was a title fragment.
        raw = pd.read_excel(file_path, sheet_name=sheet, engine=engine, header=None)
        hdr = self._detect_header_row(raw)
        df = raw.iloc[hdr + 1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in raw.iloc[hdr].tolist()]
        return self._process_dataframe(df, params)

    def _process_file(self, file_path: str, ext: str, params: Dict, source_name: str) -> Dict:
        """Decrypt and parse in the calling worker thread. open_plaintext
        decrypts when DATA_ENCRYPTION_KEY is set (uploads go through
        file_crypto.write_document) and is a no-op for plaintext files."""
        from app.core.file_crypto import open_plaintext

        with open_plaintext(file_path) as plain_path:
            if ext == ".csv":
                return self._parse_csv_sync(plain_path, params)
            if ext in (".xlsx", ".xls"):
                return self._parse_excel_sync(plain_path, params)
            if ext == ".pdf":
                return (self._pdf_too_large(plain_path, source_name)
                        or _parse_pdf_isolated(plain_path, params))
            return {
                "status": "error",
                "error": f"Unsupported format: {ext}. Use .xlsx, .csv, or .pdf",
            }

    async def _parse_pdf(
        self, file_path: str, params: Dict, source_name: str = ""
    ) -> Dict:
        """Extract tabular BOQ data from a PDF via pdfplumber.

        Walks every page, calls ``page.extract_tables()``, treats row 0 of each
        table as a header, then groups tables by header signature so
        continuation pages with identical headers concatenate into a single
        DataFrame. The largest such group is treated as the BOQ and dispatched
        to ``_process_dataframe`` (the same path Excel/CSV use), so the column
        aliasing in ``_resolve_columns`` works for any BOQ table format.

        Returns the standard BOQ result (item_count, total_cost, line_items,
        cost_breakdown). If no extractable tables are found (e.g. a pure-image
        scanned PDF that wasn't OCR'd first), surfaces a clear error pointing
        to that limitation.
        """
        # ── memory guard ────────────────────────────────────────────────
        # pdfplumber loads the entire PDF and builds every page's table grid
        # in memory; on a large priced BOQ this OOM-kills the worker and 502s
        # EVERY concurrent user, not just this request. Align the refuse with
        # the OCR plaintext gate (32 MB / PDF_OCR_BOQ_MAX_SIZE_MB for
        # BOQ-named files) so the live 26.9 MB priced bill is not 413'd
        # after it already OCR'd. Measure plaintext, not Fernet ciphertext
        # (same class as #449). See memory the-fork-boq-always-xlsx.
        refusal = await asyncio.to_thread(self._pdf_too_large, file_path, source_name)
        if refusal:
            return refusal
        return await asyncio.to_thread(self._parse_pdf_sync, file_path, params)

    @staticmethod
    def _pdf_too_large(file_path: str, source_name: str = "") -> Optional[Dict]:
        """The refusal for a PDF too large to table-parse, or None. Blocking
        (it measures the plaintext and imports doc_index): worker thread only."""
        from app.core.doc_index import _document_size_mb

        max_mb = _boq_pdf_max_mb(source_name or os.path.basename(file_path))
        size_mb = _document_size_mb(file_path)
        if size_mb > max_mb:
            return {
                "status": "error",
                "error": (
                    f"This BOQ PDF is too large to parse on the server "
                    f"({size_mb:.1f} MB, limit {max_mb:.0f} MB). PDF table "
                    f"extraction loads the whole file into memory and can crash "
                    f"the server. Please upload the .xlsx or .csv version of "
                    f"this BOQ instead — it parses reliably and computes the "
                    f"totals without the memory risk."
                ),
                "boq_pdf_too_large": True,
                "size_mb": round(size_mb, 1),
                "max_mb": max_mb,
            }
        return None

    def _parse_pdf_sync(self, file_path: str, params: Dict) -> Dict:
        import pdfplumber

        max_pages = _boq_pdf_max_pages()
        budget_s = _boq_pdf_parse_seconds()
        deadline = _parse_clock() + budget_s
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
        if page_count > max_pages:
            return {
                "status": "error",
                "error": (
                    f"This BOQ PDF has {page_count} pages -- too large to "
                    f"table-parse inside a chat turn (limit {max_pages} pages). "
                    f"Its contents are already indexed: ask about a specific "
                    f"item (e.g. by its item number) and the answer comes from "
                    f"the indexed bill, or upload the .xlsx version of this BOQ "
                    f"to compute full totals."
                ),
                "boq_pdf_too_many_pages": True,
                "page_count": page_count,
                "max_pages": max_pages,
            }
        try:
            return self._parse_pdf_pages(file_path, params, deadline)
        except _ParseBudgetExceeded as exc:
            return {
                "status": "error",
                "error": (
                    f"Parsing this BOQ PDF ran past its {budget_s:.0f} s budget "
                    f"({exc}) and was stopped; no totals are returned because "
                    f"they would be incomplete. Ask about a specific item (its "
                    f"contents are indexed) or upload the .xlsx version."
                ),
                "boq_pdf_parse_timeout": True,
                "page_count": page_count,
                "budget_seconds": budget_s,
            }

    def _parse_pdf_pages(self, file_path: str, params: Dict, deadline: float) -> Dict:
        import pdfplumber
        import pandas as pd

        def check_budget(page_index: int) -> None:
            if _parse_clock() > deadline:
                raise _ParseBudgetExceeded(f"stopped at page {page_index}")

        # Group tables by a canonical "shape" key built from normalized headers
        # so continuation pages merge naturally.
        groups: Dict[Tuple[str, ...], List[List[List[str]]]] = {}
        primary_headers_for: Dict[Tuple[str, ...], List[str]] = {}
        page_table_count = 0
        # Track pages we silently dropped so the caller knows the BOQ totals
        # may be incomplete. Previously these failures were swallowed, which
        # turned a partial-parse into a wrong client deliverable.
        pages_skipped = 0
        pages_skipped_reasons: List[Dict[str, Any]] = []

        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages, 1):
                check_budget(page_index)
                try:
                    tables = page.extract_tables() or []
                except Exception as exc:
                    _logger.warning(
                        "boq_processor: page %s extract_tables failed: %s",
                        page_index,
                        exc,
                    )
                    pages_skipped += 1
                    pages_skipped_reasons.append({
                        "page": page_index,
                        "stage": "extract_tables",
                        "error": str(exc),
                    })
                    continue
                finally:
                    # Release the page's parsed layout now: pdfplumber keeps it
                    # otherwise (43 MB/page on the live bill -> 1 MB/page).
                    page.close()
                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    headers_raw = [(str(c or "").strip()) for c in tbl[0]]
                    if not any(headers_raw):
                        continue
                    n_cols = len(headers_raw)
                    if n_cols < 2:
                        continue
                    # Canonical shape key: tuple of normalized header tokens so
                    # tables across pages with identical structure merge.
                    key = tuple(self._normalize_col(h) for h in headers_raw)
                    norm_rows: List[List[str]] = []
                    for row in tbl[1:]:
                        if row is None:
                            continue
                        cells = [(str(c) if c is not None else "") for c in row]
                        if len(cells) < n_cols:
                            cells = cells + [""] * (n_cols - len(cells))
                        elif len(cells) > n_cols:
                            cells = cells[:n_cols]
                        if not any(s.strip() for s in cells):
                            continue  # blank row
                        norm_rows.append(cells)
                    if not norm_rows:
                        continue
                    groups.setdefault(key, []).extend(norm_rows)
                    primary_headers_for.setdefault(key, headers_raw)
                    page_table_count += 1

        if not groups:
            # pdfplumber found no tabular structure. This is normal for scanned
            # BOQ PDFs where OCR captured the text but page layout doesn't yield
            # a clean table grid. Return per-page text so the caller (typically
            # an LLM agent) can extract line items from the raw OCR text.
            page_texts: List[Dict[str, Any]] = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    check_budget(i)
                    try:
                        t = (page.extract_text() or "").strip()
                    except Exception as exc:
                        _logger.warning(
                            "boq_processor: page %s extract_text failed: %s",
                            i,
                            exc,
                        )
                        pages_skipped += 1
                        pages_skipped_reasons.append({
                            "page": i,
                            "stage": "extract_text",
                            "error": str(exc),
                        })
                        t = ""
                    finally:
                        page.close()
                    if t:
                        page_texts.append({"page": i, "text": t})
            if page_texts:
                return {
                    "status": "partial",
                    "source_format": "pdf",
                    "note": (
                        "PDF has no clean tabular structure (likely scanned). "
                        "Returning raw page text — pass to an LLM for BOQ line-item extraction."
                    ),
                    "page_count": len(page_texts),
                    "page_texts": page_texts,
                    "pages_skipped": pages_skipped,
                    "pages_skipped_reasons": pages_skipped_reasons,
                }
            return {
                "status": "error",
                "error": (
                    "No extractable tables or text found in this PDF. The file "
                    "may be a pure-image scan that hasn't been OCR'd. Run OCR "
                    "first or re-upload as .xlsx/.csv."
                ),
                "pages_skipped": pages_skipped,
                "pages_skipped_reasons": pages_skipped_reasons,
            }

        # Pick the largest group as the BOQ; other groups (smaller summary
        # tables, headers/footers picked up as tables) are ignored.
        best_key = max(groups.keys(), key=lambda k: len(groups[k]))
        df = pd.DataFrame(groups[best_key], columns=primary_headers_for[best_key])

        result = self._process_dataframe(df, params)
        # Surface PDF-specific diagnostics so the caller knows we used the PDF path.
        if isinstance(result, dict):
            result.setdefault("source_format", "pdf")
            result["pdf_tables_total"] = page_table_count
            result["pdf_tables_used"] = len(groups[best_key])
            # Surface dropped pages so callers can flag the BOQ total as
            # potentially incomplete instead of silently shipping a low number.
            result["pages_skipped"] = pages_skipped
            result["pages_skipped_reasons"] = pages_skipped_reasons
        return result

    @staticmethod
    def _normalize_col(name: str) -> str:
        """Reduce a raw column header to a canonical token for alias matching.

        Strips parenthesized suffixes (currency / unit hints) and a small set of
        trailing currency tokens, then lowercases and replaces spaces/slashes
        with underscores. Lets us match real BOQ headers like 'Rate (SAR)',
        'Amount (USD)', 'Qty.', 'Item No.' against the short alias list.
        """
        import re
        n = name.strip()
        # Strip any (...) suffix — usually a currency or unit qualifier.
        n = re.sub(r"\s*\([^)]*\)\s*$", "", n)
        # Strip trailing currency tokens with optional punctuation.
        n = re.sub(
            r"[\s,;:]+(SAR|USD|AED|EUR|GBP|JPY|CNY|AUD|CAD|KWD|QAR|BHD|OMR)\b\.?$",
            "",
            n,
            flags=re.IGNORECASE,
        )
        # Drop trailing punctuation ("Qty.", "Item No.").
        n = n.rstrip(" .,:;")
        n = n.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        # Collapse repeats of underscore so 'item__no' becomes 'item_no'.
        n = re.sub(r"_+", "_", n).strip("_")
        return n

    def _resolve_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map alias-set field names to actual DataFrame column names.

        Two-pass: exact normalized match first (cheap, deterministic), then
        substring match against the same alias set as a fallback so columns
        like 'unit_rate_in_sar' still resolve when the prefix is recognised.
        """
        resolved: Dict[str, str] = {}
        normalized = [self._normalize_col(c) for c in columns]
        for field, candidates in self._COL_MAP.items():
            chosen_idx = None
            # Pass 1: exact normalized match.
            for c in candidates:
                if c in normalized:
                    chosen_idx = normalized.index(c)
                    break
            # Pass 2: substring match (longest alias first so "unit_rate" beats "unit").
            if chosen_idx is None:
                for c in sorted(candidates, key=len, reverse=True):
                    for i, ncol in enumerate(normalized):
                        if c in ncol.split("_") or ncol.startswith(c + "_") or ncol.endswith("_" + c):
                            chosen_idx = i
                            break
                    if chosen_idx is not None:
                        break
            if chosen_idx is not None:
                resolved[field] = columns[chosen_idx]
        return resolved

    def _process_dataframe(self, df, params: Dict) -> Dict:
        df.columns = [str(c).strip() for c in df.columns]
        resolved = self._resolve_columns(list(df.columns))

        include_zero = params.get("include_zero_qty", self.config.get("include_zero_qty", False))
        currency = params.get("currency") or self.config.get("currency") or ""

        line_items: List[Dict] = []
        section_totals: Dict[str, float] = {}
        warnings: List[str] = []
        skipped_items: List[Dict] = []

        quantity_col_resolved = "quantity" in resolved
        if not quantity_col_resolved:
            warnings.append(
                "No quantity column detected; zero-quantity filter disabled and all rows retained."
            )

        for _, row in df.iterrows():
            description = str(row.get(resolved.get("description", ""), "")).strip()
            if not description or description.lower() == "nan":
                continue

            raw_qty = row.get(resolved.get("quantity", ""), 0)
            try:
                qty = _to_float(raw_qty)
            except ValueError:
                skipped_items.append({"description": description, "raw_value": raw_qty})
                continue
            if quantity_col_resolved and not include_zero and qty == 0:
                continue

            rate = _to_float_safe(row.get(resolved.get("rate", ""), 0))
            total = _to_float_safe(row.get(resolved.get("total", ""), 0))
            if total == 0 and qty > 0 and rate > 0:
                total = qty * rate

            unit_raw = str(row.get(resolved.get("unit", ""), "")).strip()
            # Reconcile the OCR/printed unit against the unit implied by the work
            # type (CESMM4/POMI): fill a blank/unreadable unit, and flag a stated
            # unit that contradicts the work type (kept, never overwritten).
            unit, unit_source, unit_suspect, unit_expected = reconcile_unit(
                unit_raw, description
            )
            section = str(row.get(resolved.get("section", ""), "General")).strip()
            if section.lower() == "nan":
                section = "General"

            item_key = description.lower().replace(" ", "_")[:50]

            line_items.append(
                {
                    "item_key": item_key,
                    "description": description,
                    "quantity": qty,
                    "unit": unit,
                    "unit_source": unit_source,
                    "unit_suspect": unit_suspect,
                    "unit_expected": unit_expected,
                    "unit_cost": rate,
                    "total_cost": round(total, 2),
                    "section": section,
                    "currency": currency,
                }
            )
            section_totals[section] = section_totals.get(section, 0.0) + total

        total_cost = sum(i["total_cost"] for i in line_items)
        cost_breakdown = {
            section: {
                "total": round(v, 2),
                "percentage": round(v / total_cost * 100, 1) if total_cost > 0 else 0,
            }
            for section, v in sorted(section_totals.items(), key=lambda x: x[1], reverse=True)
        }

        result = {
            "status": "success",
            "item_count": len(line_items),
            "total_cost": round(total_cost, 2),
            "currency": currency,
            "line_items": line_items,
            "cost_breakdown": cost_breakdown,
            "sections": list(section_totals.keys()),
            "columns_detected": resolved,
        }
        if warnings:
            result["warnings"] = warnings
        if skipped_items:
            result["skipped_items"] = skipped_items
        return result


def _to_float(val) -> float:
    """Coerce ``val`` to float. Raises ``ValueError`` for non-empty non-numeric
    strings so callers can distinguish a true zero from an unparseable value
    like ``"Lot"`` or ``"Provisional Sum"``."""
    if val is None:
        return 0.0
    s = str(val).replace(",", "").strip()
    if s == "" or s.lower() == "nan":
        return 0.0
    try:
        return float(s)
    except TypeError:
        return 0.0


def _to_float_safe(val) -> float:
    """Best-effort coercion to float; returns 0.0 on any failure."""
    try:
        return _to_float(val)
    except (ValueError, TypeError):
        return 0.0
