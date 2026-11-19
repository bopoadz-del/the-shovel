"""PLAN (never write) the derived bill sheet for ONE document, inside the ingest task.

Run by ``.github/workflows/bill-plan-once.yml`` through the ingest worker:
``p1b_ingest_drive_server.main`` hands over to :func:`main_from_env` when
``BILL_PLAN_DOCUMENT_ID`` is set. The ingest worker's ``WORKER_COMMAND`` is a
Secrets Manager secret that beats argv and a same-name environment override,
so the selectors travel as distinct environment names:

* ``BILL_PLAN_PROJECT_ID`` / ``BILL_PLAN_DOCUMENT_ID`` (both required)
* ``BILL_PLAN_PAGES`` (optional ``first:stop``, 0-based half-open; empty =
  every page). A long document is planned in ranges small enough to finish
  inside one task run; each page's word boxes stay in the work directory, so
  the next range (or a final whole-document run) reads them back instead of
  OCRing again.
* ``BILL_PLAN_WORK_DIR`` (optional; default ``$DATA_DIR/bill_plan/<id>``).

Without ``BILL_PLAN_EXECUTE_HASH`` this calls
:func:`app.core.bill_derived.plan_bill_sheet` only. It adds no document row,
indexes nothing and writes no database row; the only bytes it keeps are the
per-page word boxes under the work directory.

EXECUTE (``BILL_PLAN_EXECUTE_HASH``, a 64-hex plan hash): the run rebuilds
the whole-document plan from the saved word boxes, refuses unless the rebuilt
plan's hash equals the given hash AND every run condition (:class:`Gates`)
holds on that rebuilt plan, and only then calls the existing
:func:`app.core.bill_derived.derive_bill_sheet` with that hash (which checks
the hash again). The conditions are parameters with defaults, never tied to
one document: a priced verdict; every page read back from the work dir (no
OCR, no fresh text-layer read); at most a stated share of priced rows not
reconciling and of priced rows with an empty description; no header
conflict and no priced row without a section path; and an optional floor of
priced, reconciled rows given for the run (``BILL_EXECUTE_MIN_RECONCILED``).
It reports counts only: rows written, chunks added, chunks longer than the
embedder window, the derived document's status and the superseded id.

The last line is ``BILLPLANSUMMARY <json>``. That line reaches a GitHub job
summary, which is world-readable, so it carries counts, codes and hashes
only: the classifier verdict and its counts, the page range, pages cached
before and after, rows, unreconciled rows, the number of section paths, the
rows per path depth, the number of sections (distinct first path elements,
the section title), the rows in each section as a list of counts and the
rows whose path has no title, pages by word source (text layer, OCR, work-dir cache),
wall time, peak RSS and the plan hash. It is built by
:func:`public_summary` from a fixed key list, and every string in it must be
a short code token (at most a hash long); anything else is replaced by
``"redacted"``. An error is a code plus the exception CLASS, never its
message. Nothing else is printed.

The full plan, section labels included, is kept privately in the work
directory on the task's data mount (``plan_<first>_<stop>.json``), and an
exception's message goes to ``last_error.json`` beside it. Neither is
printed. ``work_dir_durable`` says whether that directory is on a durable
mount (``byte_store.is_durable_mount``): only then do the word boxes and
the private plan outlive the task.

After a plan, ``geometry`` carries the counts-only geometry diagnostic
(``bill_geometry_diag.geometry_diag``) over the SAVED word boxes of the
planned range: it reads no page again (a page with no saved boxes is counted
as ``pages_missing``, never read), and it fails soft: an exception is
reported as its class under ``geometry.error_class`` and leaves the plan's
own outcome unchanged.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUMMARY_PREFIX = "BILLPLANSUMMARY "
#: Path depths reported individually; deeper paths are counted together.
MAX_LISTED_DEPTH = 3
#: Plan fields the summary carries as they are (all ids or counts; ``columns``
#: is the column-source counts of ``bill_table_geometry.ColumnStats``).
PLAN_FIELDS = ("reason", "pages", "pages_in_file", "rows", "unreconciled", "plan_hash",
               "columns")
#: Page-source counts the summary carries (pages with and without a text
#: layer, and pages read fresh from each source or back from the work dir).
SOURCE_FIELDS = (
    "pages_text_layer", "pages_without_text_layer",
    "pages_read_text_layer", "pages_read_ocr", "pages_read_cache",
)
#: Verdict counts the summary carries. The PLAN judges every page it read
#: (``bill_classify.verdict_of_reads``), and the diagnostic counts say what
#: those pages were read from and why their lines were not header lines (by
#: reason code), so a refused verdict separates "no table on these pages"
#: from "the word boxes do not form a header line the aliasing resolves".
VERDICT_FIELDS = (
    "pages_read", "pages_sampled", "pages_text_layer", "pages_ocr", "pages_without_words",
    "header_pages", "lines_tested", "header_lines",
    "rejected_fewer_than_3_cells", "rejected_no_description_column",
    "rejected_no_figure_column",
    "measured_rows", "priced_rows", "reconciled_rows", "inferred_pages",
)
#: A plan hash is a sha256 hex digest; no summary string may be longer.
HASH_LEN = 64
#: The only shape a summary string may take: a code token, an id or a hash.
_TOKEN = re.compile(rf"[A-Za-z0-9_.:+-]{{1,{HASH_LEN}}}")
REDACTED = "redacted"
#: Every key the public summary may carry; anything else is dropped.
SUMMARY_KEYS = (
    "ok", "error", "error_class", "project_id", "document_id", "build_sha",
    "requested_pages", "pages_cached_before", "pages_cached_after", "wall_s",
    "peak_rss_mb", "verdict", *PLAN_FIELDS, *SOURCE_FIELDS, "section_paths", "rows_by_path_depth",
    "section_count", "rows_per_section", "rows_empty_title",
    "private_plan_saved", "work_dir_durable", "geometry",
    "mode", "execute_hash", "gates", "refused", "derived_document_id", "superseded",
    "rows_written", "chunks_added", "chunks_over_window", "document_status", "index_status",
)
#: Error codes (the summary never carries a message).
ERR_MISSING_IDS = "missing_ids"
ERR_BAD_PAGES = "bad_pages"
ERR_NOT_FOUND = "document_not_found"
ERR_WRONG_PROJECT = "document_not_in_project"
ERR_RAISED = "plan_raised"
ERR_NO_PLAN = "no_plan"
ERR_NOT_SAVED = "plan_not_saved"
ERR_BAD_EXECUTE = "bad_execute_params"
ERR_REFUSED = "execute_refused"
ERR_NOT_DERIVED = "not_derived"

MODE_EXECUTE = "execute"
#: An execute hash is exactly a lowercase sha256 hex digest.
_HASH_RE = re.compile(r"[0-9a-f]{64}")

#: Refusal codes of an execute (each names the condition that failed).
REFUSE_NO_PLAN = "no_plan"
REFUSE_HASH = "plan_hash_mismatch"
REFUSE_NOT_WHOLE = "plan_not_whole_document"
REFUSE_NOT_PRICED = "verdict_not_priced"
REFUSE_NOT_CACHED = "pages_not_all_from_cache"
REFUSE_COUNTS_MISSING = "plan_counts_missing"
REFUSE_FEW_RECONCILED = "reconciled_rows_below_floor"
REFUSE_UNRECONCILED = "unreconciled_share_over_ceiling"
REFUSE_NO_DESCRIPTION = "empty_description_share_over_ceiling"
REFUSE_CONFLICTS = "header_conflicts_over_ceiling"
REFUSE_NO_PATH = "priced_rows_without_path_over_ceiling"
#: The plan's column counts every condition reads.
_GATE_COUNTS = ("rows_priced_reconciled", "rows_figures_not_reconciling",
                "priced_rows_empty_description", "priced_rows_empty_path", "header_conflicts")


@dataclass(frozen=True)
class Gates:
    """The conditions an execute enforces on the rebuilt plan.

    Shares are of the plan's priced rows (reconciling or not). The floor of
    priced, reconciled rows is given per run; 0 means no floor."""

    min_reconciled_rows: int = 0
    max_unreconciled_share: float = 0.005
    max_empty_description_share: float = 0.02
    max_header_conflicts: int = 0
    max_priced_rows_without_path: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def gate_refusals(plan: Mapping[str, Any], execute_hash: str, gates: Gates) -> list[str]:
    """Every condition the rebuilt ``plan`` fails, as refusal codes (empty =
    the write may go ahead)."""
    from app.core import bill_classify

    if plan.get("sections") is None or "plan_hash" not in plan:
        return [REFUSE_NO_PLAN]
    out: list[str] = []
    if plan["plan_hash"] != execute_hash:
        out.append(REFUSE_HASH)
    pages_in_file = plan.get("pages_in_file")
    if list(plan.get("pages") or []) != [0, pages_in_file]:
        out.append(REFUSE_NOT_WHOLE)
    if plan.get("reason") != bill_classify.BILL_PRICED:
        out.append(REFUSE_NOT_PRICED)
    if (plan.get("pages_read_cache") != pages_in_file
            or plan.get("pages_read_ocr") != 0
            or plan.get("pages_read_text_layer") != 0):
        out.append(REFUSE_NOT_CACHED)
    cols = plan.get("columns") or {}
    if any(not isinstance(cols.get(k), int) for k in _GATE_COUNTS):
        return [*out, REFUSE_COUNTS_MISSING]
    reconciled = cols["rows_priced_reconciled"]
    unreconciled = cols["rows_figures_not_reconciling"]
    priced = reconciled + unreconciled
    if reconciled < gates.min_reconciled_rows:
        out.append(REFUSE_FEW_RECONCILED)
    if unreconciled > gates.max_unreconciled_share * priced:
        out.append(REFUSE_UNRECONCILED)
    if cols["priced_rows_empty_description"] > gates.max_empty_description_share * priced:
        out.append(REFUSE_NO_DESCRIPTION)
    if cols["header_conflicts"] > gates.max_header_conflicts:
        out.append(REFUSE_CONFLICTS)
    if cols["priced_rows_empty_path"] > gates.max_priced_rows_without_path:
        out.append(REFUSE_NO_PATH)
    return out


def _env_number(env: Mapping[str, str], name: str, default, kind):
    value = (env.get(name) or "").strip()
    if not value:
        return default
    out = kind(value)  # ValueError on a non-number
    if out < 0 or (kind is float and out > 1):
        raise ValueError(f"{name} out of range")
    return out


def parse_execute(env: Mapping[str, str]) -> tuple[str, Gates] | None:
    """``(hash, gates)`` when ``BILL_PLAN_EXECUTE_HASH`` is set, else None.
    Raises ValueError on a malformed hash or parameter."""
    raw = (env.get("BILL_PLAN_EXECUTE_HASH") or "").strip()
    if not raw:
        return None
    if not _HASH_RE.fullmatch(raw):
        raise ValueError("execute hash must be 64 lowercase hex")
    d = Gates()
    return raw, Gates(
        min_reconciled_rows=_env_number(
            env, "BILL_EXECUTE_MIN_RECONCILED", d.min_reconciled_rows, int),
        max_unreconciled_share=_env_number(
            env, "BILL_EXECUTE_MAX_UNRECONCILED_SHARE", d.max_unreconciled_share, float),
        max_empty_description_share=_env_number(
            env, "BILL_EXECUTE_MAX_EMPTY_DESCRIPTION_SHARE", d.max_empty_description_share, float),
    )


@dataclass
class Deps:
    """The I/O this run needs; production wires the real modules."""

    get_document: Callable[[str], dict[str, Any] | None]
    plan: Callable[..., dict[str, Any]]
    #: The verdict over the pages the plan read (``verdict_of_reads``).
    classify: Callable[[Any], Any] | None = None
    log: Callable[[str], None] = field(default=print)
    build_sha: Callable[[], str | None] = field(default=lambda: None)
    peak_rss_mb: Callable[[], float | None] = field(default=lambda: None)
    clock: Callable[[], float] = field(default=time.monotonic)
    #: Whether the work dir is on a durable mount (the word boxes and the
    #: private plan outlive the task only then); None = not known.
    durable: Callable[[str], bool | None] = field(default=lambda _path: None)
    #: Counts-only geometry diagnostic over the saved word boxes of a page
    #: range ``(work_dir, first, stop)``; None = not run.
    geometry: Callable[[str, int, int], dict[str, Any]] | None = None
    #: EXECUTE only: ``bill_derived.derive_bill_sheet``.
    derive: Callable[..., dict[str, Any]] | None = None
    #: EXECUTE only: the length summary of one document's stored chunks
    #: ``(project_id, document_id)`` (``chunk_bounds.bounds_summary``).
    chunk_bounds: Callable[[str, str], dict[str, Any]] | None = None


def parse_pages(raw: str | None) -> tuple[int, int] | None:
    """``"first:stop"`` -> ``(first, stop)``; empty -> ``None`` (every page)."""
    value = (raw or "").strip()
    if not value:
        return None
    first_s, sep, stop_s = value.partition(":")
    if not sep:
        raise ValueError(f"pages must be first:stop, got {raw!r}")
    first, stop = int(first_s), int(stop_s)  # ValueError on a non-number
    if first < 0 or stop <= first:
        raise ValueError(f"pages must satisfy 0 <= first < stop, got {raw!r}")
    return first, stop


def cached_pages(work_dir: str) -> int:
    """Pages whose word boxes are already saved under ``work_dir``."""
    if not os.path.isdir(work_dir):
        return 0
    return sum(
        1 for name in os.listdir(work_dir)
        if name.startswith("page_") and name.endswith(".json")
    )


def depth_distribution(sections: list[dict[str, Any]]) -> dict[str, int]:
    """Rows per section-path depth: ``"1"``, ``"2"``, ``"3+"`` (``MAX_LISTED_DEPTH``)."""
    out: dict[str, int] = {}
    for section in sections:
        depth = len(section.get("path") or ())
        key = f"{MAX_LISTED_DEPTH}+" if depth >= MAX_LISTED_DEPTH else str(depth)
        out[key] = out.get(key, 0) + int(section.get("rows") or 0)
    return dict(sorted(out.items()))


def section_counts(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by section title (a path's first element): the number of
    sections, the rows of each as a list of counts (largest first, no
    title), and the rows whose path has no title (empty or blank first
    element). A section is identified by its title only, so two sections
    printed with the same title are counted as one."""
    per: dict[str, int] = {}
    empty = 0
    for section in sections:
        path = section.get("path") or ()
        title = str(path[0]).strip() if path else ""
        rows = int(section.get("rows") or 0)
        if title:
            per[title] = per.get(title, 0) + rows
        else:
            empty += rows
    return {
        "section_count": len(per),
        "rows_per_section": sorted(per.values(), reverse=True),
        "rows_empty_title": empty,
    }


def verdict_counts(classify: Callable[[Any], Any]) -> tuple[Callable[[Any], Any], dict[str, Any]]:
    """Wrap ``classify`` so the run can report the verdict's counts."""
    seen: dict[str, Any] = {}

    def wrapped(reads):
        verdict = classify(reads)
        seen["is_bill"] = bool(verdict.is_bill)
        seen["reason"] = verdict.reason
        for name in VERDICT_FIELDS:
            seen[name] = getattr(verdict, name, None)
        return verdict

    return wrapped, seen


def _public_value(value: Any) -> Any:
    """Numbers, booleans and None pass; a string passes only as a code token."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if _TOKEN.fullmatch(value) else REDACTED
    if isinstance(value, Mapping):
        return {_public_value(str(k)): _public_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_value(v) for v in value]
    return REDACTED


def public_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The world-readable summary: fixed keys, counts/codes/hashes only."""
    return {k: _public_value(report[k]) for k in SUMMARY_KEYS if k in report}


def _write_private(work_dir: str, name: str, payload: Any) -> str | None:
    """Write ``payload`` as JSON under ``work_dir`` (atomic). Returns None, or
    the failing OSError's class name: the caller reports it as an outcome."""
    try:
        os.makedirs(work_dir, exist_ok=True)
        path = os.path.join(work_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, default=str)
        os.replace(tmp, path)
    except OSError as exc:
        return type(exc).__name__
    return None


def private_plan_name(plan: Mapping[str, Any]) -> str:
    first, stop = (plan.get("pages") or [0, 0])[:2]
    return f"plan_{int(first)}_{int(stop)}.json"


def plan_one(
    project_id: str,
    document_id: str,
    *,
    pages: tuple[int, int] | None,
    work_dir: str,
    deps: Deps,
) -> dict[str, Any]:
    """Plan one document's bill sheet; never writes a row, never raises on I/O."""
    report: dict[str, Any] = {
        "project_id": project_id,
        "document_id": document_id,
        "build_sha": deps.build_sha(),
        "requested_pages": list(pages) if pages else None,
        "ok": False,
    }
    doc = deps.get_document(document_id)
    if doc is None:
        report["error"] = ERR_NOT_FOUND
        return report
    if doc.get("project_id") != project_id:
        report["error"] = ERR_WRONG_PROJECT
        return report
    report["pages_cached_before"] = cached_pages(work_dir)
    report["work_dir_durable"] = deps.durable(work_dir)
    classify, verdict = verdict_counts(deps.classify) if deps.classify else (None, {})
    started = deps.clock()
    try:
        plan = deps.plan(
            project_id, document_id, pages=pages, classify=classify, work_dir=work_dir,
        )
    except Exception as exc:  # noqa: BLE001 - reported; saved word boxes stay for a resume
        # The message may quote the document; it stays in the private work dir.
        _write_private(work_dir, "last_error.json",
                       {"error_class": type(exc).__name__, "message": str(exc)})
        plan = {"error": ERR_RAISED, "error_class": type(exc).__name__}
    report["wall_s"] = round(deps.clock() - started, 1)
    report["peak_rss_mb"] = deps.peak_rss_mb()
    report["pages_cached_after"] = cached_pages(work_dir)
    report["verdict"] = verdict
    if "error" in plan:
        report["error"] = plan["error"]
        report["error_class"] = plan.get("error_class")
        return report
    for name in (*PLAN_FIELDS, *SOURCE_FIELDS):
        if name in plan:
            report[name] = plan[name]
    sections = plan.get("sections")
    if sections is None:  # a skip: the plan carries only its reason
        report["error"] = ERR_NO_PLAN
        return report
    not_saved = _write_private(work_dir, private_plan_name(plan), plan)
    report["private_plan_saved"] = not_saved is None
    if not_saved:  # the advisor reads the labels from that file: no file, no plan
        report["error"] = ERR_NOT_SAVED
        report["error_class"] = not_saved
        return report
    report["section_paths"] = len(sections)
    report["rows_by_path_depth"] = depth_distribution(sections)
    report.update(section_counts(sections))
    report["ok"] = True
    if deps.geometry is not None:
        first, stop = (plan.get("pages") or [0, 0])[:2]
        try:
            report["geometry"] = deps.geometry(work_dir, int(first), int(stop))
        except Exception as exc:  # noqa: BLE001 - a diagnostic never fails the plan
            report["geometry"] = {"error_class": type(exc).__name__}
    return report


def _failed(report: dict[str, Any], work_dir: str, exc: Exception) -> dict[str, Any]:
    # The message may quote the document; it stays in the private work dir.
    _write_private(work_dir, "last_error.json",
                   {"error_class": type(exc).__name__, "message": str(exc)})
    report.update(error=ERR_RAISED, error_class=type(exc).__name__)
    return report


def execute_one(
    project_id: str,
    document_id: str,
    *,
    execute_hash: str,
    gates: Gates,
    work_dir: str,
    deps: Deps,
) -> dict[str, Any]:
    """Rebuild the whole-document plan, and derive the bill sheet only when its
    hash is ``execute_hash`` and every condition in ``gates`` holds."""
    report: dict[str, Any] = {
        "mode": MODE_EXECUTE,
        "project_id": project_id,
        "document_id": document_id,
        "build_sha": deps.build_sha(),
        "execute_hash": execute_hash,
        "gates": gates.as_dict(),
        "ok": False,
    }
    doc = deps.get_document(document_id)
    if doc is None:
        report["error"] = ERR_NOT_FOUND
        return report
    if doc.get("project_id") != project_id:
        report["error"] = ERR_WRONG_PROJECT
        return report
    if deps.derive is None:
        report["error"] = ERR_BAD_EXECUTE
        return report
    report["pages_cached_before"] = cached_pages(work_dir)
    classify, verdict = verdict_counts(deps.classify) if deps.classify else (None, {})
    started = deps.clock()
    try:
        plan = deps.plan(project_id, document_id, pages=None, classify=classify,
                         work_dir=work_dir)
    except Exception as exc:  # noqa: BLE001 - reported as its class
        return _failed(report, work_dir, exc)
    report["verdict"] = verdict
    for name in (*PLAN_FIELDS, *SOURCE_FIELDS):
        if name in plan:
            report[name] = plan[name]
    refused = gate_refusals(plan, execute_hash, gates)
    report["refused"] = refused
    if refused:
        report["error"] = ERR_REFUSED
        report["wall_s"] = round(deps.clock() - started, 1)
        return report
    try:
        result = deps.derive(project_id, document_id, plan_hash=execute_hash,
                             classify=classify, work_dir=work_dir)
    except Exception as exc:  # noqa: BLE001 - reported as its class
        return _failed(report, work_dir, exc)
    report["wall_s"] = round(deps.clock() - started, 1)
    report["peak_rss_mb"] = deps.peak_rss_mb()
    if not result.get("derived"):
        report["error"] = ERR_NOT_DERIVED
        report["refused"] = [str(result.get("reason") or "")]
        return report
    new_id = str(result.get("document_id") or "")
    report["derived_document_id"] = new_id
    report["superseded"] = result.get("superseded")
    report["rows_written"] = result.get("rows")
    report["chunks_added"] = result.get("chunks")
    report["index_status"] = result.get("index_status")
    report["document_status"] = (deps.get_document(new_id) or {}).get("status")
    if deps.chunk_bounds is not None:
        try:
            report["chunks_over_window"] = deps.chunk_bounds(project_id, new_id).get("over_ceiling")
        except Exception as exc:  # noqa: BLE001 - a count never undoes the write
            report["chunks_over_window"] = None
            report["error_class"] = type(exc).__name__
    report["ok"] = True
    return report


def _document_chunk_bounds(project_id: str, document_id: str) -> dict[str, Any]:
    from app.core.chunk_bounds_plan import document_chunk_bounds
    from app.core.db import SessionLocal

    with SessionLocal() as session:
        return document_chunk_bounds(session, project_id, document_id)


def saved_geometry(work_dir: str, first: int, stop: int) -> dict[str, Any]:
    """The geometry diagnostic over the word boxes saved for ``first:stop``.
    Reads the work dir only; a page with no saved boxes is counted, not read."""
    from app.core import bill_derived, bill_geometry_diag

    words, read, missing = [], [], 0
    for index in range(first, stop):
        path = bill_derived.cached_page_path(work_dir, index)
        if not os.path.exists(path):
            missing += 1
            continue
        words.extend(bill_derived.cached_page_read(path, index).words)
        read.append(index)
    out = bill_geometry_diag.geometry_diag(words, read)
    out["pages_read"] = len(read)
    out["pages_missing"] = missing
    return out


def _peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MiB (Linux ``ru_maxrss`` is KiB)."""
    if os.name != "posix":
        return None
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def production_deps() -> Deps:
    from app.core import bill_classify, bill_derived, byte_store, projects
    from app.routers import health as _health

    return Deps(
        get_document=projects.get_document,
        plan=bill_derived.plan_bill_sheet,
        classify=bill_classify.verdict_of_reads,
        build_sha=_health._deployed_build_sha,
        peak_rss_mb=_peak_rss_mb,
        durable=byte_store.is_durable_mount,
        geometry=saved_geometry,
        derive=bill_derived.derive_bill_sheet,
        chunk_bounds=_document_chunk_bounds,
    )


def default_work_dir(document_id: str, data_dir: str | None) -> str:
    """Under ``DATA_DIR`` (the task's persistent mount), one directory per document."""
    return os.path.join((data_dir or "data").strip(), "bill_plan", document_id)


def main_from_env(
    env: Mapping[str, str] | None = None, deps: Deps | None = None,
) -> int:
    env = os.environ if env is None else env
    project_id = (env.get("BILL_PLAN_PROJECT_ID") or "").strip()
    document_id = (env.get("BILL_PLAN_DOCUMENT_ID") or "").strip()
    log = deps.log if deps else print
    if not project_id or not document_id:
        log(SUMMARY_PREFIX + json.dumps({"ok": False, "error": ERR_MISSING_IDS}))
        return 2
    try:
        pages = parse_pages(env.get("BILL_PLAN_PAGES"))
    except ValueError:
        log(SUMMARY_PREFIX + json.dumps({"ok": False, "error": ERR_BAD_PAGES}))
        return 2
    try:
        execute = parse_execute(env)
    except ValueError:
        log(SUMMARY_PREFIX + json.dumps({"ok": False, "error": ERR_BAD_EXECUTE}))
        return 2
    if execute is not None and pages is not None:  # a write is whole-document only
        log(SUMMARY_PREFIX + json.dumps({"ok": False, "error": ERR_BAD_EXECUTE}))
        return 2
    work_dir = (env.get("BILL_PLAN_WORK_DIR") or "").strip() or default_work_dir(
        document_id, env.get("DATA_DIR"))
    deps = deps or production_deps()
    if execute is not None:
        execute_hash, gates = execute
        report = execute_one(project_id, document_id, execute_hash=execute_hash,
                             gates=gates, work_dir=work_dir, deps=deps)
    else:
        report = plan_one(project_id, document_id, pages=pages, work_dir=work_dir, deps=deps)
    deps.log(SUMMARY_PREFIX + json.dumps(public_summary(report), sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main_from_env())
