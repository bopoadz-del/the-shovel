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

This calls :func:`app.core.bill_derived.plan_bill_sheet` only. It adds no
document row, indexes nothing and writes no database row; the only bytes it
keeps are the per-page word boxes under the work directory.

The last line is ``BILLPLANSUMMARY <json>``. That line reaches a GitHub job
summary, which is world-readable, so it carries counts, codes and hashes
only: the classifier verdict and its counts, the page range, pages cached
before and after, rows, unreconciled rows, the number of section paths, the
rows per path depth, wall time, peak RSS and the plan hash. It is built by
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
#: Plan fields the summary carries as they are (all ids or counts).
PLAN_FIELDS = ("reason", "pages", "pages_in_file", "rows", "unreconciled", "plan_hash")
#: Classifier counts the summary carries.
VERDICT_FIELDS = (
    "pages_sampled", "header_pages", "measured_rows", "priced_rows", "reconciled_rows",
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
    "peak_rss_mb", "verdict", *PLAN_FIELDS, "section_paths", "rows_by_path_depth",
    "private_plan_saved", "work_dir_durable",
)
#: Error codes (the summary never carries a message).
ERR_MISSING_IDS = "missing_ids"
ERR_BAD_PAGES = "bad_pages"
ERR_NOT_FOUND = "document_not_found"
ERR_WRONG_PROJECT = "document_not_in_project"
ERR_RAISED = "plan_raised"
ERR_NO_PLAN = "no_plan"
ERR_NOT_SAVED = "plan_not_saved"


@dataclass
class Deps:
    """The I/O this run needs; production wires the real modules."""

    get_document: Callable[[str], dict[str, Any] | None]
    plan: Callable[..., dict[str, Any]]
    classify: Callable[[str], Any] | None = None
    log: Callable[[str], None] = field(default=print)
    build_sha: Callable[[], str | None] = field(default=lambda: None)
    peak_rss_mb: Callable[[], float | None] = field(default=lambda: None)
    clock: Callable[[], float] = field(default=time.monotonic)
    #: Whether the work dir is on a durable mount (the word boxes and the
    #: private plan outlive the task only then); None = not known.
    durable: Callable[[str], bool | None] = field(default=lambda _path: None)


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


def verdict_counts(classify: Callable[[str], Any]) -> tuple[Callable[[str], Any], dict[str, Any]]:
    """Wrap ``classify`` so the run can report the verdict's counts."""
    seen: dict[str, Any] = {}

    def wrapped(path: str):
        verdict = classify(path)
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
    for name in PLAN_FIELDS:
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
    report["ok"] = True
    return report


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
        classify=bill_classify.classify_image_only_bill,
        build_sha=_health._deployed_build_sha,
        peak_rss_mb=_peak_rss_mb,
        durable=byte_store.is_durable_mount,
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
    work_dir = (env.get("BILL_PLAN_WORK_DIR") or "").strip() or default_work_dir(
        document_id, env.get("DATA_DIR"))
    deps = deps or production_deps()
    report = plan_one(project_id, document_id, pages=pages, work_dir=work_dir, deps=deps)
    deps.log(SUMMARY_PREFIX + json.dumps(public_summary(report), sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main_from_env())
