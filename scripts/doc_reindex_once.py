"""Re-index ONE existing document from its Drive original, inside the ingest task.

Run by ``.github/workflows/doc-reindex-once.yml`` through the ingest worker:
``p1b_ingest_drive_server.main`` hands over to :func:`main_from_env` when
``DOC_REINDEX_DOCUMENT_ID`` is set. The ingest worker's ``WORKER_COMMAND`` is
a Secrets Manager secret that beats argv and a same-name environment
override, so the selectors travel as distinct environment names:

* ``DOC_REINDEX_PROJECT_ID`` / ``DOC_REINDEX_DOCUMENT_ID`` (both required)
* ``DOC_REINDEX_FORCE_OCR`` (``1``/``true`` -> OCR empty-text pages under the
  force page budget; default on)
* ``DOC_REINDEX_HEADING_RE`` (optional regex; matching lines of the new
  chunks are listed in ``chunk_index`` order -- the caller chooses the
  pattern, nothing is committed here)

The container store keeps no copy of a Drive original, so the bytes are
downloaded by the row's ``drive_file_id`` to a transient path, the row's
``file_path`` is pointed at it, and ``index_document(force_ocr=...,
stamp_as_indexed=True)`` runs. ``index_document`` swaps chunks only on a
successful extraction (#229), so a failed run leaves the existing chunks in
place. The transient copy is deleted on every exit.

The last line is ``REINDEXSUMMARY <json>`` with the before/after
``chunk_count``, ``ingest_status``, ``pages_in_file``, ``pages_ocrd`` and the
heading lines found. A successful run with a heading pattern also stores those
lines on the document row as ``metadata.headings`` (``pattern``, ``lines``,
``at``).

The summary is one log line, and a log line has a size limit: it carries the
heading count and only the first ``DOC_REINDEX_SUMMARY_HEADINGS`` lines (each
clipped), so it always parses; the full list is the one on the row. It also
names the build that ran (``build_sha``) and the ingest profile and OCR skip
reason the index reported, so a run on an image older than the code it was
meant to test says so.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUMMARY_PREFIX = "REINDEXSUMMARY "
#: Document attribute (``metadata.headings``) holding the run's heading lines.
HEADINGS_KEY = "headings"
DOC_FIELDS = ("chunk_count", "ingest_status", "pages_in_file", "pages_ocrd")
#: ``index_document`` result fields the summary carries.
INDEX_RESULT_FIELDS = (
    "status", "error", "indexed", "rag_indexed", "ingest_profile",
    "ocr_skip_reason",
)
#: Heading lines the summary line carries by default (the row keeps them all).
DEFAULT_SUMMARY_HEADINGS = 40
#: Longest heading line, in characters, the summary line carries.
SUMMARY_HEADING_CHARS = 160
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass
class Deps:
    """The I/O this run needs; production wires the real modules."""

    get_document: Callable[[str], Optional[Dict[str, Any]]]
    drive_file_id_of: Callable[[Dict[str, Any]], str]
    download: Callable[[str], Tuple[Optional[bytes], Optional[str]]]
    write_transient: Callable[[bytes, str], Path]
    discard_transient: Callable[[Path], None]
    set_file_path: Callable[[str, str], Any]
    index_document: Callable[..., Dict[str, Any]]
    chunk_texts: Callable[[str, str], List[str]]
    update_metadata: Callable[[str, Dict[str, Any]], Any]
    log: Callable[[str], None] = field(default=print)
    build_sha: Callable[[], Optional[str]] = field(default=lambda: None)


def doc_state(doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The four fields the report compares, from one document row."""
    doc = doc or {}
    return {k: doc.get(k) for k in DOC_FIELDS}


def heading_lines(texts: List[str], pattern: Optional[str]) -> List[str]:
    """Lines of ``texts`` matching ``pattern``, first-seen order, no repeats."""
    if not pattern:
        return []
    rx = re.compile(pattern)
    seen: set[str] = set()
    out: List[str] = []
    for text in texts:
        for line in (text or "").splitlines():
            line = " ".join(line.split())
            if line and rx.search(line) and line not in seen:
                seen.add(line)
                out.append(line)
    return out


def summary_headings(lines: List[str], limit: int) -> Dict[str, Any]:
    """The bounded view of ``lines`` the summary line carries.

    ``count`` is every line found; ``lines`` is the first ``limit`` of them,
    each clipped to ``SUMMARY_HEADING_CHARS``; ``truncated`` says whether any
    line was left out.
    """
    limit = max(0, int(limit))
    kept = [
        line if len(line) <= SUMMARY_HEADING_CHARS
        else line[:SUMMARY_HEADING_CHARS - 3] + "..."
        for line in lines[:limit]
    ]
    return {"count": len(lines), "lines": kept, "truncated": len(lines) > limit}


def parse_count(raw: Optional[str], *, default: int) -> int:
    value = (raw or "").strip()
    if not value:
        return default
    count = int(value)  # ValueError on a non-number
    if count < 0:
        raise ValueError(f"not a non-negative count: {raw!r}")
    return count


def parse_flag(raw: Optional[str], *, default: bool) -> bool:
    value = (raw or "").strip().lower()
    if not value:
        return default
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"not a boolean: {raw!r}")


def reindex_one(
    project_id: str,
    document_id: str,
    *,
    force_ocr: bool,
    heading_re: Optional[str],
    deps: Deps,
    max_summary_headings: int = DEFAULT_SUMMARY_HEADINGS,
) -> Dict[str, Any]:
    """Re-index one document from its Drive original; never raises on I/O."""
    report: Dict[str, Any] = {
        "project_id": project_id,
        "document_id": document_id,
        "force_ocr": force_ocr,
        "build_sha": deps.build_sha(),
        "ok": False,
    }
    if heading_re:
        re.compile(heading_re)  # a bad pattern fails before any work
    doc = deps.get_document(document_id)
    if doc is None:
        report["error"] = "document not found"
        return report
    if doc.get("project_id") != project_id:
        report["error"] = "document does not belong to project"
        return report
    report["filename"] = doc.get("original_name") or ""
    report["before"] = doc_state(doc)
    from app.core.ingest_status import is_excluded

    if is_excluded(doc):
        report["error"] = "document is excluded by owner decision; not re-indexed"
        return report
    drive_id = (deps.drive_file_id_of(doc) or "").strip()
    if not drive_id:
        report["error"] = "document has no drive_file_id"
        return report
    raw, err = deps.download(drive_id)
    if err or not raw:
        report["error"] = f"DOWNLOAD_FAILED: {err or 'empty download'}"
        return report
    report["downloaded_bytes"] = len(raw)
    try:
        path = deps.write_transient(raw, report["filename"] or document_id)
    except Exception as exc:  # report it; nothing was pointed or indexed
        report["error"] = f"TRANSIENT_WRITE_FAILED: {type(exc).__name__}: {exc}"
        return report
    try:
        deps.set_file_path(document_id, str(path))
        result = deps.index_document(
            project_id, document_id, force_ocr=force_ocr, stamp_as_indexed=True,
        )
    except Exception as exc:  # report it; the swap keeps the old chunks
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        deps.discard_transient(path)
    report["index_result"] = {
        k: result.get(k) for k in INDEX_RESULT_FIELDS if k in result
    }
    report["after"] = doc_state(deps.get_document(document_id))
    headings = heading_lines(deps.chunk_texts(project_id, document_id), heading_re)
    bounded = summary_headings(headings, max_summary_headings)
    report["headings"] = bounded["lines"]
    report["headings_count"] = bounded["count"]
    report["headings_truncated"] = bounded["truncated"]
    report["ok"] = not result.get("error") and result.get("status") != "error"
    if not report["ok"]:
        report["error"] = result.get("error") or "index_document reported an error"
    report["headings_stored"] = False
    if report["ok"] and heading_re:
        # The row keeps what this run found in its own new chunks, with the
        # pattern that found it; a failed run kept the old chunks, so it
        # writes nothing.
        deps.update_metadata(document_id, {
            HEADINGS_KEY: {
                "pattern": heading_re,
                "lines": list(headings),
                "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        })
        report["headings_stored"] = True
    return report


def write_transient_copy(raw: bytes, name: str) -> Path:
    """Write ``raw`` to one transient copy the extractors can read.

    The path comes from the ingest worker's own reservation
    (``_transient_dest``): decrypted-temp prefix, so the plaintext sweeper
    reaps it if the process dies, and the stored name keeps the extension the
    extractor dispatches on. A failed write leaves no file behind.
    """
    from scripts import p1b_ingest_drive_server as p1b

    dest = p1b._transient_dest(p1b._safe_stored_name(name))  # noqa: SLF001
    try:
        dest.write_bytes(raw)
    except BaseException:
        p1b._discard_transient_copy(dest)  # noqa: SLF001
        raise
    return dest


def production_deps() -> Deps:
    from app.core import doc_index, gdrive_service, projects as projects_mod
    from app.core.rag import vector_store as _vs
    from app.routers import health as _health
    from scripts import p1b_ingest_drive_server as p1b

    def _chunk_texts(project_id: str, document_id: str) -> List[str]:
        return _vs.get_store().doc_chunk_texts(project_id, [document_id]).get(document_id, [])

    return Deps(
        get_document=projects_mod.get_document,
        drive_file_id_of=lambda d: projects_mod.extract_document_source_pointers(d)["drive_file_id"],
        download=gdrive_service.download_file_bytes,
        write_transient=write_transient_copy,
        discard_transient=p1b._discard_transient_copy,  # noqa: SLF001
        set_file_path=projects_mod.set_document_file_path,
        # The tool exists to read what the upload gates refuse (a 221 MB
        # drawing volume): it indexes on the corpus_build profile.
        index_document=functools.partial(
            doc_index.index_document, profile=doc_index.PROFILE_CORPUS_BUILD,
        ),
        chunk_texts=_chunk_texts,
        update_metadata=projects_mod.update_document_metadata,
        log=p1b.log,
        build_sha=_health._deployed_build_sha,  # noqa: SLF001
    )


def main_from_env(env: Optional[Dict[str, str]] = None, deps: Optional[Deps] = None) -> int:
    env = dict(os.environ if env is None else env)
    project_id = (env.get("DOC_REINDEX_PROJECT_ID") or "").strip()
    document_id = (env.get("DOC_REINDEX_DOCUMENT_ID") or "").strip()
    out = deps.log if deps else print
    if not project_id or not document_id:
        out("ERROR: DOC_REINDEX_PROJECT_ID and DOC_REINDEX_DOCUMENT_ID are both required")
        return 2
    try:
        force_ocr = parse_flag(env.get("DOC_REINDEX_FORCE_OCR"), default=True)
        heading_re = env.get("DOC_REINDEX_HEADING_RE") or None
        if heading_re:
            re.compile(heading_re)
        max_headings = parse_count(
            env.get("DOC_REINDEX_SUMMARY_HEADINGS"), default=DEFAULT_SUMMARY_HEADINGS,
        )
    except (ValueError, re.error) as exc:
        out(f"ERROR: {exc}")
        return 2
    deps = deps or production_deps()
    report = reindex_one(
        project_id, document_id, force_ocr=force_ocr, heading_re=heading_re, deps=deps,
        max_summary_headings=max_headings,
    )
    deps.log(SUMMARY_PREFIX + json.dumps(report, sort_keys=True, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main_from_env())
