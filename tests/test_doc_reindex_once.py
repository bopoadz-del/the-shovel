"""doc-reindex-once: one document, re-indexed from its Drive original.

Every test builds its own document rows and fakes; no corpus, no network.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from scripts import doc_reindex_once as dro


class FakeWorld:
    """A one-table document store plus a chunk store and a Drive."""

    def __init__(self, docs: Dict[str, Dict[str, Any]], drive: Dict[str, bytes],
                 index_result: Optional[Dict[str, Any]] = None,
                 new_chunks: Optional[List[str]] = None,
                 old_chunks: Optional[List[str]] = None,
                 index_raises: Optional[Exception] = None):
        self.docs = docs
        self.drive = drive
        self.index_result = index_result or {"status": "ok", "indexed": 1, "rag_indexed": 3}
        self.new_chunks = new_chunks if new_chunks is not None else []
        self.chunks: List[str] = list(old_chunks or [])
        self.index_raises = index_raises
        self.index_calls: List[tuple] = []
        self.paths_written: List[Path] = []
        self.paths_discarded: List[Path] = []
        self.file_path_seen_by_index: Optional[str] = None
        self.bytes_seen_by_index: Optional[bytes] = None
        self.lines: List[str] = []

    def deps(self, tmp_path: Path) -> dro.Deps:
        def write(raw: bytes, name: str) -> Path:
            p = tmp_path / f"t{len(self.paths_written)}_{name}"
            p.write_bytes(raw)
            self.paths_written.append(p)
            return p

        def discard(p: Path) -> None:
            Path(p).unlink(missing_ok=True)
            self.paths_discarded.append(Path(p))

        def set_fp(doc_id: str, fp: str) -> None:
            self.docs[doc_id]["file_path"] = fp

        def index(pid: str, did: str, **kw: Any) -> Dict[str, Any]:
            self.index_calls.append((pid, did, kw))
            fp = self.docs[did].get("file_path")
            self.file_path_seen_by_index = fp
            self.bytes_seen_by_index = Path(fp).read_bytes()
            if self.index_raises is not None:
                raise self.index_raises
            if self.index_result.get("status") != "error":
                self.chunks = list(self.new_chunks)
                self.docs[did]["chunk_count"] = len(self.new_chunks)
                self.docs[did]["ingest_status"] = "INDEXED"
                self.docs[did]["pages_ocrd"] = 7
            return dict(self.index_result)

        def download(fid: str):
            if fid in self.drive:
                return self.drive[fid], None
            return None, "404 not found"

        return dro.Deps(
            get_document=lambda d: dict(self.docs[d]) if d in self.docs else None,
            drive_file_id_of=lambda d: d.get("drive_file_id") or "",
            download=download,
            write_transient=write,
            discard_transient=discard,
            set_file_path=set_fp,
            index_document=index,
            chunk_texts=lambda pid, did: list(self.chunks),
            log=self.lines.append,
        )


def _doc(project: str, drive: str = "drv1", **kw: Any) -> Dict[str, Any]:
    row = {
        "project_id": project, "original_name": "Vol 1 bill.pdf",
        "drive_file_id": drive, "chunk_count": 2, "ingest_status": "TEXT_SPARSE",
        "pages_in_file": 9, "pages_ocrd": 0, "file_path": "/gone/old.pdf",
    }
    row.update(kw)
    return row


# --- the happy path: download, point, index, report before/after ---------

@pytest.mark.parametrize("force_ocr", [True, False])
def test_reindex_reports_before_after_and_passes_force_flag(tmp_path, force_ocr):
    w = FakeWorld({"d1": _doc("p1")}, {"drv1": b"%PDF-bytes"},
                  new_chunks=["HEAD A\nrow", "row", "HEAD B"], old_chunks=["x", "y"])
    rep = dro.reindex_one("p1", "d1", force_ocr=force_ocr, heading_re=r"^HEAD ", deps=w.deps(tmp_path))
    assert rep["ok"] is True
    assert rep["before"] == {"chunk_count": 2, "ingest_status": "TEXT_SPARSE", "pages_in_file": 9, "pages_ocrd": 0}
    assert rep["after"] == {"chunk_count": 3, "ingest_status": "INDEXED", "pages_in_file": 9, "pages_ocrd": 7}
    assert w.index_calls == [("p1", "d1", {"force_ocr": force_ocr, "stamp_as_indexed": True})]
    assert rep["headings"] == ["HEAD A", "HEAD B"]
    assert rep["downloaded_bytes"] == len(b"%PDF-bytes")


@pytest.mark.parametrize("payload", [b"first-bytes", b"second, longer bytes"])
def test_index_reads_the_downloaded_bytes_and_the_copy_is_deleted(tmp_path, payload):
    w = FakeWorld({"d1": _doc("p1")}, {"drv1": payload}, new_chunks=["a"])
    dro.reindex_one("p1", "d1", force_ocr=True, heading_re=None, deps=w.deps(tmp_path))
    assert w.file_path_seen_by_index == str(w.paths_written[0])
    assert w.bytes_seen_by_index == payload
    assert w.paths_discarded == w.paths_written
    assert not w.paths_written[0].exists()


# --- refusals before any download ----------------------------------------

@pytest.mark.parametrize("docs,doc_id,err", [
    ({}, "d1", "document not found"),
    ({"d1": _doc("other")}, "d1", "document does not belong to project"),
    ({"d1": _doc("p1", drive="")}, "d1", "document has no drive_file_id"),
    ({"d1": _doc("p1", drive="   ")}, "d1", "document has no drive_file_id"),
])
def test_refuses_without_touching_drive_or_index(tmp_path, docs, doc_id, err):
    w = FakeWorld(docs, {"drv1": b"bytes"})
    rep = dro.reindex_one("p1", doc_id, force_ocr=True, heading_re=None, deps=w.deps(tmp_path))
    assert rep["ok"] is False and rep["error"] == err
    assert w.index_calls == [] and w.paths_written == []


@pytest.mark.parametrize("drive", [{}, {"drv1": b""}])
def test_download_failure_never_indexes(tmp_path, drive):
    w = FakeWorld({"d1": _doc("p1")}, drive, old_chunks=["keep"])
    rep = dro.reindex_one("p1", "d1", force_ocr=True, heading_re=None, deps=w.deps(tmp_path))
    assert rep["ok"] is False and rep["error"].startswith("DOWNLOAD_FAILED")
    assert w.index_calls == []
    assert w.docs["d1"]["file_path"] == "/gone/old.pdf"


# --- a failed index keeps the old chunks and reports not-ok --------------

@pytest.mark.parametrize("result,raises", [
    ({"status": "error", "error": "OCR budget"}, None),
    (None, RuntimeError("extractor died")),
])
def test_failed_index_is_not_ok_and_old_chunks_stay(tmp_path, result, raises):
    w = FakeWorld({"d1": _doc("p1")}, {"drv1": b"b"}, index_result=result,
                  index_raises=raises, new_chunks=["new"], old_chunks=["old1", "old2"])
    rep = dro.reindex_one("p1", "d1", force_ocr=True, heading_re=None, deps=w.deps(tmp_path))
    assert rep["ok"] is False and rep["error"]
    assert w.chunks == ["old1", "old2"]
    assert rep["after"]["chunk_count"] == 2
    assert w.paths_discarded == w.paths_written and len(w.paths_written) == 1


# --- headings -------------------------------------------------------------

_HEADING_TEXTS = ["intro\n  SECTION   2  \nbody", "SECTION 1\nSECTION 2", "none"]


@pytest.mark.parametrize("pattern,want", [
    (r"^SECTION \d", ["SECTION 2", "SECTION 1"]),
    (r"^PART", []),
    (None, []),
    ("", []),
])
def test_heading_lines_order_dedupe_and_whitespace(pattern, want):
    assert dro.heading_lines(_HEADING_TEXTS, pattern) == want


# --- env entry ------------------------------------------------------------

@pytest.mark.parametrize("raw,default,want", [
    (None, True, True), ("", False, False), ("0", True, False), ("TRUE", False, True),
    ("off", True, False), ("yes", False, True),
])
def test_parse_flag(raw, default, want):
    assert dro.parse_flag(raw, default=default) is want


@pytest.mark.parametrize("raw", ["maybe", "2"])
def test_parse_flag_rejects_non_boolean(raw):
    with pytest.raises(ValueError):
        dro.parse_flag(raw, default=True)


@pytest.mark.parametrize("env", [
    {"DOC_REINDEX_PROJECT_ID": "p1"},
    {"DOC_REINDEX_DOCUMENT_ID": "d1"},
    {"DOC_REINDEX_PROJECT_ID": "p1", "DOC_REINDEX_DOCUMENT_ID": "d1", "DOC_REINDEX_FORCE_OCR": "maybe"},
    {"DOC_REINDEX_PROJECT_ID": "p1", "DOC_REINDEX_DOCUMENT_ID": "d1", "DOC_REINDEX_HEADING_RE": "(unclosed"},
])
def test_main_rejects_bad_env_before_any_io(tmp_path, env):
    w = FakeWorld({"d1": _doc("p1")}, {"drv1": b"b"})
    assert dro.main_from_env(env, deps=w.deps(tmp_path)) == 2
    assert w.index_calls == [] and w.paths_written == []


@pytest.mark.parametrize("index_result,code", [
    ({"status": "ok", "indexed": 1}, 0),
    ({"status": "error", "error": "boom"}, 1),
])
def test_main_prints_one_summary_line_and_exit_code(tmp_path, index_result, code):
    w = FakeWorld({"d1": _doc("p1")}, {"drv1": b"b"}, index_result=index_result, new_chunks=["H1"])
    env = {"DOC_REINDEX_PROJECT_ID": "p1", "DOC_REINDEX_DOCUMENT_ID": "d1",
           "DOC_REINDEX_FORCE_OCR": "false", "DOC_REINDEX_HEADING_RE": "^H"}
    assert dro.main_from_env(env, deps=w.deps(tmp_path)) == code
    summaries = [ln for ln in w.lines if ln.startswith(dro.SUMMARY_PREFIX)]
    assert len(summaries) == 1
    rep = json.loads(summaries[0][len(dro.SUMMARY_PREFIX):])
    assert rep["force_ocr"] is False and rep["ok"] is (code == 0)
    assert w.index_calls[0][2]["force_ocr"] is False


@pytest.mark.parametrize("value,hands_over", [("d1", True), ("  ", False)])
def test_p1b_main_hands_over_only_when_document_id_is_set(monkeypatch, value, hands_over):
    from scripts import p1b_ingest_drive_server as p1b

    calls: List[str] = []
    monkeypatch.setenv("DOC_REINDEX_DOCUMENT_ID", value)
    monkeypatch.setattr(dro, "main_from_env", lambda: calls.append("reindex") or 7)

    def parser_reached():
        calls.append("walk")
        raise SystemExit(0)

    monkeypatch.setattr(p1b, "build_parser", parser_reached)
    try:
        rc = p1b.main()
    except SystemExit:
        rc = None
    assert calls == (["reindex"] if hands_over else ["walk"])
    assert (rc == 7) is hands_over
