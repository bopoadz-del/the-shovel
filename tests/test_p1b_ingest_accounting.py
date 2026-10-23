"""TIER-1 ingest accounting, completion evidence, and abort behaviour.

These tests drive the real ``scripts/p1b_ingest_drive_server.main()`` against
an in-memory Drive and project store. They exist because the run that mattered
died at 101/1359 on Render leaving nothing that could distinguish "finished"
from "killed", while the number people were quoting as proof of completion —
``already_indexed + assigned == supported`` — is an identity that holds on
every run including that one.

The contract under test:

* completion is ``attempted == batch``, and NOTHING else;
* the terminal shard manifest is written only on completion, so its absence
  is a reliable "this run died" signal;
* an aborted run still leaves a report and a RUNSUMMARY line saying so;
* a failed chunk-count resume check aborts instead of silently redefining
  ``already_indexed`` to mean "has a row".
"""
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts import p1b_ingest_drive_server as p1b

CLIENT_FOLDER_ID = "folder-client-1"
PROJECT_ID = "example_infra_pack"
FOLDER_NAME = "the client project"



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

class _FakeStore:
    def __init__(self, counts: Dict[str, int], raises: bool = False) -> None:
        self._counts = counts
        self._raises = raises

    def count_by_doc(self, project_id: str) -> Dict[str, int]:
        if self._raises:
            raise RuntimeError("SSL connection has been closed unexpectedly")
        return dict(self._counts)


class Harness:
    """Wires main() to a synthetic corpus and exposes the written artifacts."""

    def __init__(self, tmp_path: Path, manifest_path: Path) -> None:
        self.tmp_path = tmp_path
        self.manifest_path = manifest_path
        self.evidence_dir = tmp_path / "evidence"
        self.report_path = self.evidence_dir / "p1b_server_ingestion_report.json"
        self.legacy_shard_path = tmp_path / "manifests" / "ingest_shard_0_of_1.json"
        self.durable_shard_path = self.evidence_dir / "ingest_shard_0_of_1.json"
        self.index_calls = 0

    def run(self, *argv: str) -> int:
        sys.argv = [
            "p1b_ingest_drive_server.py",
            "--tier", "1", "--resume",
            "--priority-manifest", str(self.manifest_path),
            *argv,
        ]
        return p1b.main()

    def report(self) -> Dict[str, Any]:
        return json.loads(self.report_path.read_text(encoding="utf-8"))

    def run_summaries(self, captured_stderr: str) -> List[Dict[str, Any]]:
        out = []
        for line in captured_stderr.splitlines():
            marker = " RUNSUMMARY "
            if marker in line:
                out.append(json.loads(line.split(marker, 1)[1]))
        return out


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """main() installs SIGTERM/SIGINT/SIGHUP handlers; put pytest's back."""
    saved = {}
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            saved[signum] = signal.getsignal(signum)
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Build a 40-supported / 25-already-indexed synthetic tier-1 folder."""
    monkeypatch.chdir(tmp_path)
    p1b.set_run_id("-")

    supported_count, preindexed_count, zero_chunk_rows = 40, 25, 4

    files: List[Dict[str, Any]] = [
        {
            "id": f"fid{i:03d}",
            "name": f"doc{i:03d}.pdf",
            "_drive_path": f"{FOLDER_NAME}/sub/doc{i:03d}.pdf",
            "mimeType": "application/pdf",
            "size": 1000 + i,
        }
        for i in range(supported_count)
    ]
    # Google-native files are unsupported: discovered but never assigned.
    files += [
        {
            "id": f"gid{i:03d}",
            "name": f"native{i:03d}.gdoc",
            "_drive_path": f"{FOLDER_NAME}/sub/native{i:03d}.gdoc",
            "mimeType": "application/vnd.google-apps.document",
            "size": 0,
        }
        for i in range(7)
    ]

    docs: List[Dict[str, Any]] = []
    chunk_counts: Dict[str, int] = {}
    for i in range(preindexed_count):
        docs.append({
            "id": f"doc-{i:03d}",
            "metadata": {
                "drive_file_id": f"fid{i:03d}",
                "drive_path": files[i]["_drive_path"],
                "mimeType": "application/pdf",
            },
        })
        chunk_counts[f"doc-{i:03d}"] = 5
    # Rows that exist with ZERO chunks — the retry cohort. Row-presence resume
    # would count these as done; chunk-count resume correctly retries them.
    for i in range(preindexed_count, preindexed_count + zero_chunk_rows):
        docs.append({
            "id": f"doc-{i:03d}",
            "metadata": {
                "drive_file_id": f"fid{i:03d}",
                "drive_path": files[i]["_drive_path"],
                "mimeType": "application/pdf",
            },
        })
        chunk_counts[f"doc-{i:03d}"] = 0

    manifest_path = tmp_path / "priority_manifest.json"
    manifest_path.write_text(json.dumps({
        "tiers": {
            "1": {
                "folders": [
                    {
                        "project_id": PROJECT_ID,
                        "folder_name": FOLDER_NAME,
                        "folder_id": CLIENT_FOLDER_ID,
                    },
                    # Mirrors the real manifest: a tier-1 folder with no
                    # folder_id, which the run skips entirely.
                    {
                        "project_id": "construction_3_001",
                        "folder_name": "construction-3-001",
                        "folder_id": None,
                    },
                ]
            }
        }
    }), encoding="utf-8")

    h = Harness(tmp_path, manifest_path)
    h.supported_count = supported_count
    h.preindexed_count = preindexed_count
    h.zero_chunk_rows = zero_chunk_rows
    h.expected_assigned = supported_count - preindexed_count
    h.store = _FakeStore(chunk_counts)
    h.files = files
    h.docs = docs
    h.chunk_counts = chunk_counts

    monkeypatch.setenv("P1B_EVIDENCE_DIR", str(h.evidence_dir))
    monkeypatch.setenv("P1B_PARALLELISM", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "v2")

    from app.core import doc_index, gdrive_service
    from app.core import projects as projects_mod
    from app.core.rag import embeddings as emb, vector_store as vs

    monkeypatch.setattr(gdrive_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        gdrive_service, "walk_folder", lambda fid, **kw: (list(files), []),
    )
    monkeypatch.setattr(
        gdrive_service, "download_file_bytes", lambda fid: (b"x" * 256, None),
    )
    monkeypatch.setattr(
        projects_mod, "get_or_create_project",
        lambda name, **kw: ({"id": kw.get("project_id") or name}, False),
    )
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: list(docs))
    monkeypatch.setattr(
        projects_mod, "update_document_metadata", lambda did, meta: None,
    )
    monkeypatch.setattr(
        projects_mod, "set_document_drive_md5", lambda did, token: None,
    )
    monkeypatch.setattr(
        projects_mod, "set_document_file_path", lambda did, path: None,
    )
    added = {"n": 0}

    def _add_document(**kw):
        added["n"] += 1
        return {"id": f"new-{added['n']:03d}"}

    monkeypatch.setattr(projects_mod, "add_document", _add_document)

    def _index_document(project_id, doc_id, **_k):
        h.index_calls += 1
        # A couple of realistic failures so the tally is not all-success.
        if h.index_calls % 7 == 0:
            return {"status": "ok", "rag_indexed": 0}
        if h.index_calls % 11 == 0:
            return {"status": "error", "error": "ZERO_CHUNK"}
        return {"status": "ok", "rag_indexed": 3}

    monkeypatch.setattr(doc_index, "index_document", _index_document)
    monkeypatch.setattr(vs, "get_store", lambda: h.store)
    monkeypatch.setattr(vs, "reset_store_cache", lambda: None)
    monkeypatch.setattr(emb, "reset_embedder_cache", lambda: None)
    # SQLite has no advisory locks; keep the guard out of the way except
    # where a test targets it explicitly.
    monkeypatch.setattr(
        p1b, "run_advisory_lock",
        lambda tier, project_id: __import__("contextlib").nullcontext("unsupported"),
    )
    return h


# ── the identity is not a completion signal ───────────────────────────────


def test_identity_holds_on_an_aborted_run_which_is_not_complete(harness, monkeypatch):
    """already_indexed + assigned == supported, while run_complete is False.

    This is the exact number that was read as proof TIER-1 had finished. It is
    computed at discovery time, before a single file is downloaded, so it
    cannot say anything about completion — and here it is, true, on a run that
    aborted with zero files processed.
    """
    harness.store = _FakeStore({}, raises=True)
    assert harness.run() == 1

    acc = harness.report()["accounting"]
    assert acc["already_indexed"] + acc["assigned"] == acc["supported"]
    assert harness.report()["run_complete"] is False


def test_identity_also_holds_when_complete_so_it_never_discriminates(harness):
    assert harness.run() == 0
    acc = harness.report()["accounting"]
    assert acc["already_indexed"] + acc["assigned"] == acc["supported"]
    assert harness.report()["run_complete"] is True


def test_quarantine_counted_in_run_summary(harness, monkeypatch, capfd):
    """A QUARANTINED_* index result is its own run-summary bucket."""
    from app.core import doc_index

    def _index(project_id, doc_id, **_k):
        return {
            "status": "error",
            "error": "QUARANTINED_NOT_PDF",
            "document_id": doc_id,
        }

    monkeypatch.setattr(doc_index, "index_document", _index)
    assert harness.run() == 0
    tally = harness.report()["global_tally"]
    assert tally["quarantined"] == harness.expected_assigned
    assert tally["zero_chunk"] == 0
    assert tally["errors"] == 0
    err = capfd.readouterr().err
    assert f"quarantined={harness.expected_assigned}" in err


# ── attempted reconciliation ──────────────────────────────────────────────


def test_attempted_equals_assigned_only_on_completion(harness):
    assert harness.run() == 0
    report = harness.report()
    acc = report["accounting"]

    assert acc["assigned"] == harness.expected_assigned
    assert acc["batch"] == acc["assigned"]
    assert acc["attempted"] == acc["batch"]
    assert acc["outstanding"] == 0
    assert report["run_complete"] is True
    # The per-file result list is the independent check on `attempted`.
    assert len(report["results"]) == acc["attempted"]


def test_global_tally_cannot_be_summed_to_the_attempted_count(harness):
    """skipped_unsupported carries discovery-side files, so the sum is wrong.

    Summing global_tally yields `discovered`, a number that looks like a
    legitimate total and is not one. The correct reconciliation excludes
    already_indexed and skipped_unsupported.
    """
    assert harness.run() == 0
    report = harness.report()
    tally, acc = report["global_tally"], report["accounting"]

    outcome_keys = (
        "succeeded", "zero_chunk", "skipped_too_large", "skipped_too_small",
        "skipped_unsupported", "skipped_empty", "duplicate_sha",
        "download_failed", "errors", "already_indexed",
    )
    outcome_sum = sum(tally[k] for k in outcome_keys)
    assert outcome_sum != acc["attempted"]
    assert outcome_sum == acc["discovered"]

    per_file = (
        tally["succeeded"] + tally["zero_chunk"] + tally["errors"]
        + tally["download_failed"] + tally["skipped_empty"]
        + tally["skipped_too_large"] + tally["skipped_too_small"]
    )
    assert per_file == acc["attempted"]


def test_zero_chunk_rows_are_retried_not_counted_as_already_indexed(harness):
    assert harness.run() == 0
    acc = harness.report()["accounting"]
    assert acc["already_indexed"] == harness.preindexed_count
    folder = harness.report()["folder_accounting"][0]
    assert folder["zero_chunk_retries"] == harness.zero_chunk_rows


# ── terminal shard manifest is written only on completion ─────────────────


def test_completed_run_writes_the_terminal_shard_manifest(harness):
    assert harness.run() == 0
    for path in (harness.durable_shard_path, harness.legacy_shard_path):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["run_complete"] is True
        assert manifest["exit_reason"] == "completed"
        assert manifest["attempted"] == harness.expected_assigned
        # `processed` is succeeded-only and is kept for existing readers.
        assert manifest["processed"] == manifest["global_tally"]["succeeded"]
        assert manifest["processed"] <= manifest["attempted"]


def test_aborted_run_writes_no_shard_manifest_but_does_write_a_report(
    harness, capsys,
):
    harness.store = _FakeStore({}, raises=True)
    assert harness.run() == 1

    assert not harness.durable_shard_path.exists()
    assert not harness.legacy_shard_path.exists()

    report = harness.report()
    assert report["run_complete"] is False
    assert report["complete"] is False
    assert report["exit_reason"] == "aborted_chunk_count_unavailable"

    summaries = harness.run_summaries(capsys.readouterr().err)
    assert len(summaries) == 1
    assert summaries[0]["run_complete"] is False
    assert summaries[0]["exit_reason"] == "aborted_chunk_count_unavailable"


def test_sigterm_flushes_an_aborted_report_and_no_shard_manifest(
    harness, monkeypatch, capsys,
):
    """The PID-662 signature, with evidence: killed mid-run, nothing terminal.

    Composed with #481: SIGTERM is a cooperative RunLifecycle stop (drain
    in-flight work, exit 143, ``exit_reason=signal:SIGTERM``), not an
    immediate ``os._exit``. Accounting still records the abort: attempted
    never equals batch, and the terminal shard manifest is not written.
    """
    real_drain = p1b.future_result_or_error
    state = {"n": 0}

    def _drain_then_signal(fut, file_meta=None):
        out = real_drain(fut, file_meta)
        state["n"] += 1
        if state["n"] == 5:
            signal.raise_signal(signal.SIGTERM)
        return out

    monkeypatch.setattr(p1b, "future_result_or_error", _drain_then_signal)

    assert harness.run() == 128 + signal.SIGTERM

    assert not harness.durable_shard_path.exists()
    assert not harness.legacy_shard_path.exists()

    report = harness.report()
    acc = report["accounting"]
    assert report["run_complete"] is False
    assert report["complete"] is False
    assert report["exit_reason"] == "signal:SIGTERM"
    assert 0 < acc["attempted"] < acc["batch"]
    assert acc["outstanding"] == acc["batch"] - acc["attempted"]
    # The identity survives the abort untouched — again, it proves nothing.
    assert acc["already_indexed"] + acc["assigned"] == acc["supported"]

    summaries = harness.run_summaries(capsys.readouterr().err)
    assert summaries[-1]["exit_reason"] == "signal:SIGTERM"
    assert summaries[-1]["run_complete"] is False
    assert summaries[-1]["complete"] is False


# ── fail closed on an unusable resume check ───────────────────────────────


def test_chunk_count_failure_aborts_instead_of_row_presence_resume(harness, capsys):
    """Degrading here silently reclassified 40 broken docs as already indexed."""
    harness.store = _FakeStore({}, raises=True)
    assert harness.run() == 1

    err = capsys.readouterr().err
    assert "aborting rather than" in err
    assert "falling back to row-presence resume" not in err
    # Nothing was ingested, so nothing can be mistaken for progress.
    assert harness.index_calls == 0
    assert harness.report()["accounting"]["attempted"] == 0


# ── concurrency guard ─────────────────────────────────────────────────────


def test_run_lock_conflict_aborts_before_any_ingest(harness, monkeypatch, capsys):
    def _busy(tier, project_id):
        raise p1b.RunLockUnavailable(
            f"another ingest run holds the lock for tier={tier} "
            f"project_id={project_id}"
        )

    monkeypatch.setattr(p1b, "run_advisory_lock", _busy)
    assert harness.run() == 1

    assert harness.index_calls == 0
    assert not harness.durable_shard_path.exists()
    report = harness.report()
    assert report["exit_reason"] == "aborted_run_lock_unavailable"
    assert report["run_complete"] is False
    assert "refusing to start a second concurrent run" in capsys.readouterr().err


def test_advisory_lock_keys_are_stable_signed_int32(harness):
    keys = p1b.advisory_lock_keys(1, PROJECT_ID)
    assert keys == p1b.advisory_lock_keys(1, PROJECT_ID)
    assert all(-(2 ** 31) <= k < 2 ** 31 for k in keys)
    # The namespace half is shared; the object half separates tiers/projects.
    assert p1b.advisory_lock_keys(2, PROJECT_ID)[0] == keys[0]
    assert p1b.advisory_lock_keys(2, PROJECT_ID)[1] != keys[1]
    assert p1b.advisory_lock_keys(1, "other_project")[1] != keys[1]


def test_run_lock_is_a_logged_noop_on_sqlite(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with p1b.run_advisory_lock(1, PROJECT_ID) as state:
        assert state == "unsupported"
    assert "concurrent-run protection is OFF" in capsys.readouterr().err


@pytest.mark.skipif(
    not (os.getenv("DATABASE_URL") or "").startswith(("postgres://", "postgresql")),
    reason="the real pg_try_advisory_lock round trip needs a Postgres DATABASE_URL",
)
def test_advisory_lock_is_exclusive_and_released_on_postgres():
    """The round trip CI's test-postgres job covers and SQLite cannot.

    The nested acquisition opens a second Postgres session, which is exactly
    the shape of two ingest runs racing for the same folder.
    """
    probe = "p1b_lock_probe_project"
    with p1b.run_advisory_lock(1, probe) as state:
        assert state == "acquired"
        with pytest.raises(p1b.RunLockUnavailable):
            with p1b.run_advisory_lock(1, probe):
                pass
        # A different tier is a different key and must not be blocked.
        with p1b.run_advisory_lock(2, probe) as other:
            assert other == "acquired"
    # Released on exit, so the next run can take it.
    with p1b.run_advisory_lock(1, probe) as state:
        assert state == "acquired"


# ── atomic report writes ──────────────────────────────────────────────────


def test_atomic_write_json_leaves_no_temp_file(tmp_path):
    target = tmp_path / "report.json"
    p1b.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["report.json"]


def test_atomic_write_json_failure_leaves_the_previous_file_intact(tmp_path):
    """A truncating open() would have destroyed the previous report here."""
    target = tmp_path / "report.json"
    p1b.atomic_write_json(target, {"good": True})

    class _Unserializable:
        pass

    with pytest.raises(TypeError):
        p1b.atomic_write_json(target, {"bad": _Unserializable()})

    assert json.loads(target.read_text(encoding="utf-8")) == {"good": True}
    assert [p.name for p in tmp_path.iterdir()] == ["report.json"]


def test_report_is_replaced_not_truncated_in_place(harness):
    assert harness.run() == 0
    # A completed run leaves exactly the two evidence files, no .tmp debris.
    assert sorted(p.name for p in harness.evidence_dir.iterdir()) == [
        "ingest_shard_0_of_1.json",
        "p1b_server_ingestion_report.json",
    ]


# ── run attribution and the terminal summary line ─────────────────────────


def test_every_log_line_carries_the_run_id(harness, capsys):
    assert harness.run() == 0
    run_id = harness.report()["run_id"]
    lines = [
        line for line in capsys.readouterr().err.splitlines()
        if line.startswith("[p1b-server]")
    ]
    assert lines
    assert all(line.startswith(f"[p1b-server] run_id={run_id} ") for line in lines)


def test_runsummary_is_one_terminal_machine_readable_line(harness, capsys):
    assert harness.run() == 0
    err = capsys.readouterr().err

    summaries = harness.run_summaries(err)
    assert len(summaries) == 1
    # One greppable RUNSUMMARY. RunLifecycle may log EXIT after the flush;
    # that human-readable line does not replace the machine-readable summary.
    summary_lines = [line for line in err.splitlines() if " RUNSUMMARY " in line]
    assert len(summary_lines) == 1
    assert summary_lines[0].endswith(json.dumps(
        summaries[0], separators=(",", ":"), ensure_ascii=False, sort_keys=True,
    ))

    summary = summaries[0]
    assert summary["run_complete"] is True
    assert summary["complete"] is True
    assert summary["exit_reason"] == "completed"
    assert summary["run_id"] == harness.report()["run_id"]
    assert summary["accounting"]["attempted"] == harness.expected_assigned
    assert "results" not in summary


# ── coverage caveats a completed run does not cover ───────────────────────


def test_folder_without_a_folder_id_is_recorded_not_silently_dropped(harness):
    assert harness.run() == 0
    report = harness.report()
    assert report["accounting"]["skipped_no_folder_id"] == 1
    assert report["skipped_no_folder_id_folders"] == [
        {"folder": "construction-3-001", "project_id": "construction_3_001"}
    ]


def test_walk_errors_are_counted_and_kept(harness, monkeypatch):
    from app.core import gdrive_service

    files, _ = gdrive_service.walk_folder(CLIENT_FOLDER_ID)
    monkeypatch.setattr(
        gdrive_service, "walk_folder",
        lambda fid, **kw: (files, ["gdrive walk(sub/private): 403 forbidden"]),
    )
    assert harness.run() == 0
    report = harness.report()
    assert report["accounting"]["walk_errors"] == 1
    assert report["walk_error_messages"] == [
        f"{FOLDER_NAME}: gdrive walk(sub/private): 403 forbidden"
    ]


def test_folder_accounting_records_the_denominator_before_work_starts(harness):
    assert harness.run() == 0
    folder = harness.report()["folder_accounting"][0]
    assert folder["folder"] == FOLDER_NAME
    assert folder["project_id"] == PROJECT_ID
    assert folder["supported"] == harness.supported_count
    assert folder["already_indexed"] == harness.preindexed_count
    assert folder["assigned"] == harness.expected_assigned
    assert folder["attempted"] == folder["batch"]


# ── stale-extractor .docx resume ──────────────────────────────────────────


def _add_sparse_docx(
    harness,
    *,
    n: int = 2,
    extractor_version: str | None = "pre-sdt",
    omit_fields: bool = False,
) -> None:
    """Append TEXT_SPARSE .docx rows that already have chunks."""
    from app.core import ingest_status as ist

    for i in range(n):
        fid = f"stale{i:03d}"
        path = f"{FOLDER_NAME}/sub/sparse{i:03d}.docx"
        harness.files.append({
            "id": fid,
            "name": f"sparse{i:03d}.docx",
            "_drive_path": path,
            "mimeType": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            "size": 8000 + i,
        })
        doc: Dict[str, Any] = {
            "id": f"stale-doc-{i:03d}",
            "original_name": f"sparse{i:03d}.docx",
            "metadata": {
                "drive_file_id": fid,
                "drive_path": path,
                "mimeType": (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
            },
        }
        if not omit_fields:
            doc["ingest_status"] = ist.TEXT_SPARSE
            doc["ingest_status_reason"] = "single_window:terminal"
            doc["extractor_version"] = extractor_version
        harness.docs.append(doc)
        harness.chunk_counts[doc["id"]] = 3
    harness.store = _FakeStore(harness.chunk_counts)


def test_stale_extractor_docx_with_chunks_is_not_already_indexed(
    harness, monkeypatch,
):
    """The defect: chunks>0 hid TEXT_SPARSE .docx from an older extractor."""
    from app.core import ingest_status as ist
    from app.core.rag import vector_store as vs

    _add_sparse_docx(harness, n=2, extractor_version="pre-sdt")
    monkeypatch.setattr(vs, "get_store", lambda: harness.store)
    monkeypatch.setenv("P1B_LARGEST_FIRST", "1")

    assert harness.run("--limit", "2") == 0
    report = harness.report()
    acc, tally = report["accounting"], report["global_tally"]

    assert acc["stale_extractor_open"] == 2
    assert acc["already_indexed"] == harness.preindexed_count
    assert acc["retried"] == 2
    assert acc["now_indexed"] == 2
    assert tally["stale_extractor_open"] == 2
    assert tally["retried"] == 2
    assert tally["now_indexed"] == 2
    assert report["run_complete"] is True
    # In-place: same row, no add_document for the stale cohort.
    reindexed = [
        r for r in report["results"]
        if (r.get("result") or {}).get("stale_extractor_retry")
    ]
    assert len(reindexed) == 2
    assert all(r["result"].get("reindexed_existing_doc") for r in reindexed)
    assert all(r["result"].get("ingest_status") == ist.INDEXED for r in reindexed)


def test_edited_drive_md5_is_not_already_indexed(harness, monkeypatch):
    """Same Drive id + new md5Checksum must leave the row assigned."""
    from app.core.rag import vector_store as vs

    harness.docs[0]["drive_md5"] = "old-md5"
    harness.docs[0]["ingest_status"] = "INDEXED"
    harness.files[0]["md5Checksum"] = "new-md5"
    monkeypatch.setattr(vs, "get_store", lambda: harness.store)

    assert harness.run() == 0
    acc = harness.report()["accounting"]
    assert acc["already_indexed"] == harness.preindexed_count - 1
    assert acc["assigned"] == harness.expected_assigned + 1


def test_stamped_current_extractor_docx_is_already_indexed(harness, monkeypatch):
    from app.core import ingest_status as ist
    from app.core.rag import vector_store as vs

    _add_sparse_docx(harness, n=2, extractor_version=ist.EXTRACTOR_VERSION)
    monkeypatch.setattr(vs, "get_store", lambda: harness.store)

    assert harness.run() == 0
    acc = harness.report()["accounting"]
    assert acc["stale_extractor_open"] == 0
    assert acc["retried"] == 0
    assert acc["already_indexed"] == harness.preindexed_count + 2
    assert acc["assigned"] == harness.expected_assigned


def test_stale_extractor_open_without_retry_marks_run_incomplete(
    harness, monkeypatch, capsys,
):
    """open>0 and retried==0 is RUN INCOMPLETE — the runs 1–7 gate."""
    from app.core.rag import vector_store as vs

    _add_sparse_docx(harness, n=2, extractor_version="pre-sdt")
    monkeypatch.setattr(vs, "get_store", lambda: harness.store)

    # Offset past the whole assigned set so nothing is sent to _ingest_file.
    assert harness.run("--offset", "9999") == 1

    report = harness.report()
    acc = report["accounting"]
    assert acc["stale_extractor_open"] == 2
    assert acc["retried"] == 0
    assert acc["now_indexed"] == 0
    assert report["run_complete"] is False
    assert report["complete"] is False
    assert report["exit_reason"] == "stale_extractor_not_retried"
    assert not harness.durable_shard_path.exists()

    err = capsys.readouterr().err
    assert "RUN INCOMPLETE" in err
    assert "stale_extractor_open=2" in err
    assert "retried=0" in err
    assert "now_indexed=0" in err
    summaries = harness.run_summaries(err)
    assert summaries[-1]["complete"] is False
    assert summaries[-1]["accounting"]["stale_extractor_open"] == 2
    assert summaries[-1]["accounting"]["retried"] == 0


# ── duplicate-content files are recorded, not re-downloaded ───────────────


def test_second_resume_makes_zero_downloads_for_known_duplicates(harness, monkeypatch):
    """A source file whose bytes match an existing row was downloaded, hashed
    and dropped on every pass, because nothing recorded its id. The first run
    records it against the canonical row; the second run never downloads it."""
    import hashlib

    from app.core import gdrive_service
    from app.core import projects as projects_mod

    dup_bytes = b"identical content " * 16
    dup_sha = hashlib.sha256(dup_bytes).hexdigest()
    canonical = harness.docs[0]  # already indexed, 5 chunks
    dup_fids = {f["id"] for f in harness.files[30:34]}
    for f in harness.files[30:34]:
        f["md5Checksum"] = "same-md5"
    downloads: Dict[str, int] = {}

    def _download(fid):
        downloads[fid] = downloads.get(fid, 0) + 1
        return (dup_bytes if fid in dup_fids else f"unique {fid}".encode() * 16), None

    def _find_by_sha(pid, sha):
        return canonical if sha == dup_sha else None

    def _record_alias(doc_id, source_file_id, token):
        assert doc_id == canonical["id"]
        canonical["metadata"].setdefault("source_aliases", {})[source_file_id] = token or ""

    monkeypatch.setattr(gdrive_service, "download_file_bytes", _download)
    monkeypatch.setattr(projects_mod, "find_document_by_sha", _find_by_sha)
    monkeypatch.setattr(projects_mod, "record_source_alias", _record_alias)

    harness.run()
    first = {fid: downloads.get(fid, 0) for fid in dup_fids}
    assert all(n == 1 for n in first.values()), first
    assert set(canonical["metadata"]["source_aliases"]) == dup_fids

    downloads.clear()
    harness.run()
    assert {fid: downloads.get(fid, 0) for fid in dup_fids} == dict.fromkeys(dup_fids, 0)
    acc = harness.report()["accounting"]
    assert acc["already_indexed"] >= harness.preindexed_count + len(dup_fids)


def test_duplicate_whose_source_changed_is_downloaded_again(harness, monkeypatch):
    """The alias holds only while the source's content token is unchanged."""
    from app.core import gdrive_service

    canonical = harness.docs[0]
    fid = harness.files[31]["id"]
    canonical["metadata"]["source_aliases"] = {fid: "old-md5"}
    harness.files[31]["md5Checksum"] = "new-md5"
    downloads: Dict[str, int] = {}

    def _download(f):
        downloads[f] = downloads.get(f, 0) + 1
        return f"unique {f}".encode() * 16, None

    monkeypatch.setattr(gdrive_service, "download_file_bytes", _download)
    harness.run()
    assert downloads.get(fid) == 1


def test_file_over_the_memory_budget_is_recorded_not_downloaded_then_skipped(harness, monkeypatch):
    """Admission before download: a file whose store-and-index demand cannot
    fit is never downloaded, gets a closed ledger row, and the next resume
    skips it."""
    from app.core import doc_index, gdrive_service
    from app.core import projects as projects_mod

    big = harness.files[35]
    big["size"] = 5 * 1024 * 1024
    big["md5Checksum"] = "big-md5"
    monkeypatch.setenv("DOC_INDEX_FILE_MEM_MB", "1")  # 1 MB child budget
    downloads: Dict[str, int] = {}

    def _download(fid):
        downloads[fid] = downloads.get(fid, 0) + 1
        return b"x" * 256, None

    def _add_document(**kw):
        row = {"id": f"row-{kw['metadata']['drive_file_id']}", "metadata": dict(kw["metadata"]),
               "drive_md5": kw.get("drive_md5"), "original_name": kw["original_name"]}
        harness.docs.append(row)
        return row

    def _attempt(doc_id, key):
        row = next(d for d in harness.docs if d["id"] == doc_id)
        row["metadata"]["ingest_attempts"] = {"n": 1, "key": key}
        return 1

    def _stamp(doc_id, filename, chunk_count, **kw):
        row = next(d for d in harness.docs if d["id"] == doc_id)
        row["ingest_status"] = "EXTRACT_FAILED"
        row["ingest_status_reason"] = kw.get("reason_override")

    monkeypatch.setattr(gdrive_service, "download_file_bytes", _download)
    monkeypatch.setattr(projects_mod, "add_document", _add_document)
    monkeypatch.setattr(projects_mod, "record_ingest_attempt", _attempt)
    monkeypatch.setattr(doc_index, "_stamp_index_ledger", _stamp)

    harness.run()
    assert downloads.get(big["id"], 0) == 0
    row = next(d for d in harness.docs if d["metadata"].get("drive_file_id") == big["id"])
    assert row["ingest_status"] == "EXTRACT_FAILED"
    assert row["ingest_status_reason"].startswith("memory_budget")

    downloads.clear()
    harness.run()
    assert downloads.get(big["id"], 0) == 0
    acc = harness.report()["accounting"]
    assert acc["already_indexed"] >= harness.preindexed_count + 1


# ── the RAG takes text formats only (docs/INGEST_EXCLUSION_RULE.md) ────────


def test_excluded_formats_are_never_assigned_downloaded_or_recorded(harness, monkeypatch):
    """Video, CAD and photos in the Drive folder: zero downloads, zero rows,
    never assigned on this run or on resume -- only counted by format."""
    from app.core import gdrive_service
    from app.core import projects as projects_mod

    excluded = [
        {"id": "vid001", "name": "drone.mp4", "_drive_path": f"{FOLDER_NAME}/media/drone.mp4",
         "mimeType": "video/mp4", "size": 400_000_000},
        {"id": "cad001", "name": "plan.dwg", "_drive_path": f"{FOLDER_NAME}/cad/plan.dwg",
         "mimeType": "application/acad", "size": 9_000_000},
        {"id": "img001", "name": "site.jpg", "_drive_path": f"{FOLDER_NAME}/photos/site.jpg",
         "mimeType": "image/jpeg", "size": 6_000_000},
    ]
    harness.files.extend(excluded)
    downloads: Dict[str, int] = {}
    rows: List[str] = []

    def _download(fid):
        downloads[fid] = downloads.get(fid, 0) + 1
        return b"x" * 256, None

    def _add_document(**kw):
        rows.append((kw.get("metadata") or {}).get("drive_file_id"))
        return {"id": f"new-{len(rows):03d}"}

    monkeypatch.setattr(gdrive_service, "download_file_bytes", _download)
    monkeypatch.setattr(projects_mod, "add_document", _add_document)

    for _ in range(2):  # the run, then a resume
        assert harness.run() == 0
        acc = harness.report()["accounting"]
        assert acc["assigned"] == harness.expected_assigned, acc
        assert acc["unsupported_by_format"].get(".mp4") == 1
        assert acc["unsupported_by_format"].get(".dwg") == 1
        assert acc["unsupported_by_format"].get(".jpg") == 1
    assert all(downloads.get(f["id"], 0) == 0 for f in excluded), downloads
    assert not {f["id"] for f in excluded} & set(rows), rows


def test_zero_byte_drive_files_are_never_downloaded(harness, monkeypatch):
    """Drive says 0 bytes: nothing to ingest. They used to be downloaded,
    found empty and dropped with no record, so every resume fetched them
    again. Now they are filtered at discovery and counted as (empty)."""
    from app.core import gdrive_service

    empty = [
        {"id": f"empty{i}", "name": f"placeholder{i}.pdf",
         "_drive_path": f"{FOLDER_NAME}/sub/placeholder{i}.pdf",
         "mimeType": "application/pdf", "size": "0"}
        for i in range(3)
    ]
    harness.files.extend(empty)
    downloads: Dict[str, int] = {}

    def _download(fid):
        downloads[fid] = downloads.get(fid, 0) + 1
        return b"x" * 256, None

    monkeypatch.setattr(gdrive_service, "download_file_bytes", _download)
    for _ in range(2):
        assert harness.run() == 0
        acc = harness.report()["accounting"]
        assert acc["assigned"] == harness.expected_assigned, acc
        assert acc["unsupported_by_format"].get("(empty)") == 3
    assert all(downloads.get(f["id"], 0) == 0 for f in empty), downloads
