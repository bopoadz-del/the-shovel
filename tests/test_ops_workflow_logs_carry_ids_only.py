"""The owner-document ops workflows publish ids, codes and counts only.

This repository is public: a workflow's run log, step summary, run page and
artifacts are public pages. The exclude / include routes answer with the
document's original file name, its paths and free-text reasons. The
workflows print a response ONLY through ``scripts/ops_public_view.py`` (a
whitelist), upload nothing, and refuse a non-code reason before any request.

Every checker is first exercised on inputs built here (one that must pass,
one that must fail), then on the committed workflow files. The filter itself
is RUN on responses built here that carry a file name, a path and free text.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WF = _ROOT / ".github" / "workflows"
_SCRIPT = _ROOT / "scripts" / "ops_public_view.py"
_WORKFLOWS = ("document-exclude.yml", "document-include.yml")

_spec = importlib.util.spec_from_file_location("ops_public_view", _SCRIPT)
opv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opv)

# What a response may carry and the public view must never show.
_NAME = "Tower B Facade Spec Rev 4.pdf"
_STEM = "Tower B Facade Spec Rev 4"
_PATH = "/data/uploads/p-77/Tower B Facade Spec Rev 4.pdf"
_DRIVE = "Clients/Harbour Works/Volume 3/Tower B Facade Spec Rev 4.pdf"
_REASON = "duplicate of the volume three facade spec sent by the consultant"
_SECRETS = (_NAME, _STEM, _PATH, _DRIVE, _REASON, "Harbour Works", "Facade")


def _stamp(reason: str) -> dict:
    return {"by": "owner", "at": "2026-11-17T21:48:00Z", "reason": reason,
            "note": _DRIVE}


def _plan(**extra) -> dict:
    plan = {
        "document_id": "doc-4f1a9c", "project_id": "p-77",
        "original_name": _NAME, "drive_path": _DRIVE, "file_path": _PATH,
        "doc_type": "specification", "ingest_status": "INDEXED",
        "ingest_status_reason": _REASON, "retrieval_visible": True,
        "chunk_count": 41, "chunk_rows": 41,
        "rows_to_delete": ["c-1", "c-2", _PATH],
        "project_chunk_total": 9120, "already_excluded": False,
        "excluded": _stamp(_REASON), "eligible": False,
        "not_eligible": f"{_NAME} carries no owner exclusion",
        "status_after": "INDEXED", "plan_hash": "9f2c0e7d" * 8,
        # A key no view names: a deny-list would print it, a whitelist never reads it.
        "source_title": _STEM, "folder": {"name": _DRIVE},
    }
    plan.update(extra)
    return plan


_RESPONSES = {
    "exclude-plan": {"plan": _plan(), "executed": False, "refused": None},
    "exclude-applied": {
        "plan": _plan(), "executed": True, "refused": f"{_NAME}: nothing written",
        "chunks_deleted": 41, "index_pruned": 41, "chunk_rows_after": 0,
        "chunk_count_after": 0, "project_chunk_total_after": 9079,
        "excluded": _stamp(_REASON), "ingest_status": "TOMBSTONED",
        "retrieval_visible": False, "original_name": _NAME,
    },
    "include-plan": {"plan": _plan(), "executed": False, "refused": None},
    "include-applied": {
        "plan": _plan(), "executed": True, "refused": None,
        "excluded_after": None,
        "included_back": dict(_stamp(_REASON), excluded=_stamp(_REASON)),
        "ingest_status": "UNVERIFIED", "retrieval_visible": True,
        "chunk_count": 0, "original_name": _NAME, "drive_path": _DRIVE,
    },
}


def _run_view(kind: str, response: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    src = tmp_path / f"{kind}.json"
    src.write_text(json.dumps(response), encoding="utf-8")
    return subprocess.run([sys.executable, "-I", str(_SCRIPT), kind, str(src)],
                          capture_output=True, text=True, check=False)


@pytest.mark.parametrize("kind", sorted(opv.VIEWS))
def test_public_view_carries_no_name_path_or_free_text(kind, tmp_path):
    assert kind in _RESPONSES
    proc = _run_view(kind, _RESPONSES[kind], tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for secret in _SECRETS:
        assert secret.lower() not in out.lower(), (kind, secret)
    assert ".pdf" not in out and "/" not in out
    # ...and it still carries what the operator needs.
    if kind.endswith("plan"):
        assert "- plan.document_id: doc-4f1a9c" in out
        assert "- plan.plan_hash: " + "9f2c0e7d" * 8 in out
    else:
        assert "- executed: true" in out


@pytest.mark.parametrize("key, value, shown", [
    ("doc_type", "specification", "specification"),
    ("doc_type", _NAME, f"<text len={len(_NAME)}>"),
    ("chunk_rows", 41, "41"),
    ("rows_to_delete", ["c-1", _PATH], "<list len=2>"),
    ("refused", "plan_hash", "<text len=9>"),   # prose fields: length even when code-like
    ("not_eligible", None, "null"),
])
def test_render_prints_codes_and_numbers_and_withholds_text(key, value, shown):
    assert opv.render(value, key) == shown


@pytest.mark.parametrize("reason, ok", [
    ("duplicate_volume", True),
    ("", True),
    (_REASON, False),
    ("spec.pdf", False),
])
def test_reason_gate_admits_a_code_and_refuses_free_text(reason, ok, tmp_path):
    proc = subprocess.run([sys.executable, "-I", str(_SCRIPT), "reason-gate", reason],
                          capture_output=True, text=True, check=False)
    assert (proc.returncode == 0) is ok
    if reason:
        assert reason not in proc.stdout + proc.stderr


# ---- workflow structure ---------------------------------------------------

def _steps(doc: dict) -> list[dict]:
    return [s for job in (doc.get("jobs") or {}).values() for s in job.get("steps") or []]


def uploads(doc: dict) -> list[str]:
    """Steps that publish a file as a run artifact."""
    return [str(s.get("name") or s["uses"]) for s in _steps(doc)
            if "upload-artifact" in str(s.get("uses", ""))]


_JQ_RE = re.compile(r"\bjq\s+(?:-\w+\s+)*(?:--arg\s+\S+\s+\S+\s+)?'([^']*)'")
# The only jq reads left: URL-encoding an input, and machine checks whose
# value is a hash, a boolean or a count (never echoed as text).
_JQ_ALLOWED = {"$v|@uri", ".plan.plan_hash", ".executed", ".chunk_rows_after", ".excluded_after"}


def stray_reads(doc: dict) -> list[str]:
    """jq filters on a response outside the allowed machine reads."""
    run = "\n".join(s["run"] for s in _steps(doc) if isinstance(s.get("run"), str))
    return [f for f in _JQ_RE.findall(run) if f not in _JQ_ALLOWED]


def gate_first(doc: dict) -> bool:
    """The first step after checkout refuses a non-code reason, unconditionally."""
    steps = [s for s in _steps(doc) if "actions/checkout" not in str(s.get("uses", ""))]
    return bool(steps) and "if" not in steps[0] and \
        "ops_public_view.py reason-gate" in str(steps[0].get("run", ""))


_BAD = {"jobs": {"j": {"steps": [
    {"name": "plan", "run": "jq '.plan' plan.json"},
    {"name": "Refuse", "run": "python3 scripts/ops_public_view.py reason-gate \"$REASON\""},
    {"name": "Upload responses", "uses": "actions/upload-artifact@v4",
     "with": {"path": "plan.json"}},
]}}}
_GOOD = {"jobs": {"j": {"steps": [
    {"uses": "actions/checkout@v4"},
    {"name": "Refuse", "run": "python3 scripts/ops_public_view.py reason-gate \"$REASON\""},
    {"name": "plan", "run": "python3 scripts/ops_public_view.py exclude-plan plan.json\n"
                            "echo \"P=$(jq -r '.plan.plan_hash' plan.json)\""},
]}}}


def test_checkers_flag_an_upload_a_raw_read_and_a_late_gate():
    assert uploads(_BAD) == ["Upload responses"] and uploads(_GOOD) == []
    assert stray_reads(_BAD) == [".plan"] and stray_reads(_GOOD) == []
    assert not gate_first(_BAD) and gate_first(_GOOD)


@pytest.mark.parametrize("name", _WORKFLOWS)
def test_owner_document_workflows_publish_the_whitelist_only(name):
    doc = yaml.safe_load((_WF / name).read_text(encoding="utf-8"))
    assert uploads(doc) == []
    assert stray_reads(doc) == []
    assert gate_first(doc)
    # The reason input is not in the job env (it would print in every step's header).
    for job in doc["jobs"].values():
        assert "REASON" not in (job.get("env") or {})
    # Every input description says it is public.
    on = doc.get("on") or doc.get(True)
    for spec in on["workflow_dispatch"]["inputs"].values():
        if spec.get("type") != "boolean":
            assert "PUBLIC" in spec["description"]


# --- #412 follow-up: a hash prints only when it has the shape of a hash ------

_HASH = "9f2c0e7d" * 8
_NOT_HASH = _HASH[:63] + " " + _NAME


@pytest.mark.parametrize("value, shown", [
    (_HASH, _HASH),                                    # a sha256 hexdigest prints
    (_NOT_HASH, f"<text len={len(_NOT_HASH)}>"),       # a hash plus a name does not
    (_DRIVE, f"<text len={len(_DRIVE)}>"),             # a path never prints
    ("", "<text len=0>"),
])
def test_hash_view_prints_a_hash_and_withholds_anything_else(value, shown):
    proc = subprocess.run([sys.executable, "-I", str(_SCRIPT), "hash", value],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0 and proc.stdout.strip() == shown


_HASH_VAR_RE = re.compile(r"\$\{?(PLANNED|EXPECTED)\b")


def raw_hash_echoes(run: str) -> list[str]:
    """Printed lines naming a plan hash without the hash filter.

    A write to ``$GITHUB_ENV`` is not printed, so it is skipped.
    """
    out = []
    for ln in run.splitlines():
        ln = ln.strip()
        if not ln.startswith("echo ") or ln.endswith('>> "$GITHUB_ENV"'):
            continue
        if _HASH_VAR_RE.search(ln) and "ops_public_view.py hash" not in ln:
            out.append(ln)
    return out


@pytest.mark.parametrize("run, bad", [
    ('echo "::error::live plan_hash=${PLANNED} != expected_plan_hash=${EXPECTED}"', 1),
    ('echo "- expected (operator): $(python3 scripts/ops_public_view.py hash "$EXPECTED")"', 0),
    ("echo \"PLANNED=$(jq -r '.plan.plan_hash' plan.json)\" >> \"$GITHUB_ENV\"", 0),
])
def test_raw_hash_echo_checker(run, bad):
    assert len(raw_hash_echoes(run)) == bad


@pytest.mark.parametrize("name", _WORKFLOWS)
def test_owner_document_workflows_print_hashes_through_the_filter(name):
    doc = yaml.safe_load((_WF / name).read_text(encoding="utf-8"))
    run = "\n".join(s["run"] for s in _steps(doc) if isinstance(s.get("run"), str))
    assert "ops_public_view.py hash" in run
    assert raw_hash_echoes(run) == []
    # The gate still compares the raw values.
    assert '[ "$PLANNED" != "$EXPECTED" ]' in run
