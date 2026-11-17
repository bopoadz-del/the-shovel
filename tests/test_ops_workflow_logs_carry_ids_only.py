"""The owner-document ops workflows print ids, codes and counts only.

This repository is public: a workflow's run log and step summary are public
pages. The exclude / include workflows read a plan that carries the
document's original file name and free-text reasons; the run steps may drop
those keys (inside ``del(...)``) but never print them. A dispatch input is
printed in the run log, so the operator's reason is refused unless it is a
short code. The checker below is exercised on inputs built here first, then
on the committed workflow files.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
# Row keys that hold a client file name, a path or free text.
_NAME_KEYS = ("original_name", "drive_path", "file_path", "filename")
_DEL_RE = re.compile(r"del\([^)]*\)")
_REASON_GATE = "grep -Eq '^[a-z0-9_.-]{1,64}$'"


def _run_text(doc: dict) -> str:
    out = []
    for job in (doc.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            if isinstance(step.get("run"), str):
                out.append(step["run"])
    return "\n".join(out)


def printed_name_keys(run: str) -> list[str]:
    """Name-bearing keys a run script reads outside a ``del(...)``."""
    rest = _DEL_RE.sub("", run)
    return [k for k in _NAME_KEYS if re.search(r"\." + k + r"\b", rest)]


@pytest.mark.parametrize("run, leaked", [
    ("""echo "- id: $(jq -r '.plan.document_id' p.json) ($(jq -r '.plan.original_name' p.json))\"""",
     ["original_name"]),
    ("""jq '.plan | walk(if type == "object" then del(.original_name, .drive_path) else . end)' p.json""",
     []),
])
def test_checker_flags_a_printed_name_and_passes_a_deleted_one(run, leaked):
    assert printed_name_keys(run) == leaked


@pytest.mark.parametrize("name", ["document-exclude.yml", "document-include.yml"])
def test_owner_document_workflows_print_no_file_name(name):
    doc = yaml.safe_load((_WF / name).read_text(encoding="utf-8"))
    run = _run_text(doc)
    assert printed_name_keys(run) == []
    # Whole-object dumps are gone: every jq print of a plan/applied object
    # either whitelists fields or strips names first.
    assert "jq '.' " not in run
    assert "jq '.plan' " not in run
    # The reason input is printed in the run log: it must be a code.
    assert _REASON_GATE in run


@pytest.mark.parametrize("reason, ok", [
    ("duplicate_volume", True),
    ("Duplicate of Harbour Tower Vol 3", False),
])
def test_reason_gate_admits_a_code_and_refuses_free_text(reason, ok):
    pattern = re.search(r"'(\^[^']+\$)'", _REASON_GATE).group(1)
    assert bool(re.fullmatch(pattern, reason)) is ok
