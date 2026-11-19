"""The ops whitelist (scripts/ops_public_view.py) read against the REAL routes.

The exclude / include routes return plain dicts (no response model), so the
keys are taken from the routes themselves: a synthetic project (the ``world``
fixture of the exclusion tests) is planned, refused and executed through
``admin_exclude_document`` / ``admin_include_document``, and each response is
checked against the view of its kind:

* every whitelisted key occurs in a real response (a renamed route key would
  otherwise vanish from the log without a failure);
* every real key the view leaves out is one this test names as withheld, so a
  new route key must be classified before it ships;
* the filter, run as a subprocess on the real response, prints neither the
  document's file name, nor the project id, nor the stored reason.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests import test_owner_excluded_document as _owner

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ops_public_view.py"
_spec = importlib.util.spec_from_file_location("ops_public_view", _SCRIPT)
opv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opv)

# The seeded project of the exclusion tests (three documents, chunks, rows).
world = _owner.world

REASON = "dup-of-rev-b"
WRONG = "0" * 64
# Route keys the view deliberately never reads: the plan's file name and
# project id; an applied answer's echo of the whole plan.
WITHHELD = {"plan": {"plan.project_id", "plan.original_name"}, "applied": {"plan"}}
AUTH = {"role": "admin", "user_id": "u1"}


def _exclude(**kw):
    from app.routers.admin import admin_exclude_document

    return admin_exclude_document(auth=AUTH, **{"plan_hash": None, "reason": "", **kw})


def _include(**kw):
    from app.routers.admin import admin_include_document

    return admin_include_document(auth=AUTH, **{"plan_hash": None, "reason": "", **kw})


def _responses(world, kind: str) -> list[dict]:
    """Real responses of one kind, as the workflow obtains them.

    A plan kind is the execute=false answer (the workflow's plan step), for an
    eligible and an ineligible document; an applied kind is the execute=true
    answer and a refusal.
    """
    _projects, _store, _emb, pid, docs = world
    doc, other = docs["drop.txt"], docs["keep_a.txt"]
    plan = _exclude(document_id=doc, project_id=pid, execute=False)
    if kind == "exclude-plan":
        return [plan, _exclude(document_id=other, project_id=pid, execute=False)]
    done = _exclude(document_id=doc, project_id=pid, execute=True,
                    plan_hash=plan["plan"]["plan_hash"], reason=REASON)
    if kind == "exclude-applied":
        return [done, _exclude(document_id=other, project_id=pid, execute=True,
                               plan_hash=WRONG, reason=REASON)]
    iplan = _include(document_id=doc, project_id=pid, execute=False)
    if kind == "include-plan":
        return [iplan, _include(document_id=other, project_id=pid, execute=False)]
    idone = _include(document_id=doc, project_id=pid, execute=True,
                     plan_hash=iplan["plan"]["plan_hash"], reason=REASON)
    return [idone, _include(document_id=other, project_id=pid, execute=True,
                            plan_hash=WRONG, reason=REASON)]


def view_paths(view: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, sub in view.items():
        path = f"{prefix}{key}"
        out.add(path)
        if isinstance(sub, dict):
            out |= view_paths(sub, f"{path}.")
    return out


def response_paths(obj: dict, view: dict, prefix: str = "") -> set[str]:
    """Keys of a response, descending only where the view nests a dict."""
    out: set[str] = set()
    for key, val in obj.items():
        path = f"{prefix}{key}"
        out.add(path)
        if isinstance(view.get(key), dict) and isinstance(val, dict):
            out |= response_paths(val, view[key], f"{path}.")
    return out


def stamp_gaps(responses: list[dict], view: dict) -> list[str]:
    """Stamp sub-fields the view names that a real stored record lacks."""
    gaps: list[str] = []
    for key, sub in view.items():
        for r in responses:
            val = r.get(key)
            if isinstance(sub, tuple) and isinstance(val, dict):
                gaps += [f"{key}.{k}" for k in sub if k not in val]
            elif isinstance(sub, dict) and isinstance(val, dict):
                gaps += stamp_gaps([val], sub)
    return gaps


@pytest.mark.parametrize("kind", sorted(opv.VIEWS))
def test_whitelist_matches_the_real_route_responses(world, kind):
    view = opv.VIEWS[kind]
    responses = _responses(world, kind)
    real: set[str] = set()
    for r in responses:
        real |= response_paths(r, view)
    wanted = view_paths(view)
    assert wanted - real == set(), f"{kind}: whitelisted keys no route returns"
    withheld = WITHHELD[kind.split("-")[1]]
    assert real - wanted - withheld == set(), f"{kind}: unclassified route keys"
    assert stamp_gaps(responses, view) == []


@pytest.mark.parametrize("kind", sorted(opv.VIEWS))
def test_filter_on_real_responses_prints_no_name_project_or_reason(world, kind, tmp_path):
    _projects, _store, _emb, pid, _docs = world
    for i, r in enumerate(_responses(world, kind)):
        src = tmp_path / f"{kind}-{i}.json"
        src.write_text(json.dumps(r), encoding="utf-8")
        proc = subprocess.run([sys.executable, "-I", str(_SCRIPT), kind, str(src)],
                              capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        for secret in (".txt", pid, REASON):
            assert secret not in proc.stdout, (kind, i, secret)
