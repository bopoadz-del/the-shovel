"""Pin the two pull-request CI behaviours main already has.

1. A newer push to a PR cancels that PR's older run: every workflow that a
   pull_request triggers keys its concurrency group on the PR ref and its
   ``cancel-in-progress`` is true for a pull_request event.
2. A draft PR skips the heavy jobs: in the full-suite and image workflows
   every job except the change classifier is gated on the draft state
   (by its own ``if``, by an ``if`` on every one of its steps, or by needing
   a job that is gated), and the run fires again on ``ready_for_review``.

No workflow is changed here; this only reads them. A run on main (push,
schedule) is out of scope: what it cancels can only be seen after merge.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
# The workflows that carry the full suite and the image build.
HEAVY = ("test.yml", "docker-publish.yml")
# Job names that only classify the change (they must run on a draft so the
# rest can read the draft state from them).
CLASSIFIERS = {"changes"}

_EXPR = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
_CMP = re.compile(r"^github\.event_name\s*(==|!=)\s*'([^']+)'$")
_PR_KEYS = {"github.ref", "github.head_ref", "github.event.pull_request.number"}
_DRAFT = re.compile(r"pull_request\.draft|outputs\.is_draft\s*!=\s*'true'")


def _triggers(doc: dict) -> dict:
    # PyYAML reads the bare key `on` as the boolean True.
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {k: None for k in on}
    return dict(on or {})


def _eval(expr: str, event: str):
    """Evaluate the small expression forms the workflows use, for one event.

    Raises ValueError on anything else, so an unreadable policy fails the
    test instead of passing it.
    """
    expr = expr.strip()
    if expr in ("true", "false"):
        return expr == "true"
    m = _CMP.match(expr)
    if m:
        op, lit = m.groups()
        return (event == lit) if op == "==" else (event != lit)
    m = re.match(r"^(.+?)\s*&&\s*(\S+)\s*\|\|\s*(\S+)$", expr)
    if m:
        cond, a, b = m.groups()
        return a if _eval(cond, event) else b
    if re.fullmatch(r"github(\.[a-z_]+)+", expr):
        return expr
    raise ValueError(f"cannot read expression: {expr!r}")


def _value(raw, event: str):
    """A YAML scalar that may hold ${{ }} parts, resolved for one event."""
    if isinstance(raw, bool):
        return raw
    text = str(raw)
    whole = _EXPR.fullmatch(text.strip())
    if whole:
        result = _eval(whole.group(1), event)
        if isinstance(result, bool):
            return result
    return _EXPR.sub(lambda m: "{" + str(_eval(m.group(1), event)) + "}", text)


def pr_cancel_faults(doc: dict) -> list[str]:
    """Why a newer PR push would NOT cancel the older run (empty = it does)."""
    if "pull_request" not in _triggers(doc):
        return []
    conc = doc.get("concurrency")
    if not isinstance(conc, dict):
        return ["no concurrency block"]
    faults = []
    group = _value(conc.get("group", ""), "pull_request")
    keys = set(re.findall(r"\{([^}]*)\}", str(group)))
    if not keys & _PR_KEYS:
        faults.append(f"group {group!r} is not keyed on the PR")
    if keys - _PR_KEYS - {"github.workflow"}:
        faults.append(f"group {group!r} has a per-run key")
    if _value(conc.get("cancel-in-progress", False), "pull_request") is not True:
        faults.append("cancel-in-progress is not true for pull_request")
    return faults


def _gated(job: dict) -> bool:
    if _DRAFT.search(str(job.get("if", ""))):
        return True
    steps = job.get("steps") or []
    return bool(steps) and all(_DRAFT.search(str(s.get("if", ""))) for s in steps)


def draft_faults(doc: dict) -> list[str]:
    """Jobs that would still run their work on a draft PR (empty = none)."""
    trig = _triggers(doc)
    pr = trig.get("pull_request") or {}
    faults = []
    if "ready_for_review" not in (pr.get("types") or []):
        faults.append("no ready_for_review trigger: marking ready runs nothing")
    jobs = doc.get("jobs") or {}
    gated = {n for n, j in jobs.items() if _gated(j)}
    changed = True
    while changed:  # a job that needs a gated job is skipped with it
        changed = False
        for n, j in jobs.items():
            needs = j.get("needs") or []
            needs = [needs] if isinstance(needs, str) else needs
            via_needs = any(x in gated for x in needs)
            if n not in gated and via_needs and "always()" not in str(j.get("if", "")):
                gated.add(n)
                changed = True
    for n in jobs:
        if n not in CLASSIFIERS and n not in gated:
            faults.append(f"job {n} runs on a draft")
    return faults


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _pr_workflows() -> list[str]:
    return sorted(
        p.name for p in WORKFLOWS.glob("*.yml")
        if "pull_request" in _triggers(yaml.safe_load(p.read_text(encoding="utf-8")))
    )


# --- the real workflows -----------------------------------------------------

def test_there_are_pr_workflows_to_check():
    names = _pr_workflows()
    assert set(HEAVY) <= set(names)
    assert len(names) >= 4


@pytest.mark.parametrize("name", _pr_workflows())
def test_a_newer_pr_push_cancels_the_older_run(name):
    assert pr_cancel_faults(_load(name)) == []


@pytest.mark.parametrize("name", HEAVY)
def test_a_draft_pr_skips_the_heavy_jobs(name):
    assert draft_faults(_load(name)) == []


# --- the checker, on workflows built here -------------------------------------

_OK_CANCEL = {
    "on": {"pull_request": None, "push": None},
    "concurrency": {"group": "t-${{ github.ref }}", "cancel-in-progress": True},
}


@pytest.mark.parametrize("doc, expected", [
    (_OK_CANCEL, []),
    ({"on": {"pull_request": None},
      "concurrency": {"group": "t-${{ github.event.pull_request.number }}",
                      "cancel-in-progress": "${{ github.event_name != 'push' }}"}}, []),
    # a split policy: per PR on pull_request, per run elsewhere
    ({"on": {"pull_request": None},
      "concurrency": {
          "group": "t-${{ github.event_name == 'pull_request' && github.ref || github.run_id }}",
          "cancel-in-progress": "${{ github.event_name == 'pull_request' }}"}}, []),
    # a workflow no PR triggers is out of scope
    ({"on": {"push": None}}, []),
])
def test_cancel_checker_passes_kept_policies(doc, expected):
    assert pr_cancel_faults(doc) == expected


@pytest.mark.parametrize("doc, fault", [
    ({"on": {"pull_request": None}}, "no concurrency block"),
    ({"on": {"pull_request": None},
      "concurrency": {"group": "t-${{ github.ref }}", "cancel-in-progress": False}},
     "cancel-in-progress is not true"),
    ({"on": {"pull_request": None},
      "concurrency": {"group": "t-${{ github.ref }}",
                      "cancel-in-progress": "${{ github.event_name == 'push' }}"}},
     "cancel-in-progress is not true"),
    ({"on": {"pull_request": None},
      "concurrency": {"group": "t-${{ github.run_id }}", "cancel-in-progress": True}},
     "not keyed on the PR"),
    ({"on": {"pull_request": None},
      "concurrency": {"group": "t-${{ github.ref }}-${{ github.sha }}",
                      "cancel-in-progress": True}},
     "per-run key"),
    ({"on": ["pull_request"], "concurrency": "t"}, "no concurrency block"),
])
def test_cancel_checker_names_a_broken_policy(doc, fault):
    faults = pr_cancel_faults(doc)
    assert any(fault in f for f in faults), faults


@pytest.mark.parametrize("expr", [
    "${{ contains(github.ref, 'x') }}",
    "${{ github.event_name >= 'push' }}",
])
def test_cancel_checker_refuses_an_expression_it_cannot_read(expr):
    doc = {"on": {"pull_request": None},
           "concurrency": {"group": "t-${{ github.ref }}", "cancel-in-progress": expr}}
    with pytest.raises(ValueError):
        pr_cancel_faults(doc)


_RUN = {"steps": [{"run": "make"}]}
_PR = {"pull_request": {"types": ["opened", "synchronize", "ready_for_review"]}}


def _wf(jobs, on=None):
    return {"on": on or _PR, "jobs": jobs}


@pytest.mark.parametrize("doc", [
    _wf({"build": {"if": "! github.event.pull_request.draft", **_RUN},
         "health": {"needs": "build", **_RUN}}),
    _wf({"changes": _RUN,
         "test": {"needs": "changes", "steps": [
             {"if": "needs.changes.outputs.is_draft != 'true'", "run": "a"},
             {"if": "x && needs.changes.outputs.is_draft != 'true'", "run": "b"}]},
         "after": {"needs": ["changes", "test"],
                   "if": "always() && needs.changes.outputs.is_draft != 'true'", **_RUN}}),    # a classifier that is itself skipped on a draft skips what needs it
    _wf({"changes": {"if": "! github.event.pull_request.draft", **_RUN},
         "build": {"needs": "changes", **_RUN}}),
])
def test_draft_checker_passes_gated_jobs(doc):
    assert draft_faults(doc) == []


@pytest.mark.parametrize("doc, fault", [
    # one step escapes the draft gate
    (_wf({"changes": _RUN, "test": {"needs": "changes", "steps": [
        {"if": "needs.changes.outputs.is_draft != 'true'", "run": "a"},
        {"run": "b"}]}}), "job test runs on a draft"),
    # needing only the classifier does not skip a job
    (_wf({"changes": _RUN, "test": {"needs": "changes", **_RUN}}),
     "job test runs on a draft"),
    # always() runs a job even when the job it needs was skipped
    (_wf({"build": {"if": "! github.event.pull_request.draft", **_RUN},
          "report": {"needs": "build", "if": "always()", **_RUN}}),
     "job report runs on a draft"),
    # a job with no steps is not gated by "all of its steps"
    (_wf({"empty": {"steps": []}}), "job empty runs on a draft"),
    # no ready_for_review: the full run never fires when the draft is marked ready
    (_wf({"build": {"if": "! github.event.pull_request.draft", **_RUN}},
         on={"pull_request": {"types": ["opened", "synchronize"]}}),
     "no ready_for_review"),
    (_wf({"build": {"if": "! github.event.pull_request.draft", **_RUN}},
         on={"pull_request": None}),
     "no ready_for_review"),
])
def test_draft_checker_names_a_job_that_runs_on_a_draft(doc, fault):
    faults = draft_faults(doc)
    assert any(fault in f for f in faults), faults
