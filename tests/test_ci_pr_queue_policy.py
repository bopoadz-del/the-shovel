"""Pin the two pull-request CI behaviours main already has.

1. A newer push to a PR cancels that PR's older run: every workflow that a
   pull_request triggers keys its concurrency group on the PR ref and its
   ``cancel-in-progress`` is true for a pull_request event.
2. A draft PR skips the heavy jobs, and a ready PR runs them. In the
   full-suite and image workflows each job's ``if`` (and each step's ``if``)
   is evaluated the way GitHub evaluates it, once for a draft pull_request
   and once for a ready one:
   - a job whose ``if`` has no status function needs every job it needs to
     have succeeded; a job whose steps are all skipped still succeeds, so a
     job that needs it runs;
   - ``always()`` and ``!cancelled()`` run a job after a skipped need;
   - a value an ``if`` reads from a job's outputs is found by running that
     job's own step script for the scenario, not by trusting its name.
   On a draft, a job that does work is a fault unless another job's ``if``
   reads its outputs (it classifies the change). On a ready PR, a job that
   does no work is a fault. The run must also fire again on
   ``ready_for_review``.

No workflow is changed here; this only reads them (and runs a classifier
step's own shell script with a scratch output file). A run on main (push,
schedule) is out of scope: what it cancels can only be seen after merge.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
# The workflows that carry the full suite and the image build.
HEAVY = ("test.yml", "docker-publish.yml")

_EXPR = re.compile(r"\$\{\{\s*(.*?)\s*\}\}", re.DOTALL)
_CMP = re.compile(r"^github\.event_name\s*(==|!=)\s*'([^']+)'$")
_PR_KEYS = {"github.ref", "github.head_ref", "github.event.pull_request.number"}


def _triggers(doc: dict) -> dict:
    # PyYAML reads the bare key `on` as the boolean True.
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {k: None for k in on}
    return dict(on or {})


def _eval(expr: str, event: str):
    """Evaluate the small expression forms the concurrency blocks use.

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


# --- GitHub `if` expressions, three-valued ----------------------------------
# UNKNOWN is a value the scenario does not fix (a matrix cell, a SHA). An
# `if` counts as false only when it is false whatever UNKNOWN turns out to be.

class _Unknown:
    def __repr__(self):
        return "UNKNOWN"


UNKNOWN = _Unknown()
_STATUS = ("always", "success", "failure", "cancelled")
_STATUS_CALL = re.compile(r"\b(?:always|success|failure|cancelled)\s*\(")
_TOKEN = re.compile(
    r"(?P<str>'(?:[^']|'')*')"
    r"|(?P<num>\d+(?:\.\d+)?)"
    r"|(?P<op>&&|\|\||==|!=|<=|>=|<|>|!|\(|\)|,)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_\-]*(?:\.[A-Za-z_*][A-Za-z0-9_\-]*)*)"
)
_RUNNING = {"always": True, "success": True, "failure": False, "cancelled": False}


def _tokens(expr: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _TOKEN.match(expr, pos)
        if not m:
            raise ValueError(f"cannot read expression: {expr!r}")
        out.append((m.lastgroup, m.group(m.lastgroup)))
        pos = m.end()
    return out


def _truthy(v):
    if v is UNKNOWN:
        return UNKNOWN
    if isinstance(v, float) and math.isnan(v):
        return False
    return bool(v)


def _num(v):
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = v.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return math.nan


def _equal(a, b):
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    if type(a) is type(b):
        return a == b
    return _num(a) == _num(b)


class _Parser:
    """`||` < `&&` < `==`/`!=` < `!` < primary, as GitHub orders them."""

    def __init__(self, expr: str, ctx: dict, status: dict):
        self.expr, self.toks, self.i = expr, _tokens(expr), 0
        self.ctx, self.status = ctx, status

    def _bad(self):
        return ValueError(f"cannot read expression: {self.expr!r}")

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def _take(self, value=None):
        tok = self._peek()
        if tok[0] is None or (value is not None and tok[1] != value):
            raise self._bad()
        self.i += 1
        return tok

    def parse(self):
        v = self._or()
        if self.i != len(self.toks):
            raise self._bad()
        return v

    def _or(self):
        v = self._and()
        while self._peek()[1] == "||":
            self._take()
            w = self._and()
            tv, tw = _truthy(v), _truthy(w)
            if tv is True:
                continue
            v = w if (tw is True or tv is False) else UNKNOWN
        return v

    def _and(self):
        v = self._cmp()
        while self._peek()[1] == "&&":
            self._take()
            w = self._cmp()
            tv, tw = _truthy(v), _truthy(w)
            if tv is False:
                continue
            v = w if (tw is False or tv is True) else UNKNOWN
        return v

    def _cmp(self):
        v = self._not()
        while self._peek()[1] in ("==", "!=", "<", ">", "<=", ">="):
            op = self._take()[1]
            if op not in ("==", "!="):
                raise self._bad()
            eq = _equal(v, self._not())
            v = eq if (op == "==" or eq is UNKNOWN) else (not eq)
        return v

    def _not(self):
        if self._peek()[1] == "!":
            self._take()
            t = _truthy(self._not())
            return UNKNOWN if t is UNKNOWN else (not t)
        return self._primary()

    def _primary(self):
        kind, text = self._take()
        if text == "(":
            v = self._or()
            self._take(")")
            return v
        if kind == "str":
            return text[1:-1].replace("''", "'")
        if kind == "num":
            return float(text)
        if kind == "name":
            if self._peek()[1] == "(":
                self._take("(")
                self._take(")")
                if text not in _STATUS:
                    raise self._bad()
                return self.status[text]
            if text in ("true", "false"):
                return text == "true"
            if text == "null":
                return None
            return self.ctx.get(text, UNKNOWN)
        raise self._bad()


def _if_text(raw) -> str:
    if isinstance(raw, bool):
        return "true" if raw else "false"
    text = str(raw).strip()
    whole = _EXPR.fullmatch(text)
    return whole.group(1) if whole else text


def _if_value(raw, ctx: dict, status: dict):
    """An `if:` as GitHub runs it: without a status function it is
    `success() && (if)`."""
    if raw is None or str(raw).strip() == "":
        return status["success"]
    text = _if_text(raw)
    v = _truthy(_Parser(text, ctx, status).parse())
    if _STATUS_CALL.search(text):
        return v
    s = status["success"]
    if v is False or s is False:
        return False
    return True if (v is True and s is True) else UNKNOWN


def _render(text, ctx: dict) -> str:
    def one(m):
        v = _Parser(m.group(1), ctx, _RUNNING).parse()
        if v is UNKNOWN or v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)
    return _EXPR.sub(one, str(text))


def _step_outputs(step: dict, ctx: dict) -> dict:
    """Run one step's own shell script for the scenario; its $GITHUB_OUTPUT."""
    bash = shutil.which("bash")
    assert bash, "bash is needed to read a classifier step"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.write_text("", encoding="utf-8")
        script = Path(tmp) / "step.sh"
        script.write_text(_render(step.get("run", ""), ctx), encoding="utf-8", newline="\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
        env.update({str(k): _render(v, ctx) for k, v in (step.get("env") or {}).items()})
        env["GITHUB_OUTPUT"] = out.as_posix()
        env["GITHUB_ENV"] = (Path(tmp) / "env").as_posix()
        subprocess.run(
            [bash, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()],
            cwd=REPO, env=env, capture_output=True, timeout=60, check=True,
        )
        lines = out.read_text(encoding="utf-8").splitlines()
        return dict(line.split("=", 1) for line in lines if "=" in line)


def _job_outputs(job: dict, ctx: dict) -> dict:
    steps = {s.get("id"): s for s in job.get("steps") or [] if s.get("id")}
    ran: dict = {}
    found = {}
    for key, raw in (job.get("outputs") or {}).items():
        m = re.fullmatch(r"\s*\$\{\{\s*steps\.([\w-]+)\.outputs\.([\w-]+)\s*\}\}\s*", str(raw))
        if m and m.group(1) in steps:
            sid = m.group(1)
            if sid not in ran:
                ran[sid] = _step_outputs(steps[sid], ctx)
            found[key] = ran[sid].get(m.group(2), "")
        elif m:
            found[key] = ""
        else:
            found[key] = _render(raw, ctx)
    return found


def _needs(job: dict) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def _run_jobs(doc: dict, draft: bool) -> dict:
    """Each job's state for one pull_request scenario: `skipped`, `inert`
    (it runs, every step is skipped, it succeeds) or `works`."""
    jobs = doc.get("jobs") or {}
    base = {"github.event_name": "pull_request", "github.event.pull_request.draft": draft}
    state: dict = {}
    pending = list(jobs)
    while pending:
        ready = [n for n in pending
                 if all(x in state for x in _needs(jobs[n]) if x in jobs)]
        if not ready:
            raise ValueError(f"needs cannot be resolved: {pending}")
        for name in ready:
            job = jobs[name]
            needs = _needs(job)
            ctx = dict(base)
            for n in needs:
                ran = state.get(n) in ("inert", "works")
                if ran:
                    for k, v in _job_outputs(jobs[n], base).items():
                        ctx[f"needs.{n}.outputs.{k}"] = v
                ctx[f"needs.{n}.result"] = "success" if ran else "skipped"
            status = dict(_RUNNING, success=all(state.get(n) != "skipped" for n in needs))
            if _if_value(job.get("if"), ctx, status) is False:
                state[name] = "skipped"
            else:
                steps = job.get("steps") or []
                live = [s for s in steps if _if_value(s.get("if"), ctx, _RUNNING) is not False]
                state[name] = "works" if (live or not steps) else "inert"
            pending.remove(name)
    return state


def _classifiers(doc: dict) -> set[str]:
    """Jobs whose outputs another job reads through `needs`."""
    text = yaml.safe_dump(doc.get("jobs") or {})
    return set(re.findall(r"needs\.([\w-]+)\.outputs\.", text))


def draft_faults(doc: dict) -> list[str]:
    """Jobs that do work on a draft PR or none on a ready one (empty = none)."""
    trig = _triggers(doc)
    pr = trig.get("pull_request") or {}
    faults = []
    if "ready_for_review" not in (pr.get("types") or []):
        faults.append("no ready_for_review trigger: marking ready runs nothing")
    classifiers = _classifiers(doc)
    for name, st in _run_jobs(doc, draft=True).items():
        if st == "works" and name not in classifiers:
            faults.append(f"job {name} runs on a draft")
    for name, st in _run_jobs(doc, draft=False).items():
        if st != "works":
            faults.append(f"job {name} does no work on a ready PR")
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


@pytest.mark.parametrize("on, in_scope", [("pull_request", True), ("push", False)])
def test_a_string_trigger_is_read(on, in_scope):
    assert _triggers({"on": on}) == {on: None}
    expected = ["no concurrency block"] if in_scope else []
    assert pr_cancel_faults({"on": on}) == expected


@pytest.mark.parametrize("cancel", [
    "${{ github.event_name == 'pull_request' && github.ref || github.sha }}",
    "${{ github.ref }}",
])
def test_a_cancel_value_that_is_not_a_bool_is_a_fault(cancel):
    doc = {"on": {"pull_request": None},
           "concurrency": {"group": "t-${{ github.ref }}", "cancel-in-progress": cancel}}
    assert "cancel-in-progress is not true for pull_request" in pr_cancel_faults(doc)


_RUN = {"steps": [{"run": "make"}]}
_PR = {"pull_request": {"types": ["opened", "synchronize", "ready_for_review"]}}
_GATE = "needs.changes.outputs.is_draft != 'true'"


def _wf(jobs, on=None):
    return {"on": on or _PR, "jobs": jobs}


def _classifier(run, env=None, out="${{ steps.d.outputs.is_draft }}"):
    step = {"id": "d", "run": run}
    if env:
        step["env"] = env
    return {"outputs": {"is_draft": out, "docs_only": "${{ steps.d.outputs.docs_only }}"},
            "steps": [{"uses": "actions/checkout@v4"}, step]}


# Two different ways a classifier can derive the draft state from the event.
_BY_IF = _classifier(
    'echo "docs_only=false" >> "$GITHUB_OUTPUT"\n'
    'if [ "${{ github.event_name }}" = "pull_request" ] && '
    '[ "${{ github.event.pull_request.draft }}" = "true" ]; then\n'
    '  echo "is_draft=true" >> "$GITHUB_OUTPUT"\nelse\n'
    '  echo "is_draft=false" >> "$GITHUB_OUTPUT"\nfi\n')
_BY_ENV = _classifier(
    'echo "docs_only=false" >> "$GITHUB_OUTPUT"\necho "is_draft=$D" >> "$GITHUB_OUTPUT"\n',
    env={"D": "${{ github.event.pull_request.draft }}"})


@pytest.mark.parametrize("cls", [_BY_IF, _BY_ENV], ids=["by-if", "by-env"])
def test_a_classifier_reports_the_draft_state_it_reads(cls):
    on_draft = _job_outputs(cls, {"github.event_name": "pull_request",
                                  "github.event.pull_request.draft": True})
    on_ready = _job_outputs(cls, {"github.event_name": "pull_request",
                                  "github.event.pull_request.draft": False})
    assert on_draft["is_draft"] == "true"
    assert on_ready["is_draft"] == "false"


@pytest.mark.parametrize("doc", [
    _wf({"build": {"if": "! github.event.pull_request.draft", **_RUN},
         "health": {"needs": "build", **_RUN}}),
    _wf({"build": {"if": "github.event_name != 'pull_request' || ! github.event.pull_request.draft",
                   **_RUN},
         "health": {"needs": "build", "steps": [{"run": "a"}, {"if": "always()", "run": "b"}]}}),
    _wf({"changes": _BY_IF,
         "test": {"needs": "changes", "steps": [
             {"if": _GATE, "run": "a"},
             {"if": f"x && {_GATE}", "run": "b"}]},
         "after": {"needs": ["changes", "test"], "if": f"always() && {_GATE}", **_RUN}}),
    _wf({"changes": _BY_ENV,
         "test": {"needs": "changes", "steps": [
             {"if": f"needs.changes.outputs.docs_only != 'true' && {_GATE}", "run": "a"}]},
         "slow": {"needs": "changes", "if": _GATE, **_RUN},
         "after": {"needs": ["changes", "test", "slow"], "if": f"always() && {_GATE}", **_RUN}}),
    # an `if` with no status function still needs its needs to succeed
    _wf({"build": {"if": "! github.event.pull_request.draft", **_RUN},
         "report": {"needs": "build", "if": "github.event_name == 'pull_request'", **_RUN}}),
    # success() is false after a skipped need, and so is failure()
    _wf({"build": {"if": "! github.event.pull_request.draft", **_RUN},
         "report": {"needs": "build", "if": "success() || failure()", **_RUN}}),
    # a classifier that is itself skipped on a draft skips what needs it
    _wf({"changes": {"if": "! github.event.pull_request.draft", **_RUN},
         "build": {"needs": "changes", **_RUN}}),
])
def test_draft_checker_passes_gated_jobs(doc):
    assert draft_faults(doc) == []


_SKIPPED_BUILD = {"if": "! github.event.pull_request.draft", **_RUN}


@pytest.mark.parametrize("doc, fault", [
    # one step escapes the draft gate
    (_wf({"changes": _BY_IF, "test": {"needs": "changes", "steps": [
        {"if": _GATE, "run": "a"}, {"run": "b"}]}}), "job test runs on a draft"),
    # needing only the classifier does not skip a job
    (_wf({"changes": _BY_IF, "test": {"needs": "changes", **_RUN}}),
     "job test runs on a draft"),
    # always(), !cancelled() and !failure() && !cancelled() run after a skipped need
    (_wf({"build": _SKIPPED_BUILD, "report": {"needs": "build", "if": "always()", **_RUN}}),
     "job report runs on a draft"),
    (_wf({"build": _SKIPPED_BUILD, "report": {"needs": "build", "if": "${{ !cancelled() }}", **_RUN}}),
     "job report runs on a draft"),
    (_wf({"build": _SKIPPED_BUILD, "report": {"needs": "build", "if": "!failure() && !cancelled()", **_RUN}}),
     "job report runs on a draft"),
    # a job gated only per step still succeeds, so a job that needs it runs
    (_wf({"changes": _BY_IF,
          "test": {"needs": "changes", "steps": [{"if": _GATE, "run": "a"}]},
          "heavy": {"needs": "test", **_RUN}}), "job heavy runs on a draft"),
    # a condition that names the draft field but runs ONLY on drafts
    (_wf({"build": {"if": "github.event.pull_request.draft", **_RUN}}), "job build runs on a draft"),
    (_wf({"build": {"if": "github.event.pull_request.draft", **_RUN}}),
     "job build does no work on a ready PR"),
    # a gate made void by an `||`
    (_wf({"changes": _BY_IF, "test": {"needs": "changes", "steps": [
        {"if": f"always() || {_GATE}", "run": "a"}]}}), "job test runs on a draft"),
    (_wf({"changes": _BY_ENV, "slow": {"needs": "changes",
          "if": f"needs.changes.outputs.docs_only != 'true' || {_GATE}", **_RUN}}),
     "job slow runs on a draft"),
    # a value the scenario does not fix may run the step
    (_wf({"changes": _BY_IF, "test": {"needs": "changes", "steps": [
        {"if": f"matrix.full == 'yes' || {_GATE}", "run": "a"}]}}), "job test runs on a draft"),
    (_wf({"changes": _BY_IF, "test": {"needs": "changes", "steps": [
        {"if": "!(matrix.full == 'yes' && needs.changes.outputs.is_draft == 'true')", "run": "a"}]}}),
     "job test runs on a draft"),
    # a classifier that never reports a draft gates nothing
    (_wf({"changes": _classifier('echo "is_draft=false" >> "$GITHUB_OUTPUT"'),
          "slow": {"needs": "changes", "if": _GATE, **_RUN}}), "job slow runs on a draft"),
    (_wf({"changes": _classifier(_BY_IF["steps"][1]["run"], out="false"),
          "slow": {"needs": "changes", "if": _GATE, **_RUN}}), "job slow runs on a draft"),
    # a job with no steps is not gated by "all of its steps"
    (_wf({"empty": {"steps": []}}), "job empty runs on a draft"),
    # a job that never runs on a ready PR
    (_wf({"build": {"if": "false", **_RUN}}), "job build does no work on a ready PR"),
    (_wf({"build": {"steps": [{"if": "${{ false }}", "run": "a"}]}}),
     "job build does no work on a ready PR"),
    # no ready_for_review: the full run never fires when the draft is marked ready
    (_wf({"build": _SKIPPED_BUILD}, on={"pull_request": {"types": ["opened", "synchronize"]}}),
     "no ready_for_review"),
    (_wf({"build": _SKIPPED_BUILD}, on={"pull_request": None}), "no ready_for_review"),
])
def test_draft_checker_names_a_job_that_runs_on_a_draft(doc, fault):
    faults = draft_faults(doc)
    assert any(fault in f for f in faults), faults


@pytest.mark.parametrize("cond", [
    "contains(github.ref, 'x')",
    "github.run_number >= 3",
])
def test_draft_checker_refuses_an_expression_it_cannot_read(cond):
    with pytest.raises(ValueError):
        draft_faults(_wf({"build": {"if": cond, **_RUN}}))


# One-line edits to the real workflows that make heavy work run on a draft.
# Each must turn the pin red (the edited text must still be in the file).
@pytest.mark.parametrize("name, old, new", [
    ("test.yml", 'echo "is_draft=true"', 'echo "is_draft=false"'),
    ("test.yml", "is_draft: ${{ steps.diff.outputs.is_draft }}", "is_draft: 'false'"),
    ("docker-publish.yml", "|| ! github.event.pull_request.draft", "|| github.event.pull_request.draft"),
    ("docker-publish.yml", "|| ! github.event.pull_request.draft", "|| ! github.event.pull_request.draft || true"),
    ("test.yml", "    if: needs.changes.outputs.docs_only != 'true' && needs.changes.outputs.is_draft != 'true'\n",
     "    if: needs.changes.outputs.docs_only != 'true' || needs.changes.outputs.is_draft != 'true'\n"),
    ("docker-publish.yml", "    needs: build\n", "    needs: build\n    if: ${{ !cancelled() }}\n"),
    ("test.yml", "\n  test-timing:\n",
     ("\n  heavy-report:\n    needs: test\n    runs-on: ubuntu-latest\n"
      "    steps:\n      - run: pytest\n\n  test-timing:\n")),
])
def test_an_edit_that_runs_heavy_work_on_a_draft_turns_the_pin_red(name, old, new):
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert old in text
    assert draft_faults(yaml.safe_load(text)) == []
    assert draft_faults(yaml.safe_load(text.replace(old, new, 1))) != []
