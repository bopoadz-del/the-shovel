# Reasoning Engine — Plan 3: Sandbox

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.
> Part of the Reasoning Engine — see `2026-05-20-reasoning-engine-INDEX.md`. Independent of Plans 1/1b/2.

**Goal:** A jail in which the reasoning engine can run Python it did **not** write — LLM-generated formula code from Plan 4 — without that code being able to crash the host, touch the filesystem, open sockets, or import anything dangerous. The sandbox compiles untrusted source with RestrictedPython, execs it in a namespace stripped of unsafe builtins, allows imports only from a small whitelist, injects per-session state as variables, captures `print` output plus a designated result variable, and always hands back a structured `SandboxResult` — sandboxed code may *fail*, but it can never *escape*.

**Architecture:** `app/core/sandbox.py` holds the whole subsystem — no schemas package change. The public surface is one function, `run_sandboxed(code, state, result_var) -> SandboxResult`, plus the `SandboxResult` Pydantic model and two policy constants (`ALLOWED_MODULES`, `BLOCKED_BUILTINS`). Plan 4's `formula_executor_v2` is the only caller: it generates code, calls `run_sandboxed`, and retries on `SandboxResult.success is False`.

**Tech Stack:** Python 3.11, Pydantic v2, `RestrictedPython` 8.x. RestrictedPython rewrites the AST at compile time (banning `eval`/`exec`/`__import__`/attribute tricks) and supplies `safe_builtins`; this plan layers a whitelisted `__import__` and session-state injection on top. This file is `app/core/sandbox.py` — **not** the unrelated legacy `app/blocks/sandbox.py` block.

**Run tests:** `& .venv\Scripts\python.exe -m pytest <path> -q` from `C:\Users\example\The_Fork`.

---

### Task 1: Add the RestrictedPython dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency line**

In `requirements.txt`, add a section after the Construction blocks:

```
# Reasoning Engine — sandboxed code execution (Plan 3)
RestrictedPython>=7.0
```

- [ ] **Step 2: Install it into the venv**

Run: `& .venv\Scripts\python.exe -m pip install RestrictedPython`
Expected: `Successfully installed RestrictedPython-8.x` (8.1 at time of writing).

- [ ] **Step 3: Verify the import**

Run: `& .venv\Scripts\python.exe -c "from RestrictedPython import compile_restricted, safe_builtins; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build(deps): add RestrictedPython for the sandbox (reasoning engine plan 3)"
```

---

### Task 2: SandboxResult model + successful exec

**Files:**
- Create: `app/core/sandbox.py`
- Test: `tests/test_sandbox.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox.py`:

```python
"""Tests for the RestrictedPython sandbox — Reasoning Engine Plan 3."""

import pytest

from app.core.sandbox import (
    ALLOWED_MODULES,
    BLOCKED_BUILTINS,
    SandboxResult,
    run_sandboxed,
)


def test_successful_exec_returns_structured_result():
    r = run_sandboxed("result = 2 + 3")
    assert isinstance(r, SandboxResult)
    assert r.success is True
    assert r.result == 5
    assert r.error is None and r.error_type is None


def test_allowed_module_import_works():
    r = run_sandboxed("import math\nresult = math.sqrt(144)")
    assert r.success is True
    assert r.result == 12.0


def test_pm_computations_is_importable():
    code = (
        "from app.lib.pm_computations import compute_cpm\n"
        "result = compute_cpm is not None"
    )
    r = run_sandboxed(code)
    assert r.success is True and r.result is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_sandbox.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.sandbox'`

- [ ] **Step 3: Write the sandbox module**

Create `app/core/sandbox.py`. It defines the policy constants, the
`SandboxResult` Pydantic model, a whitelisted `__import__` factory, the
restricted-globals builder, and `run_sandboxed`:

```python
"""RestrictedPython sandbox — Reasoning Engine Plan 3.

Executes untrusted, LLM-generated Python in a jailed namespace so the
reasoning engine can run code it did not write without risking the host
process. The sandbox:

* compiles code with ``RestrictedPython.compile_restricted`` (AST rewrite),
* exposes only a whitelist of safe builtins and importable modules,
* injects session state as ordinary variables,
* captures ``print`` output and the value of a designated result variable,
* always returns a structured :class:`SandboxResult` — sandboxed code can
  fail, but it can never crash or escape the caller.

This is `app/core/sandbox.py`. It is unrelated to the legacy
`app/blocks/sandbox.py` block.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector

ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "app.lib.pm_computations",
        "math",
        "statistics",
        "datetime",
        "json",
    }
)

BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
)

DEFAULT_RESULT_VAR = "result"


class SandboxError(Exception):
    """Raised for sandbox *setup* failures (bad config), never for errors
    inside sandboxed code — those are reported via :class:`SandboxResult`."""


class SandboxResult(BaseModel):
    """Structured outcome of one :func:`run_sandboxed` call."""

    success: bool
    stdout: str = ""
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    namespace: Dict[str, Any] = Field(default_factory=dict)
```

The implementation continues with `_make_guarded_import` (an `__import__`
replacement that raises `ImportError` for anything outside `ALLOWED_MODULES`
and forbids relative imports), `_build_builtins` (copies `safe_builtins`,
re-adds the `_EXTRA_SAFE_BUILTINS` set of pure aggregate/iteration helpers —
`sum`/`min`/`max`/`all`/`any`/`enumerate`/`map`/`filter`/`list`/`dict`/`set`/…
that `safe_builtins` conservatively omits — pops `BLOCKED_BUILTINS`, installs
the guarded importer), `_build_globals` (adds the RestrictedPython rewrite
hooks — `_print_`, `_getattr_`, `_getitem_`, `_getiter_`,
`_iter_unpack_sequence_`, `_unpack_sequence_`, `_write_` — then `update`s
with the injected `state`), and finally `run_sandboxed`:

```python
def run_sandboxed(
    code: str,
    state: Optional[Dict[str, Any]] = None,
    result_var: str = DEFAULT_RESULT_VAR,
) -> SandboxResult:
    """Execute ``code`` in the RestrictedPython jail and return the outcome.

    Never raises for errors *inside* the sandboxed code — compile errors,
    blocked imports, and runtime exceptions are all captured into the result.
    """
    # Deep-copy injected state so sandboxed mutations never leak to the caller.
    injected = copy.deepcopy(dict(state or {}))
    glb = _build_globals(injected)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")
    except SyntaxError as exc:
        return SandboxResult(
            success=False, error=f"Syntax error: {exc}", error_type="SyntaxError"
        )

    local_ns: Dict[str, Any] = {}
    try:
        exec(byte_code, glb, local_ns)
    except Exception as exc:
        return SandboxResult(
            success=False,
            stdout=_extract_stdout(local_ns),
            error=str(exc) or type(exc).__name__,
            error_type=type(exc).__name__,
            namespace=_safe_namespace(local_ns),
        )

    return SandboxResult(
        success=True,
        stdout=_extract_stdout(local_ns),
        result=local_ns.get(result_var),
        namespace=_safe_namespace(local_ns),
    )
```

`_extract_stdout` reads RestrictedPython's `_print` `PrintCollector` from the
local namespace (calling it returns the accumulated text); `_safe_namespace`
strips underscore-prefixed internals for clean inspection. See the committed
file for the full helper bodies.

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_sandbox.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/core/sandbox.py tests/test_sandbox.py
git commit -m "feat(sandbox): RestrictedPython jail + SandboxResult (plan 3)"
```

---

### Task 3: Import whitelist enforcement

**Files:**
- Test: `tests/test_sandbox.py` (whitelist logic was written in Task 2)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sandbox.py`:

```python
@pytest.mark.parametrize("module", ["os", "sys", "subprocess", "socket", "shutil"])
def test_dangerous_module_imports_are_blocked(module):
    r = run_sandboxed(f"import {module}")
    assert r.success is False
    assert r.error_type in ("ImportError", "SyntaxError")
    if r.error_type == "ImportError":
        assert module in r.error


def test_from_import_of_blocked_module_is_blocked():
    r = run_sandboxed("from os import getcwd\nresult = getcwd()")
    assert r.success is False


def test_whitelisted_modules_are_the_safe_set():
    assert "os" not in ALLOWED_MODULES and "sys" not in ALLOWED_MODULES
    assert {"math", "statistics", "datetime", "json"} <= ALLOWED_MODULES
    assert "app.lib.pm_computations" in ALLOWED_MODULES
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_sandbox.py -q`
Expected: PASS (9 passed) — the guarded importer from Task 2 already
enforces the whitelist. If a test fails, fix `_make_guarded_import` /
`ALLOWED_MODULES` before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sandbox.py
git commit -m "test(sandbox): import-whitelist enforcement coverage"
```

---

### Task 4: Blocked builtins

**Files:**
- Test: `tests/test_sandbox.py` (builtin stripping was written in Task 2)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sandbox.py`:

```python
def test_open_is_not_available():
    r = run_sandboxed("result = open('secret.txt')")
    assert r.success is False
    assert r.error_type in ("NameError", "SyntaxError")


def test_eval_is_blocked():
    # RestrictedPython rejects eval() at compile time.
    r = run_sandboxed("result = eval('1 + 1')")
    assert r.success is False
    assert r.error_type in ("SyntaxError", "NameError")


def test_exec_is_blocked():
    r = run_sandboxed("exec('x = 1')")
    assert r.success is False


def test_dunder_import_is_blocked():
    r = run_sandboxed("result = __import__('os')")
    assert r.success is False


def test_blocked_builtins_constant_lists_the_dangerous_names():
    assert {"open", "eval", "exec", "__import__"} <= BLOCKED_BUILTINS
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_sandbox.py -q`
Expected: PASS (14 passed) — `safe_builtins` already omits these and
RestrictedPython's AST rewrite rejects `eval`/`exec`/`__import__` at compile
time. If a test fails, fix `_build_builtins` / `BLOCKED_BUILTINS`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sandbox.py
git commit -m "test(sandbox): blocked-builtin coverage"
```

---

### Task 5: State injection + output capture

**Files:**
- Test: `tests/test_sandbox.py` (injection & capture were written in Task 2)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sandbox.py`:

```python
def test_state_is_injected_as_variables():
    r = run_sandboxed("result = activities + bonus", {"activities": 10, "bonus": 5})
    assert r.success is True and r.result == 15


def test_state_round_trip_through_namespace():
    state = {"crew": [4, 6, 2]}
    r = run_sandboxed("total = sum(crew)\nresult = total", state)
    assert r.success is True
    assert r.result == 12
    assert r.namespace["total"] == 12


def test_injected_state_is_copied_not_mutated_in_caller():
    state = {"items": [1, 2, 3]}
    run_sandboxed("items.append(99)", state)
    assert state == {"items": [1, 2, 3]}  # caller's dict untouched


def test_stdout_is_captured():
    r = run_sandboxed("print('hello'); print('world')")
    assert r.success is True
    assert "hello" in r.stdout and "world" in r.stdout


def test_result_variable_is_configurable():
    r = run_sandboxed("answer = 7 * 6", result_var="answer")
    assert r.success is True and r.result == 42


def test_missing_result_variable_yields_none():
    r = run_sandboxed("x = 1")
    assert r.success is True and r.result is None
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_sandbox.py -q`
Expected: PASS (20 passed) — `run_sandboxed` copies `state` into a fresh
dict, `update`s it into the exec globals, reads `_print` for stdout and
`result_var` for the result. If a test fails, fix `_build_globals`,
`_extract_stdout`, or the `state = dict(state or {})` copy.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sandbox.py
git commit -m "test(sandbox): state injection + output capture"
```

---

### Task 6: Error handling — syntax & runtime

**Files:**
- Test: `tests/test_sandbox.py` (error capture was written in Task 2)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sandbox.py`:

```python
def test_syntax_error_is_captured_not_raised():
    r = run_sandboxed("def broken(:\n    pass")
    assert r.success is False
    assert r.error_type == "SyntaxError"
    assert r.error and "yntax" in r.error


def test_runtime_error_is_captured():
    r = run_sandboxed("result = 1 / 0")
    assert r.success is False
    assert r.error_type == "ZeroDivisionError"


def test_name_error_is_captured():
    r = run_sandboxed("result = undefined_name + 1")
    assert r.success is False
    assert r.error_type == "NameError"


def test_stdout_before_a_crash_is_still_returned():
    r = run_sandboxed("print('made it')\nresult = 1 / 0")
    assert r.success is False
    assert "made it" in r.stdout


def test_sandbox_never_raises_for_bad_code():
    # Whatever the input, run_sandboxed returns a SandboxResult.
    for bad in ["@@@", "raise Exception('boom')", "1/0", "import os"]:
        assert isinstance(run_sandboxed(bad), SandboxResult)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_sandbox.py -q`
Expected: PASS (25 passed) — `run_sandboxed` wraps compilation in an
`except SyntaxError` and execution in an `except Exception`, so every
failure path returns a `SandboxResult`. If a test fails, fix the
`try`/`except` blocks in `run_sandboxed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sandbox.py
git commit -m "test(sandbox): syntax & runtime error capture"
```

---

### Task 7: Regression check

**Files:** none — verification only.

- [ ] **Step 1: Run the reasoning-engine suite**

Run:
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_sandbox.py tests/test_pm_computations.py tests/test_pm_extended.py tests/test_session_store.py -q
```
Expected: PASS — `test_sandbox.py` (25) + `test_pm_computations.py` +
`test_pm_extended.py` (11) + `test_session_store.py` (10 passed, 1 skipped),
no failures.

- [ ] **Step 2: Run the full suite**

Run: `& .venv\Scripts\python.exe -m pytest --ignore=tests/browser -q`
Expected: PASS — prior green total + 25 new sandbox tests, skip count
unchanged. Plan 3 adds no imports to existing modules, so a regression
here would be unexpected.

- [ ] **Step 3: Commit** — nothing to commit unless a regression was fixed
(Tasks 1–6 already committed their work).

---

## Self-Review

**Spec coverage** (Reasoning Engine §5.3 / §7 — sandboxed code execution):
- RestrictedPython exec of untrusted code → Task 2 ✅ (`compile_restricted` + `exec` with restricted globals)
- Import whitelist (allow `app.lib.pm_computations`, `math`, `statistics`, `datetime`, `json`; deny `os`/`sys`/`subprocess`/`socket`/…) → Task 3 ✅ (`_make_guarded_import` + `ALLOWED_MODULES`)
- Blocked builtins (`open`/`eval`/`exec`/`__import__`) → Task 4 ✅ (`safe_builtins` minus `BLOCKED_BUILTINS`; RestrictedPython's AST rewrite also rejects `eval`/`exec`/`__import__` at compile time)
- Session-state injection → Task 5 ✅ (`state` copied and `update`d into exec globals)
- Output capture — stdout + result variable → Task 5 ✅ (`PrintCollector`, configurable `result_var`)
- Structured result, never crashes the host → Tasks 2 & 6 ✅ (`SandboxResult`; every failure path is caught)

**Decisions & limitations (noted):**
- **No CPU/memory/wall-clock timeout.** RestrictedPython is a *static* jail — it cannot stop `while True:`. `app/blocks/sandbox.py` uses POSIX `resource.setrlimit`, which is unavailable on Windows (this repo's dev OS). A timeout belongs in **Plan 4's** caller, where the code runs under a worker thread/process that can be killed; the INDEX scopes Plan 3 to the RestrictedPython jail only. Documented as out of scope here.
- **`eval`/`exec`/`__import__`** surface as `SyntaxError` (compile-time AST rejection), not `NameError` — tests accept either so they stay robust across RestrictedPython releases.
- **`open`** surfaces as `NameError` — it is simply absent from `safe_builtins`; listing it in `BLOCKED_BUILTINS` is defensive documentation.
- **Submodule whitelist is exact-match.** `import app.lib.pm_computations` is allowed; `import app` or `import app.core` is not. The reasoner only ever needs the one library module.
- **`safe_builtins` is too narrow for PM code.** RestrictedPython's `safe_builtins` omits `sum`/`min`/`max`/`all`/`any`/`enumerate`/`map`/`filter`/`list`/`dict`/`set`/… — all pure, side-effect-free functions that formula code routinely needs. `_EXTRA_SAFE_BUILTINS` re-adds exactly those; nothing with I/O or process access is added. `BLOCKED_BUILTINS` is popped *after* this re-add so the strip is authoritative.
- **Injected state is deep-copied.** Sandboxed code may mutate injected lists/dicts; `copy.deepcopy` ensures those mutations never leak back into the caller's session state. Un-copyable values fall back to a shallow copy rather than failing the run.
- `RestrictedPython` is added to `requirements.txt` (Task 1) — unlike Plan 2's optional `redis`, this is a hard dependency of Plan 4.

**Placeholder scan:** none — complete code or an exact command in every step (the long helper bodies of `app/core/sandbox.py` are summarised in Task 2 Step 3 and shipped in full in the committed file).

**Type consistency:** `run_sandboxed(str, Optional[Dict[str, Any]], str) -> SandboxResult`. `SandboxResult` is a Pydantic v2 model (`success: bool`, `stdout: str`, `result: Any`, `error/error_type: Optional[str]`, `namespace: Dict[str, Any]`). `ALLOWED_MODULES` / `BLOCKED_BUILTINS` are `frozenset[str]`. Every task begins with a failing (or red-then-confirmed) test.

---

**Plan 3 complete.** Next: Plan 4 (`formula_executor_v2` — code generation), which depends on this sandbox.
