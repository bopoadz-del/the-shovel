# Reasoning Engine — Plan 4: formula_executor v2 (LLM code generation)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.
> Part of the Reasoning Engine — see `2026-05-20-reasoning-engine-INDEX.md`. **Depends on Plan 3 (Sandbox).**

**Goal:** Replace the pattern-matching `_generate_formula` of the legacy `FormulaExecutorBlock` with a real LLM code-generation loop: an LLM writes Python for a described task, the code runs inside the Plan-3 sandbox (`app/core/sandbox.py`), a successful result is cached on the session, and a runtime failure feeds the traceback back to the LLM for a bounded retry.

**Architecture:**
- `app/prompts/codegen_system.py` — a pure function `build_codegen_prompt(...)` that assembles the system prompt for the code-generation LLM call. No I/O, no AI — just string building, so it is unit-testable without a key.
- `app/blocks/formula_executor_v2.py` — `FormulaExecutorV2Block`, a `UniversalBlock`. It owns the generate → run → cache → retry loop. The LLM call is isolated behind a single overridable method `_call_llm(...)` so mock-LLM tests can subclass it and live tests can hit the configured LLM.
- `app/blocks/formula_executor.py` — MODIFIED into a thin deprecation wrapper: `FormulaExecutorBlock.process` delegates to a `FormulaExecutorV2Block` instance, keeping the old `name = "formula_executor"` registration working unchanged.
- `app/blocks/__init__.py` — MODIFIED to import and register `FormulaExecutorV2Block` under the key `formula_executor_v2`.

**LLM provider:** the configured LLM, same call shape as `app/blocks/chat.py` (`ChatBlock._call_cloud`) — `POST https://api.moonshot.ai/v1/chat/completions`, `Authorization: Bearer <KIMI_API_KEY>`, model `kimi-k2.6`. The key (`KIMI_API_KEY`) is **pending refill** — this plan ships with **mock-LLM tests only**; the live end-to-end test is written but `@pytest.mark.skipif` on the key (Task 6) and runs once the key is funded.

**Sandbox contract (from Plan 3 — assumed API):** `app/core/sandbox.py` exposes `run_sandboxed(code: str, variables: dict, *, allowed_imports: list[str] | None = None, timeout_seconds: int = 10) -> SandboxResult`, where `SandboxResult` has fields `ok: bool`, `result: Any` (the value bound to `result` in the snippet), `stdout: str`, `error: str | None`, `traceback: str | None`. If Plan 3's final API differs, adjust the calls in Tasks 2–4 to match — the loop logic is unchanged.

**Tech Stack:** Python 3.11, Pydantic v2, `httpx` (already used by `ChatBlock`). No new dependencies.

**Run tests:** `& .venv\Scripts\python.exe -m pytest <path> -q` from `C:\Users\example\The_Fork`. **Plan 3 must be complete first.**

---

### Task 1: Code-generation system prompt builder

**Files:**
- Create: `app/prompts/__init__.py`
- Create: `app/prompts/codegen_system.py`
- Test: `tests/test_formula_executor_v2.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_formula_executor_v2.py`:

```python
"""Tests for formula_executor v2 — Reasoning Engine Plan 4."""

import pytest

from app.prompts.codegen_system import build_codegen_prompt


def test_prompt_states_the_result_contract():
    p = build_codegen_prompt("compute slab volume", {"length_m": 10})
    # The generated code must assign to a variable called `result`.
    assert "result" in p
    assert "```python" in p or "code block" in p.lower()


def test_prompt_lists_available_variables():
    p = build_codegen_prompt("area", {"length_m": 10, "width_m": 8})
    assert "length_m" in p and "width_m" in p


def test_prompt_advertises_the_pm_library():
    p = build_codegen_prompt("critical path", {})
    # Generated code may import the tested CPM library instead of re-deriving it.
    assert "app.lib.pm_computations" in p
    assert "compute_cpm" in p


def test_prompt_includes_retry_context_when_given():
    p = build_codegen_prompt(
        "area", {"length_m": 10},
        prior_code="result = length_m *",
        prior_error="SyntaxError: invalid syntax",
    )
    assert "SyntaxError" in p
    assert "result = length_m *" in p


def test_prompt_omits_retry_section_on_first_attempt():
    p = build_codegen_prompt("area", {"length_m": 10})
    assert "previous attempt" not in p.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.prompts.codegen_system'`

- [ ] **Step 3: Write the prompt builder**

Create `app/prompts/__init__.py` (empty).

Create `app/prompts/codegen_system.py`:

```python
"""Code-generation system prompt — Reasoning Engine Plan 4.

Pure string building. No AI, no I/O — unit-testable without an API key.
"""

from typing import Any, Dict, Optional

_CONTRACT = """\
You are a Python code generator for a construction-project intelligence \
platform. Write a SHORT Python snippet that solves the described task.

HARD RULES:
- Assign the final answer to a variable named `result`.
- Use ONLY the input variables listed below — they are already in scope.
- You MAY import from the standard library `math` module and from \
`app.lib.pm_computations` (tested project-management functions, e.g. \
`compute_cpm`, `resource_histogram`, `gantt_data`, `compress_schedule`).
- Do NOT read files, make network calls, or import anything else.
- Return ONLY a single fenced ```python code block — no prose.
"""


def build_codegen_prompt(
    task: str,
    variables: Dict[str, Any],
    *,
    prior_code: Optional[str] = None,
    prior_error: Optional[str] = None,
) -> str:
    """Assemble the system prompt for one code-generation attempt.

    `prior_code` / `prior_error` are supplied on a retry so the LLM can see
    what failed and why; they are omitted on the first attempt.
    """
    if variables:
        var_lines = "\n".join(
            f"  - {k} = {v!r}" for k, v in variables.items()
        )
        var_block = f"INPUT VARIABLES (already in scope):\n{var_lines}"
    else:
        var_block = "INPUT VARIABLES: none."

    parts = [_CONTRACT, f"TASK:\n{task}", var_block]

    if prior_code is not None and prior_error is not None:
        parts.append(
            "YOUR PREVIOUS ATTEMPT FAILED. Fix it.\n"
            f"Previous code:\n```python\n{prior_code}\n```\n"
            f"Error:\n{prior_error}"
        )

    return "\n\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/prompts/__init__.py app/prompts/codegen_system.py tests/test_formula_executor_v2.py
git commit -m "feat(codegen): code-generation system prompt builder (reasoning engine plan 4)"
```

---

### Task 2: FormulaExecutorV2Block — generate + run (happy path)

**Files:**
- Create: `app/blocks/formula_executor_v2.py`
- Test: `tests/test_formula_executor_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formula_executor_v2.py`:

```python
from app.blocks.formula_executor_v2 import FormulaExecutorV2Block


class _MockLLMBlock(FormulaExecutorV2Block):
    """Test double — returns canned code instead of calling the configured LLM.

    `scripted` is a list of code strings yielded one per LLM call, so a test
    can script a first failing attempt followed by a passing retry.
    """

    def __init__(self, scripted, **kw):
        super().__init__(**kw)
        self._scripted = list(scripted)
        self.llm_calls = 0

    async def _call_llm(self, prompt: str) -> str:
        self.llm_calls += 1
        return self._scripted.pop(0)


@pytest.mark.asyncio
async def test_v2_generates_and_runs_code():
    block = _MockLLMBlock(["result = length_m * width_m"])
    out = await block.process({
        "task": "rectangle area",
        "variables": {"length_m": 10, "width_m": 8},
    })
    assert out["status"] == "success"
    assert out["result"] == 80
    assert out["generated_code"] == "result = length_m * width_m"
    assert block.llm_calls == 1


@pytest.mark.asyncio
async def test_v2_strips_markdown_fences_from_llm_output():
    # LLMs wrap code in ```python fences — the block must unwrap them.
    block = _MockLLMBlock(["```python\nresult = length_m * 2\n```"])
    out = await block.process({"task": "double", "variables": {"length_m": 5}})
    assert out["status"] == "success"
    assert out["result"] == 10
    assert "```" not in out["generated_code"]
```

> The repo already runs async tests — `pytest-asyncio` is configured (see
> `tests/` for existing `@pytest.mark.asyncio` blocks).

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.blocks.formula_executor_v2'`

- [ ] **Step 3: Write the block (generate + run, no retry yet)**

Create `app/blocks/formula_executor_v2.py`:

```python
"""Formula Executor v2 — LLM code generation + sandboxed execution.

Reasoning Engine Plan 4. Supersedes the pattern-matching FormulaExecutorBlock.

Flow: build prompt -> LLM writes Python -> run in the Plan-3 sandbox ->
cache a success on the session -> retry with the traceback on a failure.
"""

import os
import re
from typing import Any, Dict, Optional

import httpx

from app.core.universal_base import UniversalBlock
from app.core.sandbox import run_sandboxed
from app.prompts.codegen_system import build_codegen_prompt

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
_ALLOWED_IMPORTS = ["math", "app.lib.pm_computations"]


def _strip_fences(text: str) -> str:
    """Pull the code out of a ```python ...``` block; return text as-is if
    the LLM did not fence it."""
    m = _FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


class FormulaExecutorV2Block(UniversalBlock):
    name = "formula_executor_v2"
    version = "2.0.0"
    description = (
        "LLM code-generation: describe a task, an LLM writes Python, it runs "
        "sandboxed, results are cached, failures retried."
    )
    layer = 3
    tags = ["domain", "construction", "codegen", "llm", "sandbox", "reasoning"]
    requires = []

    default_config = {
        "max_retries": 2,        # extra attempts after the first
        "timeout_seconds": 10,
        "model": "kimi-k2.6",
    }

    ui_schema = {
        "input": {
            "type": "json",
            "placeholder": '{"task": "concrete volume for a 10x8m slab 0.2m thick", "variables": {"length_m": 10, "width_m": 8, "thickness_m": 0.2}}',
            "multiline": True,
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "generated_code", "type": "code", "label": "Generated Code"},
                {"name": "result", "type": "text", "label": "Result"},
                {"name": "attempts", "type": "number", "label": "Attempts"},
            ],
        },
        "quick_actions": [
            {"icon": "🧮", "label": "Calculate", "prompt": "Calculate concrete volume for a 10x8m slab, 200mm thick"},
        ],
    }

    async def _call_llm(self, prompt: str) -> str:
        """Send the code-gen prompt to the configured LLM and return the raw reply.

        Overridden by test doubles. Raises RuntimeError when no key is set so
        callers get a clear, non-secret error.
        """
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            raise RuntimeError("KIMI_API_KEY not configured")
        model = self.config.get("model", "kimi-k2.6")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.moonshot.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,   # deterministic code generation
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"the configured LLM API error (HTTP {resp.status_code})"
                )
            return resp.json()["choices"][0]["message"]["content"]

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}
        task = data.get("task") or data.get("formula_description") \
            or params.get("task") or (str(input_data) if not isinstance(input_data, dict) else "")
        variables = dict(data.get("variables") or data.get("input_values") or {})

        if not task.strip():
            return {"status": "error", "error": "No task description provided"}

        prompt = build_codegen_prompt(task, variables)
        try:
            raw = await self._call_llm(prompt)
        except Exception as e:
            return {"status": "error", "error": f"Code generation failed: {e}"}

        code = _strip_fences(raw)
        sandbox = run_sandboxed(
            code, variables,
            allowed_imports=_ALLOWED_IMPORTS,
            timeout_seconds=int(self.config.get("timeout_seconds", 10)),
        )
        if sandbox.ok:
            return {
                "status": "success",
                "generated_code": code,
                "result": sandbox.result,
                "stdout": sandbox.stdout,
                "attempts": 1,
                "task": task,
            }
        return {
            "status": "error",
            "error": sandbox.error or "Sandbox execution failed",
            "generated_code": code,
            "traceback": sandbox.traceback,
            "attempts": 1,
            "task": task,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app/blocks/formula_executor_v2.py tests/test_formula_executor_v2.py
git commit -m "feat(codegen): FormulaExecutorV2Block — generate + sandboxed run"
```

---

### Task 3: Retry loop on sandbox failure

**Files:**
- Modify: `app/blocks/formula_executor_v2.py` (extract a retry loop in `process`)
- Test: `tests/test_formula_executor_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formula_executor_v2.py`:

```python
@pytest.mark.asyncio
async def test_v2_retries_after_a_runtime_failure():
    # First attempt references an undefined name; retry fixes it.
    block = _MockLLMBlock([
        "result = lenght_m * width_m",       # typo -> NameError
        "result = length_m * width_m",       # corrected
    ])
    out = await block.process({
        "task": "rectangle area",
        "variables": {"length_m": 10, "width_m": 8},
    })
    assert out["status"] == "success"
    assert out["result"] == 80
    assert out["attempts"] == 2
    assert block.llm_calls == 2


@pytest.mark.asyncio
async def test_v2_gives_up_after_max_retries():
    # Every attempt fails -> 1 initial + 2 retries = 3 LLM calls, then error.
    block = _MockLLMBlock([
        "result = undefined_a",
        "result = undefined_b",
        "result = undefined_c",
    ])
    out = await block.process({"task": "x", "variables": {}})
    assert out["status"] == "error"
    assert out["attempts"] == 3
    assert block.llm_calls == 3
    assert out["traceback"] is not None


@pytest.mark.asyncio
async def test_v2_passes_traceback_into_the_retry_prompt():
    captured = []

    class _Spy(_MockLLMBlock):
        async def _call_llm(self, prompt):
            captured.append(prompt)
            return await super()._call_llm(prompt)

    block = _Spy(["result = bad_name", "result = 1"])
    await block.process({"task": "x", "variables": {}})
    # Second prompt must carry the prior code + error for self-correction.
    assert "bad_name" in captured[1]
    assert "previous attempt" in captured[1].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: FAIL — `test_v2_retries_after_a_runtime_failure` fails: `attempts == 1` and `llm_calls == 1` (no retry loop yet).

- [ ] **Step 3: Add the retry loop**

In `app/blocks/formula_executor_v2.py`, replace the body of `process` from the
`prompt = build_codegen_prompt(...)` line onward with this loop:

```python
        max_attempts = 1 + int(self.config.get("max_retries", 2))
        prior_code: Optional[str] = None
        prior_error: Optional[str] = None
        last_code = ""
        last_traceback: Optional[str] = None
        last_error = "Code generation failed"

        for attempt in range(1, max_attempts + 1):
            prompt = build_codegen_prompt(
                task, variables,
                prior_code=prior_code, prior_error=prior_error,
            )
            try:
                raw = await self._call_llm(prompt)
            except Exception as e:
                return {"status": "error",
                        "error": f"Code generation failed: {e}",
                        "attempts": attempt, "task": task}

            code = _strip_fences(raw)
            last_code = code
            sandbox = run_sandboxed(
                code, variables,
                allowed_imports=_ALLOWED_IMPORTS,
                timeout_seconds=int(self.config.get("timeout_seconds", 10)),
            )
            if sandbox.ok:
                return {
                    "status": "success",
                    "generated_code": code,
                    "result": sandbox.result,
                    "stdout": sandbox.stdout,
                    "attempts": attempt,
                    "task": task,
                }
            # failed — carry context into the next attempt
            last_error = sandbox.error or "Sandbox execution failed"
            last_traceback = sandbox.traceback
            prior_code = code
            prior_error = sandbox.traceback or last_error

        return {
            "status": "error",
            "error": last_error,
            "generated_code": last_code,
            "traceback": last_traceback,
            "attempts": max_attempts,
            "task": task,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add app/blocks/formula_executor_v2.py tests/test_formula_executor_v2.py
git commit -m "feat(codegen): bounded retry loop feeding tracebacks back to the LLM"
```

---

### Task 4: Result caching on the session

**Files:**
- Modify: `app/blocks/formula_executor_v2.py` (cache + cache-hit short-circuit)
- Test: `tests/test_formula_executor_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formula_executor_v2.py`:

```python
from app.core.session_store import InMemorySessionStore


@pytest.mark.asyncio
async def test_v2_caches_successful_code_on_the_session():
    store = InMemorySessionStore()
    session = store.get_or_create("s1")
    block = _MockLLMBlock(["result = length_m * 2"])
    await block.process({
        "task": "double the length", "variables": {"length_m": 5},
        "session": session,
    })
    # the generated code is cached under the session's code_cache
    assert any("result = length_m * 2" in v
               for v in session.code_cache.values())


@pytest.mark.asyncio
async def test_v2_reuses_cached_code_without_calling_the_llm():
    store = InMemorySessionStore()
    session = store.get_or_create("s1")
    block = _MockLLMBlock(["result = length_m * 2"])
    first = await block.process({
        "task": "double the length", "variables": {"length_m": 5},
        "session": session,
    })
    assert first["result"] == 10 and block.llm_calls == 1

    # Same task + same variable KEYS -> cache hit, no second LLM call.
    second = await block.process({
        "task": "double the length", "variables": {"length_m": 9},
        "session": session,
    })
    assert second["status"] == "success"
    assert second["result"] == 18           # re-runs cached code with new value
    assert second.get("cache_hit") is True
    assert block.llm_calls == 1             # LLM NOT called again
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: FAIL — `test_v2_caches_successful_code_on_the_session` fails: `code_cache` is empty.

- [ ] **Step 3: Add caching**

In `app/blocks/formula_executor_v2.py`, add a cache-key helper near the top
(after `_strip_fences`):

```python
def _cache_key(task: str, variables: Dict[str, Any]) -> str:
    """Stable key for a code-gen request: the task plus the sorted variable
    NAMES (not values — cached code is re-run with fresh values)."""
    names = ",".join(sorted(variables))
    return f"{task.strip().lower()}|{names}"
```

In `process`, after `variables` is built and before the retry loop, pull the
session (passed in `data["session"]`) and check the cache:

```python
        session = data.get("session") or params.get("session")
        key = _cache_key(task, variables)

        if session is not None and key in session.code_cache:
            cached_code = session.code_cache[key]
            sandbox = run_sandboxed(
                cached_code, variables,
                allowed_imports=_ALLOWED_IMPORTS,
                timeout_seconds=int(self.config.get("timeout_seconds", 10)),
            )
            if sandbox.ok:
                return {
                    "status": "success",
                    "generated_code": cached_code,
                    "result": sandbox.result,
                    "stdout": sandbox.stdout,
                    "attempts": 0,
                    "cache_hit": True,
                    "task": task,
                }
            # stale cache (e.g. variable set changed) — fall through to regen
```

Then, in the retry loop's success branch, write the cache before returning:

```python
            if sandbox.ok:
                if session is not None:
                    session.code_cache[key] = code
                return {
                    "status": "success",
                    "generated_code": code,
                    "result": sandbox.result,
                    "stdout": sandbox.stdout,
                    "attempts": attempt,
                    "cache_hit": False,
                    "task": task,
                }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add app/blocks/formula_executor_v2.py tests/test_formula_executor_v2.py
git commit -m "feat(codegen): cache generated code on the session, reuse on hit"
```

---

### Task 5: Deprecation wrapper + block registration

**Files:**
- Modify: `app/blocks/formula_executor.py` (rewrite as a v2 delegating wrapper)
- Modify: `app/blocks/__init__.py` (register `formula_executor_v2`)
- Test: `tests/test_formula_executor_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formula_executor_v2.py`:

```python
from app.blocks import BLOCK_REGISTRY, get_block
from app.blocks.formula_executor import FormulaExecutorBlock


def test_v2_is_registered():
    assert "formula_executor_v2" in BLOCK_REGISTRY
    assert get_block("formula_executor_v2") is FormulaExecutorV2Block


def test_legacy_block_still_registered_under_old_key():
    # The old key keeps working — chains referencing it must not break.
    assert "formula_executor" in BLOCK_REGISTRY


@pytest.mark.asyncio
async def test_legacy_block_delegates_to_v2():
    # The legacy block is now a wrapper; it must hold a v2 delegate.
    legacy = FormulaExecutorBlock()
    assert isinstance(legacy._v2, FormulaExecutorV2Block)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: FAIL — `test_v2_is_registered` fails: `"formula_executor_v2" not in BLOCK_REGISTRY`.

- [ ] **Step 3: Rewrite the legacy block + register v2**

Replace the **entire** contents of `app/blocks/formula_executor.py` with a
deprecation wrapper. The old block kept a pattern-matching `_generate_formula`
and a hardcoded `_FORMULA_LIBRARY`; both are dropped — v2's LLM generation
covers them. The wrapper preserves the `name = "formula_executor"` registration
and `process` signature so existing chains keep working.

```python
"""Formula Executor Block — DEPRECATED.

Reasoning Engine Plan 4: this block is superseded by FormulaExecutorV2Block
(`app/blocks/formula_executor_v2.py`), which uses real LLM code generation
instead of pattern matching. This wrapper stays registered under the original
`formula_executor` key so existing chains do not break; it forwards every call
to a v2 instance. New code should target `formula_executor_v2` directly.
"""

import warnings
from typing import Any, Dict

from app.core.universal_base import UniversalBlock
from app.blocks.formula_executor_v2 import FormulaExecutorV2Block


class FormulaExecutorBlock(UniversalBlock):
    name = "formula_executor"
    version = "2.0.0"          # bumped — now backed by v2
    description = "DEPRECATED — delegates to formula_executor_v2 (LLM code-gen)."
    layer = 3
    tags = ["domain", "construction", "formula", "deprecated"]
    requires = []

    default_config = FormulaExecutorV2Block.default_config
    ui_schema = FormulaExecutorV2Block.ui_schema

    def __init__(self, hal_block=None, config: Dict = None):
        super().__init__(hal_block=hal_block, config=config)
        self._v2 = FormulaExecutorV2Block(hal_block=hal_block, config=config)

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        warnings.warn(
            "FormulaExecutorBlock is deprecated; use formula_executor_v2.",
            DeprecationWarning, stacklevel=2,
        )
        return await self._v2.process(input_data, params)
```

In `app/blocks/__init__.py`, add the import alongside the other Construction
Intelligence imports (after the `from .formula_executor import FormulaExecutorBlock`
line):

```python
from .formula_executor_v2 import FormulaExecutorV2Block
```

And add the registry entry inside `BLOCK_REGISTRY`, directly after the
`"formula_executor": FormulaExecutorBlock,` line:

```python
    "formula_executor_v2":  FormulaExecutorV2Block,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2.py -q`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add app/blocks/formula_executor.py app/blocks/formula_executor_v2.py app/blocks/__init__.py tests/test_formula_executor_v2.py
git commit -m "feat(codegen): deprecate FormulaExecutorBlock, register v2 block"
```

---

### Task 6: Live end-to-end test (skipped until the key is funded)

**Files:**
- Test: `tests/test_formula_executor_v2_live.py`

This is the real the configured LLM round-trip. It is **skipped** while `KIMI_API_KEY`
is unset (pending refill); once the key is funded, it runs unchanged and is the
acceptance check for genuine LLM code generation.

- [ ] **Step 1: Write the live test**

Create `tests/test_formula_executor_v2_live.py`:

```python
"""LIVE the configured LLM end-to-end test — Reasoning Engine Plan 4.

Skipped until KIMI_API_KEY is funded. This is the acceptance check for
real LLM code generation. The mock-LLM coverage is in
tests/test_formula_executor_v2.py and runs always.
"""

import os

import pytest

from app.blocks.formula_executor_v2 import FormulaExecutorV2Block

pytestmark = pytest.mark.skipif(
    not os.getenv("KIMI_API_KEY"),
    reason="KIMI_API_KEY not configured — pending refill",
)


@pytest.mark.asyncio
async def test_live_codegen_simple_arithmetic():
    block = FormulaExecutorV2Block()
    out = await block.process({
        "task": "concrete volume of a slab: length x width x thickness",
        "variables": {"length_m": 10, "width_m": 8, "thickness_m": 0.2},
    })
    assert out["status"] == "success"
    assert abs(out["result"] - 16.0) < 1e-6


@pytest.mark.asyncio
async def test_live_codegen_uses_pm_library():
    block = FormulaExecutorV2Block()
    out = await block.process({
        "task": (
            "Given activities A(3 days) -> B(5 days) -> C(2 days) in series, "
            "use app.lib.pm_computations.compute_cpm to get the project "
            "duration. Set result to the integer duration."
        ),
        "variables": {},
    })
    assert out["status"] == "success"
    assert out["result"] == 10
```

- [ ] **Step 2: Run the test**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_formula_executor_v2_live.py -q`
Expected (key unset): `2 skipped`. Expected (key funded): `2 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_formula_executor_v2_live.py
git commit -m "test(codegen): live the configured LLM e2e test (skipped until key funded)"
```

---

### Task: Regression check

**Files:** none — verification only.

- [ ] **Step 1: Run the full suite**

Run: `& .venv\Scripts\python.exe -m pytest --ignore=tests/browser -q`
Expected: PASS — all prior tests still pass, plus 15 new from this plan and 2
skipped from the live test. If any chain test that referenced
`formula_executor` fails, the wrapper's `process` signature drifted — re-check
Task 5.

- [ ] **Step 2: Commit** — nothing to commit unless a regression was fixed.

---

## Self-Review

**Spec coverage** (Reasoning Engine §7.1 — code generated on the fly; spec §5.3 — sandboxed formula executor):
- LLM code generation with a real system prompt → Tasks 1 & 2 ✅
- Sandboxed execution via the Plan-3 `run_sandboxed` jail → Task 2 ✅
- Retry on failure with the traceback fed back to the LLM → Task 3 ✅
- Successful code cached on the session, reused on a hit → Task 4 ✅
- Legacy `FormulaExecutorBlock` → deprecation wrapper delegating to v2 → Task 5 ✅
- New block registered in `BLOCK_REGISTRY` → Task 5 ✅

**LLM-key blocker handling:** every loop test uses `_MockLLMBlock`, which
overrides the single `_call_llm` seam — no key needed. The genuine round-trip
lives in `tests/test_formula_executor_v2_live.py`, `skipif` on the key. Slotting
in the live test once the key is funded needs **no code change** — just run it.

**Out of scope (noted):** `_FORMULA_LIBRARY` and `_match_library` from the old
block are intentionally dropped — v2's LLM covers ad-hoc formulas, and the
tested algorithms (CPM etc.) now live in `app/lib/pm_computations.py` per spec
§7. Code-length / line-count limits from the old block are dropped here; the
sandbox (Plan 3) owns resource limits.

**Assumption flagged:** the `run_sandboxed` signature and `SandboxResult` shape
are taken from this plan's header — Plan 3's final API must be confirmed before
Task 2; if it differs, only the `run_sandboxed(...)` call sites change.

**Placeholder scan:** none — every step has complete code or an exact command.

**Type consistency:** `build_codegen_prompt(str, Dict[str, Any], *, prior_code, prior_error) -> str`. `_strip_fences(str) -> str`. `_cache_key(str, Dict) -> str`. `FormulaExecutorV2Block.process(Any, Dict) -> Dict`; `_call_llm(str) -> str` is the only AI seam. The legacy `FormulaExecutorBlock.process` keeps its `(Any, Dict) -> Dict` signature. Every task has a failing-test step before its implementation.

---

**Plan 4 complete.** Next: Plan 5 (Project Reasoner).
