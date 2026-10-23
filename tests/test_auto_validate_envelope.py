"""`_auto_validate` attaches the number-checking verdict. Nothing tested it.

Audit bar 3, 2026-08-12, agents subsystem. Replacing the whole body with
`return` left all 150 agent tests green, and no test file in the repository so
much as mentions the function.

That matters more here than the line count suggests. This is the middleware
that runs the validation pipeline over every numeric a tool returns and
attaches the verdict the LLM is then instructed to obey -- the note it writes
literally says "refuse to report any number whose check shows overall=fail".
If it silently stopped attaching verdicts, every number would read as
unchecked-but-fine, and on a construction platform the numbers are the
product.

The tests assert the four distinct outcomes it is required to distinguish,
because collapsing any two of them is the failure that hides a broken check:

    not applicable   -> no `validation` key at all
    nothing to check -> validation.skipped
    checked, ok      -> validation.overall == "pass"
    checked, bad     -> validation.overall == "fail" + which stage rejected it

A stub that returns early looks identical to "not applicable", which is why
the skipped/absent distinction is asserted rather than assumed.
"""
from __future__ import annotations

import pytest

from app.agents import runtime


# The three shapes `_collect_numerics` actually recognises. It has no generic
# "value" handler -- {"value": 42, "unit": "m3"} is NOT collected -- so tests
# must use a real shape or they assert against a branch that never runs.
NUMERIC_RESULT = {"status": "success", "total_cost": 42000.0, "currency": "USD"}
FAILING_NUMERIC_RESULT = {"status": "success", "total_cost": -5.0, "currency": "USD"}


class _StubValidationBlock:
    """Stands in for validation_pipeline. Returns UniversalBlock's envelope
    shape: {block, status, result: {...}} with the verdict under `result`."""

    def __init__(self, overall="pass", first_failure=None):
        self.overall = overall
        self.first_failure = first_failure
        self.calls = []

    async def execute(self, numeric, params):
        self.calls.append(numeric)
        return {"block": "validation_pipeline", "status": "success",
                "result": {"overall": self.overall, "first_failure": self.first_failure}}


@pytest.fixture
def wired(monkeypatch):
    """Wire a stub validation_pipeline into the registry + instance cache."""
    from app import dependencies

    def _wire(block):
        monkeypatch.setitem(runtime.BLOCK_REGISTRY, "validation_pipeline", type(block))
        monkeypatch.setitem(dependencies.block_instances, "validation_pipeline", block)
        return block

    return _wire


# A real block whose class does not set `auto_validate = False`. It must be a
# BLOCK_REGISTRY entry: `_block_should_auto_validate` returns False for any
# name it cannot find, so a synthetic tool name here would make every test
# below pass vacuously by taking the opt-out branch.
AUTO_VALIDATING_BLOCK = "file_hasher"


def _envelope(result, name=AUTO_VALIDATING_BLOCK):
    return {"name": name, "result": result}


@pytest.mark.asyncio
async def test_a_block_that_opts_out_gets_no_validation_key(monkeypatch):
    """Absent must stay distinguishable from skipped.

    `_block_should_auto_validate` returns False for synthetic tools and for
    blocks declaring `auto_validate = False`. Those envelopes must come back
    untouched -- if they gained a `validation: {...}` the reader could not tell
    an opted-out tool from a checked one.
    """
    env2 = _envelope(dict(NUMERIC_RESULT), name="remember_fact")
    await runtime._auto_validate(env2)
    assert "validation" not in env2, env2


@pytest.mark.asyncio
async def test_a_failed_tool_result_is_not_validated():
    """There is nothing to check in a result that already errored, and
    attaching a passing verdict to one would be actively misleading."""
    env = _envelope({"status": "error", "error": "no file"})
    await runtime._auto_validate(env)
    assert "validation" not in env, env


@pytest.mark.asyncio
async def test_no_numerics_is_reported_as_skipped_not_as_a_pass(wired):
    """The distinction this test exists for.

    A result with nothing numeric in it has not been validated. Saying
    `overall: pass` would claim the numbers were checked when there were none.
    """
    wired(_StubValidationBlock())
    env = _envelope({"status": "success", "summary": "no figures here"})
    await runtime._auto_validate(env)
    assert env["validation"] == {"skipped": "no numeric value found"}, env["validation"]


@pytest.mark.asyncio
async def test_numerics_are_checked_and_the_verdict_is_attached(wired):
    block = wired(_StubValidationBlock(overall="pass"))
    env = _envelope(dict(NUMERIC_RESULT))
    await runtime._auto_validate(env)

    assert block.calls, "the validation pipeline was never called"
    validation = env["validation"]
    assert validation["overall"] == "pass", validation
    assert validation["checks"], "a verdict with no per-numeric checks proves nothing"
    assert "refuse to report" in validation["note"]


@pytest.mark.asyncio
async def test_a_failing_check_surfaces_which_stage_rejected_it(wired):
    """`overall: fail` without `first_failure` is unactionable.

    The LLM is told to explain which stage rejected the number; it can only do
    that if the stage name is carried up here.
    """
    wired(_StubValidationBlock(overall="fail", first_failure="range_check"))
    env = _envelope(dict(FAILING_NUMERIC_RESULT))
    await runtime._auto_validate(env)

    assert env["validation"]["overall"] == "fail", env["validation"]
    assert env["validation"]["first_failure"] == "range_check", env["validation"]


@pytest.mark.asyncio
async def test_numerics_are_found_inside_the_wrapped_inner_result(wired):
    """UniversalBlock double-wraps: {status, result: {status, <numerics>}}.

    Without the unwrap the middleware sees no numerics on every block that
    goes through UniversalBlock.execute and skips them all -- which would look
    exactly like "this tool returns nothing numeric" rather than a bug.
    """
    block = wired(_StubValidationBlock())
    env = _envelope({"status": "success", "result": dict(NUMERIC_RESULT)})
    await runtime._auto_validate(env)

    assert env["validation"].get("skipped") is None, (
        f"the inner result's numerics were not found: {env['validation']}"
    )
    assert block.calls, "validation pipeline never called for a wrapped result"


@pytest.mark.asyncio
async def test_an_unregistered_pipeline_is_reported_not_silently_passed(monkeypatch):
    """No validator must never read as a clean bill of health."""
    monkeypatch.delitem(runtime.BLOCK_REGISTRY, "validation_pipeline", raising=False)
    env = _envelope(dict(NUMERIC_RESULT))
    await runtime._auto_validate(env)
    assert env["validation"] == {"skipped": "validation_pipeline not registered"}, env["validation"]
