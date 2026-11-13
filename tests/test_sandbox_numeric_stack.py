"""S-M1: the numeric stack (numpy, scipy, sympy) inside the formula sandbox.

Every test builds its own inputs and checks the sandbox's answer against a
closed form or a property computed here from those same inputs. Each
mechanism runs on two inputs, so a constant answer cannot pass.
"""

import math

import pytest

from app.blocks.formula_executor_v2 import FormulaExecutorV2Block
from app.core.sandbox import ALLOWED_MODULES, ALLOWED_SUBPACKAGES, run_sandboxed
from app.prompts.codegen_system import build_codegen_prompt

# --------------------------------------------------------------------------
# Root finding: pump duty point (pump curve meets system curve)
# --------------------------------------------------------------------------

_DUTY_POINT = """\
from scipy.optimize import brentq
import math
def pump_head(q):
    return shutoff_head_m - pump_coeff * q * q
def system_head(q):
    return static_head_m + friction_coeff * q * q
q_max = math.sqrt(shutoff_head_m / pump_coeff)
q = brentq(lambda q: pump_head(q) - system_head(q), 0.0, q_max)
result = {"flow": q, "head": system_head(q)}
"""


@pytest.mark.parametrize(
    "shutoff_head_m, pump_coeff, static_head_m, friction_coeff",
    [(40.0, 0.010, 10.0, 0.002), (65.0, 0.004, 22.0, 0.0015)],
)
def test_root_finding_duty_point(shutoff_head_m, pump_coeff, static_head_m, friction_coeff):
    out = run_sandboxed(
        _DUTY_POINT,
        state={
            "shutoff_head_m": shutoff_head_m,
            "pump_coeff": pump_coeff,
            "static_head_m": static_head_m,
            "friction_coeff": friction_coeff,
        },
    )
    assert out.success, out.error
    # H0 - a Q^2 = Hs + k Q^2  =>  Q = sqrt((H0 - Hs) / (a + k))
    q_exact = math.sqrt((shutoff_head_m - static_head_m) / (pump_coeff + friction_coeff))
    assert out.result["flow"] == pytest.approx(q_exact, rel=1e-9)
    assert out.result["head"] == pytest.approx(
        static_head_m + friction_coeff * q_exact ** 2, rel=1e-9
    )


# --------------------------------------------------------------------------
# Interpolation: S-curve of cumulative progress
# --------------------------------------------------------------------------

_S_CURVE = """\
import numpy as np
from scipy.interpolate import PchipInterpolator
curve = PchipInterpolator(weeks, cumulative_pct)
grid = np.linspace(weeks[0], weeks[-1], 41)
result = {
    "at_query": float(curve(query_week)),
    "at_knots": [float(v) for v in curve(weeks)],
    "grid": [float(v) for v in curve(grid)],
}
"""


@pytest.mark.parametrize(
    "weeks, query",
    [([0.0, 4.0, 8.0, 12.0, 16.0], 10.0), ([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0], 7.5)],
)
def test_s_curve_interpolation_reproduces_linear_progress_exactly(weeks, query):
    # Steady progress: cumulative % rises linearly from 0 to 100.
    pct = [100.0 * w / weeks[-1] for w in weeks]
    out = run_sandboxed(_S_CURVE, state={"weeks": weeks, "cumulative_pct": pct, "query_week": query})
    assert out.success, out.error
    # PCHIP reproduces linear data exactly.
    assert out.result["at_query"] == pytest.approx(100.0 * query / weeks[-1], abs=1e-9)
    assert out.result["at_knots"] == pytest.approx(pct, abs=1e-9)


@pytest.mark.parametrize(
    "weeks, pct, query",
    [
        ([0.0, 2.0, 5.0, 9.0, 12.0, 14.0], [0.0, 3.0, 20.0, 70.0, 94.0, 100.0], 7.0),
        ([0.0, 4.0, 6.0, 10.0, 16.0, 20.0, 22.0], [0.0, 1.5, 8.0, 41.0, 88.0, 99.0, 100.0], 13.0),
    ],
)
def test_s_curve_interpolation_is_monotone_on_an_s_shaped_dataset(weeks, pct, query):
    out = run_sandboxed(_S_CURVE, state={"weeks": weeks, "cumulative_pct": pct, "query_week": query})
    assert out.success, out.error
    assert out.result["at_knots"] == pytest.approx(pct, abs=1e-9)
    grid = out.result["grid"]
    assert all(b >= a - 1e-12 for a, b in zip(grid, grid[1:]))  # cumulative never falls
    # Bracketed by the knots either side of the query week.
    i = next(k for k, w in enumerate(weeks) if w > query)
    assert pct[i - 1] < out.result["at_query"] < pct[i]


# --------------------------------------------------------------------------
# Integration: area under a curve
# --------------------------------------------------------------------------

_AREA = """\
from scipy.integrate import quad
area, err = quad(lambda x: c2 * x * x + c1 * x + c0, x_from, x_to)
result = area
"""


@pytest.mark.parametrize(
    "c2, c1, c0, x_from, x_to",
    [(0.5, -1.0, 3.0, 0.0, 6.0), (-0.2, 2.0, 0.0, 1.0, 9.0)],
)
def test_integration_matches_the_antiderivative(c2, c1, c0, x_from, x_to):
    out = run_sandboxed(_AREA, state={"c2": c2, "c1": c1, "c0": c0, "x_from": x_from, "x_to": x_to})
    assert out.success, out.error

    def anti(x):
        return c2 * x ** 3 / 3 + c1 * x ** 2 / 2 + c0 * x

    assert out.result == pytest.approx(anti(x_to) - anti(x_from), rel=1e-9)


# --------------------------------------------------------------------------
# Symbolic simplification, then numeric evaluation through lambdify
# --------------------------------------------------------------------------

_SYMBOLIC = """\
import sympy
x, a = sympy.symbols("x a")
expr = sympy.simplify((x ** 2 - a ** 2) / (x - a) + sympy.sin(x) ** 2 + sympy.cos(x) ** 2)
f = sympy.lambdify((x, a), expr, "numpy")
result = {"simplified": str(expr), "value": float(f(x_value, a_value))}
"""


@pytest.mark.parametrize("x_value, a_value", [(3.0, 1.5), (-2.25, 7.0)])
def test_symbolic_simplification_then_numeric_evaluation(x_value, a_value):
    out = run_sandboxed(_SYMBOLIC, state={"x_value": x_value, "a_value": a_value})
    assert out.success, out.error
    # (x^2 - a^2)/(x - a) + sin^2 + cos^2 == x + a + 1
    assert out.result["simplified"].replace(" ", "") in {"a+x+1", "x+a+1"}
    assert out.result["value"] == pytest.approx(x_value + a_value + 1.0, rel=1e-12)


# --------------------------------------------------------------------------
# Policy: scipy's numeric subpackages pass; its file/network ones do not,
# whichever way the code reaches them
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub", sorted(ALLOWED_SUBPACKAGES["scipy"]))
def test_every_allowed_scipy_subpackage_imports(sub):
    for code in (f"import scipy.{sub}\nresult = 1",
                 f"from scipy import {sub}\nresult = 1",
                 f"import scipy\nm = scipy.{sub}\nresult = 1"):
        out = run_sandboxed(code)
        assert out.success, (code, out.error)


@pytest.mark.parametrize("sub", ["io", "datasets"])
def test_scipy_file_and_network_subpackages_are_blocked_every_way(sub):
    for code in (f"import scipy.{sub}\nresult = 1",
                 f"from scipy import {sub}\nresult = 1",
                 f"import scipy\nm = scipy.{sub}\nresult = 1"):
        out = run_sandboxed(code)
        assert out.success is False, code
        assert out.error_type == "ImportError", (code, out.error)
        assert f"scipy.{sub}" in out.error


@pytest.mark.parametrize(
    "entry, member, sibling",
    [
        ("app.lib.pm_computations", "compute_cpm", "app.core"),
        ("app.schemas.cpm", "CPMInput", "app.prompts"),
    ],
)
def test_attribute_walk_from_a_dotted_entry_reaches_only_that_entry(entry, member, sibling):
    # `import app.x.y` binds `app`; the walk to the entry works, a module
    # beside it under the same root does not.
    ok = run_sandboxed(f"import {entry}\nresult = {entry}.{member} is not None")
    assert ok.success and ok.result is True, ok.error
    blocked = run_sandboxed(f"import {entry}\nresult = {sibling}")
    assert blocked.success is False
    assert blocked.error_type == "ImportError"
    assert sibling in blocked.error


def test_every_module_the_prompt_advertises_is_importable():
    prompt = build_codegen_prompt("anything", {})
    for module in sorted(ALLOWED_MODULES):
        assert f"`{module}`" in prompt
        out = run_sandboxed(f"import {module}\nresult = 1")
        assert out.success, (module, out.error)


# --------------------------------------------------------------------------
# End to end through formula_executor_v2 (LLM replaced by a fixed reply)
# --------------------------------------------------------------------------

class _FixedReplyBlock(FormulaExecutorV2Block):
    def __init__(self, reply, **kw):
        super().__init__(**kw)
        self._reply = reply

    async def _call_llm(self, prompt: str) -> str:
        return f"```python\n{self._reply}```"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variables",
    [
        {"shutoff_head_m": 40.0, "pump_coeff": 0.010, "static_head_m": 10.0, "friction_coeff": 0.002},
        {"shutoff_head_m": 65.0, "pump_coeff": 0.004, "static_head_m": 22.0, "friction_coeff": 0.0015},
    ],
)
async def test_executor_runs_generated_scipy_code(variables):
    block = _FixedReplyBlock(_DUTY_POINT)
    out = await block.process({
        "task": "flow where the pump curve meets the system curve",
        "variables": variables,
    })
    assert out["status"] == "success", out.get("error")
    q_exact = math.sqrt(
        (variables["shutoff_head_m"] - variables["static_head_m"])
        / (variables["pump_coeff"] + variables["friction_coeff"])
    )
    assert out["result"]["flow"] == pytest.approx(q_exact, rel=1e-9)
    assert out["attempts"] == 1


# --------------------------------------------------------------------------
# The policy helpers themselves (the importer and attribute guard are
# installed into the jail as values, so they are also tested by name)
# --------------------------------------------------------------------------

import importlib  # noqa: E402

from app.core.sandbox import (  # noqa: E402
    _guarded_getattr,
    _guarded_import,
    _module_reachable,
    module_allowed,
)


@pytest.mark.parametrize(
    "name, allowed, reachable",
    [
        ("numpy.linalg", True, True),
        ("scipy.optimize", True, True),
        ("scipy.io", False, False),
        ("app.lib", False, True),   # parent of a dotted entry: walk-through only
        ("app.core", False, False),
        ("os", False, False),
    ],
)
def test_module_policy(name, allowed, reachable):
    assert module_allowed(name) is allowed
    assert _module_reachable(name) is reachable


@pytest.mark.parametrize(
    "package, attr",
    [("scipy", "optimize"), ("numpy", "linalg"), ("app.lib", "pm_computations")],
)
def test_guarded_getattr_returns_an_allowed_module(package, attr):
    importlib.import_module(f"{package}.{attr}")
    pkg = importlib.import_module(package)
    assert _guarded_getattr(pkg, attr) is getattr(pkg, attr)


@pytest.mark.parametrize("package, attr", [("scipy", "io"), ("app", "core")])
def test_guarded_getattr_refuses_a_module_outside_the_policy(package, attr):
    importlib.import_module(f"{package}.{attr}")
    with pytest.raises(ImportError, match=f"{package}.{attr}"):
        _guarded_getattr(importlib.import_module(package), attr)


@pytest.mark.parametrize("obj, attr", [(math, "pi"), ([3, 1, 2], "count")])
def test_guarded_getattr_passes_plain_attributes(obj, attr):
    assert _guarded_getattr(obj, attr) == getattr(obj, attr)


@pytest.mark.parametrize(
    "name, fromlist, expected",
    [("math", (), "math"), ("scipy.optimize", (), "scipy"), ("scipy", ("optimize",), "scipy")],
)
def test_guarded_import_returns_the_module(name, fromlist, expected):
    assert _guarded_import(name, None, None, fromlist, 0).__name__ == expected


@pytest.mark.parametrize(
    "name, fromlist, level",
    [("os", (), 0), ("scipy", ("io",), 0), ("scipy.datasets", (), 0), ("math", (), 1)],
)
def test_guarded_import_refuses(name, fromlist, level):
    with pytest.raises(ImportError):
        _guarded_import(name, None, None, fromlist, level)
