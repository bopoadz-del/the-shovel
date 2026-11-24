"""KS-4: every schedule path runs its CPM through the calculation envelope.

``run_cpm`` is the one door: it runs the registered ``cpm_network`` kernel by
``run_calculation`` and hands back the same ``CPMOutput`` the bare pass
gives. Product code outside the kernel never calls ``compute_cpm`` itself.
"""
from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

from app.lib import construction_formulas as cf
from app.lib.pm_computations import CircularDependencyError, compute_cpm, run_cpm
from app.schemas.cpm import CPMInput, WorkCalendar

ROOT = Path(__file__).resolve().parents[1]

# The kernel itself and the module that defines the pass are the only places
# allowed to call the bare pass.
_PASS_OWNERS = {
    "app/lib/pm_computations.py",
    "app/agents/hats/planning/formulas/construction_formulas_planning.py",
}


def _typed_network():
    return [
        {"id": "P1", "name": "set out", "duration": 4},
        {"id": "P2", "duration": 6,
         "predecessors": [{"predecessor_id": "P1", "type": "SS", "lag": 2}]},
        {"id": "P3", "duration": 3, "predecessors": [{"predecessor_id": "P1"}]},
        {"id": "P4", "duration": 2,
         "predecessors": [{"predecessor_id": "P2"},
                          {"predecessor_id": "P3", "type": "FS", "lag": 1}]},
    ]


def _chain_network():
    return [
        {"id": "K", "duration": 7},
        {"id": "L", "duration": 2, "predecessors": [{"predecessor_id": "K"}]},
        {"id": "M", "duration": 9},
    ]


@pytest.mark.parametrize("acts, settings", [
    (_typed_network(), {}),
    (_chain_network(), {"project_start": date(2027, 3, 4),
                        "calendar": WorkCalendar(work_weekdays=[0, 1, 2, 3, 4, 5],
                                                 holidays=[date(2027, 3, 9)]),
                        "near_critical_band_days": 3}),
])
def test_envelope_output_equals_the_bare_pass(acts, settings):
    data = CPMInput(activities=acts, **settings)
    assert run_cpm(data) == compute_cpm(data)


@pytest.mark.parametrize("band, expect_named", [(None, False), (4, True)])
def test_near_critical_only_inside_a_stated_band(band, expect_named):
    data = CPMInput(activities=_chain_network(), near_critical_band_days=band)
    out = run_cpm(data)
    assert (out.near_critical is not None) is expect_named


@pytest.mark.parametrize("start, dated", [(None, False), (date(2027, 1, 11), True)])
def test_dates_only_from_a_stated_start(start, dated):
    out = run_cpm(CPMInput(activities=_typed_network(), project_start=start))
    assert (out.project_finish is not None) is dated
    assert all((r.early_start is not None) is dated for r in out.results)


def test_run_cpm_goes_through_run_calculation(monkeypatch):
    seen = []
    real = cf.run_calculation

    def spy(name, params=None):
        seen.append(name)
        return real(name, params)

    monkeypatch.setattr(cf, "run_calculation", spy)
    run_cpm(CPMInput(activities=_chain_network()))
    assert seen == ["cpm_network"]


@pytest.mark.parametrize("acts, exc_type", [
    ([{"id": "R", "duration": 1, "predecessors": [{"predecessor_id": "S"}]},
      {"id": "S", "duration": 2, "predecessors": [{"predecessor_id": "R"}]}],
     CircularDependencyError),
    ([{"id": "T", "duration": 1}, {"id": "T", "duration": 3}], ValueError),
])
def test_refusal_comes_back_as_the_kind_the_kernel_names(acts, exc_type):
    with pytest.raises(ValueError) as info:
        run_cpm(CPMInput(activities=acts))
    assert type(info.value) is exc_type


@pytest.mark.parametrize("acts, kind", [
    ([{"id": "U", "duration": 1, "predecessors": "V"},
      {"id": "V", "duration": 1, "predecessors": "U"}], "CircularDependencyError"),
    ([{"id": "W", "duration": 1}, {"id": "W", "duration": 2}], "ValueError"),
])
def test_envelope_carries_the_refusal_kind(acts, kind):
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env["status"] == "error"
    assert env.get("error_type") == kind


def _bare_pass_calls(source: str):
    """Line numbers of every call to ``compute_cpm`` in ``source`` (by name
    or as an attribute); text inside strings is not a call."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "compute_cpm":
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("source, count", [
    ("from m import compute_cpm; out = compute_cpm(x)", 1),
    ("PROMPT = 'call compute_cpm(data)'; out = run_cpm(x)", 0),
])
def test_the_guard_counts_calls_not_text(source, count):
    assert len(_bare_pass_calls(source)) == count


def test_no_product_module_calls_the_bare_pass():
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in _PASS_OWNERS:
            continue
        for n in _bare_pass_calls(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{n}")
    assert offenders == []
