"""D1 rule C, part 2: a constant a calculator's body applies is DECLARED with
its unit and source, and the door states exactly the declared constants the
run read -- with the source, or flagged as still needing one.

Every calculator and constant here is invented in this file; nothing reads a
real calculator's values.
"""

from __future__ import annotations

import pytest

from app.lib import formula_registry
from app.lib.formula_registry import Constant, Constants
from tests.calc_entry_exam import checker

_K = Constants(
    plate_factor=Constant(3.25, "ratio", "invented plate factor", source="Invented Code 7 cl. 2"),
    edge_factor=Constant(0.45, "ratio", "invented edge factor"),
)


def _plated(a: float, edge: int = 0) -> float:
    """Invented plated strip."""
    if edge:
        return a * _K.edge_factor
    return a * _K.plate_factor


def _twice(a: float) -> float:
    """Invented strip read twice."""
    return a * _K.plate_factor + _K.plate_factor


def _no_constant(a: float) -> float:
    """Invented plain strip."""
    return a + a


@pytest.fixture
def invented(monkeypatch):
    from app.lib import construction_formulas as cf

    names = {"zz_invented_plated": _plated, "zz_invented_twice": _twice,
             "zz_invented_plain": _no_constant}
    for name, fn in names.items():
        inputs = {"a": formula_registry.Param("m", 0, 1000, label="length a")}
        if fn is _plated:
            inputs["edge"] = formula_registry.Param("count", 0, 1, label="edge strip")
        formula_registry.formula(
            owner=formula_registry.BASE,
            description="Invented calculator for the declared-constant door test.",
            display_name=name.replace("_", " "), inputs=inputs,
            outputs={"value": "-"}, name=name,
        )(fn)
        monkeypatch.setitem(cf.CALCULATORS, name, fn)
    yield names
    for name in names:
        formula_registry._REGISTRY.pop(name, None)


def _run(name, params):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(name, params)
    assert env["status"] == "success", env
    return env


@pytest.mark.parametrize(
    ("params", "value", "stated", "not_stated"),
    [
        ({"a": 4, "edge": 0}, 13.0, "plate_factor", "edge_factor"),
        ({"a": 10, "edge": 1}, 4.5, "edge_factor", "plate_factor"),
    ],
)
def test_door_states_only_the_constants_this_run_read(invented, params, value, stated, not_stated):
    env = _run("zz_invented_plated", params)
    assert env["result"]["value"] == pytest.approx(value)
    names = [r["name"] for r in env["result"]["constants_used"]]
    assert names == [stated]
    assert not_stated not in names


@pytest.mark.parametrize(
    ("params", "name", "value", "source"),
    [
        ({"a": 2, "edge": 0}, "plate_factor", 3.25, "Invented Code 7 cl. 2"),
        ({"a": 8, "edge": 1}, "edge_factor", 0.45, None),
    ],
)
def test_a_row_carries_its_source_or_says_it_needs_one(invented, params, name, value, source):
    row = _run("zz_invented_plated", params)["result"]["constants_used"][0]
    assert row["name"] == name
    assert row["value"] == pytest.approx(value)
    assert row["unit"] == "ratio"
    if source is None:
        assert row["needs_source"] is True
        assert "source" not in row
    else:
        assert row["source"] == source
        assert "needs_source" not in row


@pytest.mark.parametrize(("a", "value"), [(2, 9.75), (6, 22.75)])
def test_a_constant_read_twice_is_stated_once(invented, a, value):
    env = _run("zz_invented_twice", {"a": a})
    assert env["result"]["value"] == pytest.approx(value)
    assert [r["name"] for r in env["result"]["constants_used"]] == ["plate_factor"]


@pytest.mark.parametrize(("a", "value"), [(3, 6.0), (5, 10.0)])
def test_a_run_reading_no_constant_adds_no_rows(invented, a, value):
    env = _run("zz_invented_plain", {"a": a})
    assert env["result"]["value"] == pytest.approx(value)
    assert "constants_used" not in env["result"]


@pytest.mark.parametrize(("a", "value"), [(4, 13.0), (2, 6.5)])
def test_a_direct_call_reads_the_plain_value_and_records_nothing(a, value):
    with formula_registry.recording_constants() as outer:
        pass
    assert _plated(a) == pytest.approx(value)
    assert outer.rows == []


@pytest.mark.parametrize("bad", ["missing_factor", "plate"])
def test_an_undeclared_constant_is_an_error_and_declared_ones_are_read_only(bad):
    with pytest.raises(AttributeError):
        getattr(_K, bad)
    with pytest.raises(AttributeError):
        setattr(_K, "plate_factor", 9.0)


@pytest.mark.parametrize("value", [0.0, "3.25"])
def test_a_constant_must_be_declared_as_a_constant(value):
    with pytest.raises(TypeError):
        Constants(x=value)


# ------------------------------------------------- the exam sees a declared read
@pytest.mark.parametrize(
    ("params", "seen", "unseen"),
    [
        ({"a": 4, "edge": 0}, {"plate_factor": 3.25}, "edge_factor"),
        ({"a": 4, "edge": 1}, {"edge_factor": 0.45}, "plate_factor"),
    ],
)
def test_rule_c_probe_sees_a_declared_constant_read_on_the_run(params, seen, unseen):
    _, lines = checker.traced_lines(lambda: _plated(**params), _plated)
    found = checker.named_constants(_plated, lines)
    for name, value in seen.items():
        assert found[name] == pytest.approx(value)
    assert unseen not in found


@pytest.mark.parametrize(
    ("constants", "rows", "passes"),
    [
        ({"plate_factor": 3.25}, [], False),
        ({"plate_factor": 3.25}, [{"name": "plate_factor", "value": 3.25}], True),
        ({"edge_factor": 0.45}, [{"name": "plate_factor", "value": 3.25}], False),
        ({"edge_factor": 0.45}, [{"name": "edge_factor", "value": 0.45}], True),
    ],
)
def test_rule_c_fails_a_declared_constant_the_envelope_does_not_state(constants, rows, passes):
    env = {"status": "success", "result": {"value": 1.0, "constants_used": rows}}
    ok, why = checker.check_stated(env, constants)
    assert ok is passes, why
