"""The core kernel carries VERIFIED: the line extension (quantity x rate)
declares its standard and reproduces the hand values in
tests/hand_values/bill_line_extension.json, so ``certified`` is derived True
through the public calculation path -- and is lost again the moment a hand
value or the declaration is taken away.

Each decision runs on at least two inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.base.formulas.construction_formulas_commercial import unit_cost_total
from app.lib import formula_registry as fr
from app.lib.construction_formulas import run_calculation

_KERNEL = "unit_cost_total"
_CASES = Path(__file__).parent / "hand_values" / "bill_line_extension.json"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("FORMULA_HAND_VALUES_DIR", raising=False)
    fr.clear_hand_value_cache()
    yield
    fr.clear_hand_value_cache()


def _cases() -> list:
    return json.loads(_CASES.read_text(encoding="utf-8"))


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_the_kernel_declares_a_standard_and_reproduces_every_hand_case(index):
    spec = fr.get(_KERNEL)
    assert spec is not None and spec.standard.strip()
    assert len(fr.hand_value_cases(_KERNEL)) == len(_cases()) >= 2
    assert fr.hand_value_failures(_KERNEL) == []
    assert fr.certified(_KERNEL)
    assert not [g for g in fr.certification_gaps() if g.startswith(f"formula:{_KERNEL}:")]
    row = _cases()[index]
    got = unit_cost_total(**row["inputs"])
    for key, want in row["expect"].items():
        assert got[key] == pytest.approx(want, rel=row.get("rel_tol", 1e-3))


# The kernel itself, imported from its own module: the total is the product
# rounded to 2 dp, and the result names the same standard the spec declares.
@pytest.mark.parametrize("qty,rate,total", [(3, 19.99, 59.97), (250, 0.333, 83.25), (1, 0, 0.0)])
def test_unit_cost_total_extends_quantity_by_rate(qty, rate, total):
    got = unit_cost_total(qty, rate)
    assert got["total_cost"] == total
    assert got["standard"] == fr.get(_KERNEL).standard
    assert f"{qty}" in got["note"] and f"{total:.2f}" in got["note"]


@pytest.mark.parametrize("qty,rate,total", [(12, 4850, 58200.0), (37.5, 212.4, 7965.0)])
def test_the_public_envelope_carries_certified_and_the_declared_standard(qty, rate, total):
    env = run_calculation(_KERNEL, {"quantity": qty, "unit_rate": rate})
    assert env["certified"] is True
    assert env["provenance"] == "kernel"
    assert env["standard"] == fr.get(_KERNEL).standard
    assert env["result"]["total_cost"] == pytest.approx(total)


# 1 % high on one case, 2 % low on another: both outside the 0.1 % default.
@pytest.mark.parametrize("index,factor", [(0, 1.01), (1, 0.98)])
def test_a_hand_value_the_kernel_misses_takes_the_label_away(index, factor, tmp_path, monkeypatch):
    rows = _cases()
    key = next(iter(rows[index]["expect"]))
    rows[index]["expect"][key] *= factor
    (tmp_path / "cases.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setenv("FORMULA_HAND_VALUES_DIR", str(tmp_path))
    fr.clear_hand_value_cache()
    assert not fr.certified(_KERNEL)
    assert f"formula:{_KERNEL}:hand_value_mismatch" in fr.certification_gaps()
    env = run_calculation(_KERNEL, {"quantity": 3, "unit_rate": 5})
    assert env["certified"] is False


@pytest.mark.parametrize("keep", [0, 3])
def test_one_reproduced_case_with_the_standard_is_enough(keep, tmp_path, monkeypatch):
    (tmp_path / "cases.json").write_text(json.dumps([_cases()[keep]]), encoding="utf-8")
    monkeypatch.setenv("FORMULA_HAND_VALUES_DIR", str(tmp_path))
    fr.clear_hand_value_cache()
    assert fr.certified(_KERNEL)


@pytest.mark.parametrize("where", ["empty", "missing"])
def test_without_hand_cases_the_kernel_is_not_certified(where, tmp_path, monkeypatch):
    if where == "empty":
        (tmp_path / "cases.json").write_text("[]", encoding="utf-8")
        monkeypatch.setenv("FORMULA_HAND_VALUES_DIR", str(tmp_path))
    else:
        monkeypatch.setenv("FORMULA_HAND_VALUES_DIR", str(tmp_path / "absent"))
    fr.clear_hand_value_cache()
    assert not fr.certified(_KERNEL)
    assert run_calculation(_KERNEL, {"quantity": 2, "unit_rate": 9})["certified"] is False


@pytest.mark.parametrize("pick", [0, -1])
def test_the_core_hand_file_certifies_no_other_formula(pick, tmp_path, monkeypatch):
    """With only the core hand file read, a different registered formula --
    the first and the last by name -- has no case and is not certified."""
    (tmp_path / "cases.json").write_text(_CASES.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("FORMULA_HAND_VALUES_DIR", str(tmp_path))
    fr.clear_hand_value_cache()
    others = sorted(s.name for s in fr.all_specs() if s.name != _KERNEL)
    other = others[pick]
    assert not fr.hand_value_cases(other)
    assert not fr.certified(other)
    assert fr.certified(_KERNEL)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_the_hand_file_names_only_the_core_kernel_and_each_case_is_worked(index):
    rows = _cases()
    assert len(rows) == 4
    assert rows[index]["formula"] == _KERNEL
    assert str(rows[index]["working"]).strip()
