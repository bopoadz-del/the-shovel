"""W4: a default an answer names is stated without the machinery.

The default line is written by the platform, not the model: "The <input>
used, <value>, is ... default, not a figure from the project documents."
It used to say "the platform calculator's default". The answer contract's
rule 3 counts "calculator" as machinery, and the line survived only because
its "not" read as a refusal. These tests hold the line to the contract's own
measure of machinery, and check that the contract keeps it whole.

Every formula here is invented and registered only for the test.
"""
from __future__ import annotations

import contextlib

import pytest

from app.agents import answer_contract
from app.lib import formula_registry
from app.lib.formula_registry import FormulaSpec, Param
from app.lib.source_labels import calculator_default_lines, default_sentence


def _ledge_reach(span_m: float, mode: str = "cantilevered", allowance: float = 1.15) -> dict:
    return {"reach_m": round(span_m * allowance, 3), "mode": mode, "allowance": allowance}


def _trough_fill(depth_m: float, grade: str = "fine", factor: float = 0.85) -> dict:
    return {"fill_m3": round(depth_m * factor, 3), "grade": grade, "factor": factor}


_SPECS = {
    "zz_ledge_reach": FormulaSpec(
        "zz_ledge_reach", _ledge_reach, "design", "Invented reach.", display_name="Ledge reach",
        inputs={"span_m": "m", "mode": "-", "allowance": "ratio"}, outputs={"reach_m": "m"},
        params={"span_m": Param("m", 0, 100, label="span"), "mode": Param("-", label="support mode"),
                "allowance": Param("ratio", 1, 2, label="reach allowance")},
    ),
    "zz_trough_fill": FormulaSpec(
        "zz_trough_fill", _trough_fill, "qaqc", "Invented fill.", display_name="Trough fill",
        inputs={"depth_m": "m", "grade": "-", "factor": "ratio"}, outputs={"fill_m3": "m3"},
        params={"depth_m": Param("m", 0, 50, label="trough depth"), "grade": Param("-", label="aggregate grade"),
                "factor": Param("ratio", 0, 1, label="compaction factor")},
    ),
}


@contextlib.contextmanager
def _registered():
    saved = {k: formula_registry._REGISTRY.get(k) for k in _SPECS}
    formula_registry._REGISTRY.update(_SPECS)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                formula_registry._REGISTRY.pop(k, None)
            else:
                formula_registry._REGISTRY[k] = v


#: (formula, the input it supplies, the defaults the result shows)
CASES = [
    ("zz_ledge_reach", {"span_m": 4.0}, ("cantilevered", "1.15")),
    ("zz_trough_fill", {"depth_m": 2.0}, ("fine", "0.85")),
]


@pytest.mark.parametrize("name,inputs,shown", CASES, ids=[c[0] for c in CASES])
def test_default_lines_carry_no_machinery(name, inputs, shown):
    with _registered():
        result = _SPECS[name].fn(**inputs)
        lines = calculator_default_lines(name, result, user_text="", passed=inputs)
    assert len(lines) == 2, lines
    for value in shown:
        assert any(value in line for line in lines), (value, lines)
    for line in lines:
        assert not answer_contract._MACHINE_RE.search(line), line
        assert "not a figure from the project documents" in line


@pytest.mark.parametrize("name,inputs,shown", CASES, ids=[c[0] for c in CASES])
def test_contract_keeps_the_default_line_whole(name, inputs, shown):
    with _registered():
        result = _SPECS[name].fn(**inputs)
        lines = calculator_default_lines(name, result, user_text="", passed=inputs)
    head = f"The figure is {next(iter(result.values()))}."
    text = "\n".join([head, *lines])
    out = answer_contract.apply(text, computed=True).text
    for line in lines:
        assert line in out, (line, out)


@pytest.mark.parametrize("name,key,value,label", [
    ("zz_ledge_reach", "mode", "propped_end", "support mode"),
    ("zz_trough_fill", "factor", 0.9, "compaction factor"),
])
def test_one_default_sentence_names_the_input_and_the_value(name, key, value, label):
    with _registered():
        line = default_sentence(name, key, value)
    assert line.startswith(f"The {label} used, ")
    assert ("propped end" if isinstance(value, str) else "0.9") in line
    assert "assumed default" in line
    assert "calculator" not in line.lower()
