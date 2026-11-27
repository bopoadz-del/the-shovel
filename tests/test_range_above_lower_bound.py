"""A declared range can begin ABOVE its lower bound ("more than 0"): the bound
itself is refused, the first figure above it is accepted, and every sentence
that states the range says so. Inputs here are invented; nothing is keyed to
one calculator."""
from __future__ import annotations

import json
import math

import pytest

from app.agents.driver import catalogue
from app.agents.driver import tools as driver_tools
from app.agents.hats.qaqc import validation as qaqc_validation
from app.blocks import _knowledge
from app.lib import formula_registry as fr
from app.lib import source_labels


def _spec(lo, hi, above, unit="m", required=True):
    return fr.ParamSpec(name="span", kind="number", unit=unit, lo=lo, hi=hi, label="span",
                        grade="", required=required, above_lo=above)


@pytest.mark.parametrize("lo, hi", [(0, 50), (2.5, 1e6)])
def test_the_bound_itself_is_refused_and_the_next_figure_is_accepted(lo, hi):
    spec = _spec(lo, hi, True)
    assert fr._in_range(spec, lo) is False
    assert fr._in_range(spec, math.nextafter(lo, math.inf)) is True
    assert fr._in_range(spec, hi) is True
    assert fr._in_range(spec, math.nextafter(hi, math.inf)) is False
    assert fr._in_range(spec, lo - 1) is False


@pytest.mark.parametrize("lo, hi", [(0, 50), (2.5, 1e6)])
def test_an_ordinary_range_still_includes_both_bounds(lo, hi):
    spec = _spec(lo, hi, False)
    assert fr._in_range(spec, lo) is True and fr._in_range(spec, hi) is True
    assert fr._in_range(spec, math.nextafter(lo, -math.inf)) is False
    assert fr._in_range(spec, math.nextafter(hi, math.inf)) is False


@pytest.mark.parametrize("lo, hi", [(0, 5), (2.5, 40)])
def test_a_range_is_inclusive_unless_declared_otherwise(lo, hi):
    assert fr.Param("m", lo, hi).above_lo is False
    assert fr.ParamSpec(name="span", kind="number", unit="m", lo=lo, hi=hi, label="span",
                        grade="", required=True).above_lo is False


@pytest.mark.parametrize("lo, hi, above, words", [
    (0, 5, True, "more than 0, up to 5 m"),
    (2.5, 40, True, "more than 2.5, up to 40 m"),
    (0, None, True, "more than 0 m"),
    (0, 5, False, "0 to 5 m"),
    (0, None, False, "at least 0 m"),
    (None, 5, False, "at most 5 m"),
])
def test_the_range_in_words(lo, hi, above, words):
    assert fr._range_words(lo, hi, "m", above) == words


@pytest.mark.parametrize("unit, lo, hi, above, words", [
    ("m", 0, 50, True, "more than 0 m, up to 50 m"),
    ("working days", 0, 100000, True, "more than 0 working days, up to 100,000 working days"),
    ("m", 0, 50, False, "0 m to 50 m"),
])
def test_the_users_refusal_states_the_range_as_declared(unit, lo, hi, above, words):
    row = {"parameter": "span", "label": "span", "unit": unit, "value": lo, "reason": "out_of_range",
           "lo": lo, "hi": hi, "kind": "number", "above_lo": above}
    assert source_labels._accepted_range(row) == words
    row.pop("above_lo")  # a row written before this field existed reads as inclusive
    assert "more than" not in source_labels._accepted_range(row)


@pytest.mark.parametrize("lo, hi, above, part", [
    (0, 50, True, "m, more than 0, up to 50"),
    (2.5, 1e6, True, "m, more than 2.5, up to 1e+06"),
    (0, 50, False, "m, 0 to 50"),
])
def test_the_models_input_line_states_the_range_as_declared(lo, hi, above, part):
    assert part in driver_tools._input_line("span", _spec(lo, hi, above))


@pytest.mark.parametrize("lo, hi", [(None, None), (0, None), (None, 50)])
def test_an_input_line_with_no_whole_range_states_none(lo, hi):
    line = driver_tools._input_line("span", _spec(lo, hi, lo is not None))
    assert "0 to" not in line and "more than" not in line and "up to" not in line


@pytest.mark.parametrize("name, above", [("near_critical_band_days", True), ("activities", False),
                                         ("project_start", False)])
def test_the_declared_bound_reaches_the_inputs_of_a_registered_calculator(name, above):
    params = fr.parameters(fr.get("cpm_network"))
    assert params[name].above_lo is above
    if above:
        assert (params[name].lo, params[name].hi) == (0, 100000)
    others = [p for calc, spec in ((s.name, s) for s in fr.specs_for(None))
              for p in fr.parameters(spec).values() if calc != "cpm_network"]
    assert others and not any(p.above_lo for p in others)


def _shipped(tmp_path, monkeypatch, value):
    path = tmp_path / "shipped.json"
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "t",
                                "entries": [{"id": "x.lim", "thresholds": {"v": value}, "provenance": {}}]}),
                    encoding="utf-8")
    monkeypatch.setattr(_knowledge, "_KB_PATH", str(path))


@pytest.mark.parametrize("stated", [0.0, -1.0])
def test_a_knowledge_figure_at_or_below_the_bound_is_refused_with_the_range(tmp_path, monkeypatch, stated):
    _shipped(tmp_path, monkeypatch, stated)
    constant = fr.Constant.from_knowledge("x.lim", "thresholds.v", "m",
                                          within=fr.Param("ratio", 0.0, 5.0, label="invented input", above_lo=True))
    with pytest.raises(LookupError, match=r"\(more than 0, up to 5 ratio\)"):
        constant.resolve()


@pytest.mark.parametrize("stated", [5e-324, 5.0])
def test_a_knowledge_figure_above_the_bound_is_read(tmp_path, monkeypatch, stated):
    _shipped(tmp_path, monkeypatch, stated)
    constant = fr.Constant.from_knowledge("x.lim", "thresholds.v", "m",
                                          within=fr.Param("ratio", 0.0, 5.0, label="invented input", above_lo=True))
    assert constant.resolve().value == stated


# The QA/QC calculators are held to their declared ranges by their own check
# (app.agents.hats.qaqc.validation): it reads the same declaration.

@pytest.mark.parametrize("lo, hi, value, refused", [
    (0, 5, 0, True),                       # the bound itself
    (0, 5, -1, True),
    (0, 5, 0.5, False),
    (0, 5, 5, False),
    (0, 5, 5.5, True),
    (2.5, 40, 2.5, True), (2.5, 40, 2.6, False),
])
def test_the_qaqc_range_check_reads_the_same_declaration(lo, hi, value, refused):
    spec = _spec(lo, hi, True, unit="ratio")
    assert qaqc_validation._in_range(spec, value) is (not refused)
    assert qaqc_validation._in_range(spec, value) is fr._in_range(spec, value)
    out = qaqc_validation.first_out_of_range({"span": spec}, {"span": value})
    assert (out is not None) is refused
    if refused:
        assert out["question"] == f"span must be more than {lo:g} and at most {hi:g} ratio. Please confirm the span."
        assert out["above_lo"] is True and out["status"] == "out_of_range"


@pytest.mark.parametrize("lo, hi, value, refused", [(0, 5, 0, False), (0, 5, -1, True), (2.5, 40, 2.5, False),
                                                    (2.5, 40, 41, True)])
def test_the_qaqc_range_check_keeps_an_ordinary_range_inclusive(lo, hi, value, refused):
    spec = _spec(lo, hi, False, unit="ratio")
    assert qaqc_validation._in_range(spec, value) is (not refused)
    out = qaqc_validation.out_of_range_result(spec, value)
    assert out["question"] == f"span must be between {lo:g} and {hi:g} ratio. Please confirm the span."
    assert out["above_lo"] is False
    assert (qaqc_validation.first_out_of_range({"span": spec}, {"span": value}) is not None) is refused


# NaN compares False to every bound: it is inside no range that has one,
# whichever bound is missing and whether or not the floor itself is accepted.

_NAN = float("nan")


@pytest.mark.parametrize("lo, hi, above", [
    (0, None, False), (0, None, True),        # a floor and no ceiling
    (None, 5, False),                         # a ceiling and no floor
    (0, 5, False), (0, 5, True),
    (2.5, None, True), (-10, None, False),
])
def test_a_number_that_is_not_a_number_is_inside_no_range_that_has_a_bound(lo, hi, above):
    spec = _spec(lo, hi, above, unit="ratio")
    assert fr._in_range(spec, _NAN) is False
    assert qaqc_validation._in_range(spec, _NAN) is False
    if lo is not None and hi is not None:
        # and it is what the calculator's caller is told: refused, never computed on
        assert qaqc_validation.first_out_of_range({"span": spec}, {"span": _NAN}) is not None


@pytest.mark.parametrize("lo, hi, above, value, inside", [
    # the same answers main gave for an ordinary range, bound by bound
    (0, None, False, 0, True), (0, None, False, -0.001, False), (0, None, False, 1e300, True),
    (0, None, False, float("inf"), True), (0, None, False, float("-inf"), False),
    (None, 5, False, 5, True), (None, 5, False, 5.001, False), (None, 5, False, float("-inf"), True),
    (None, None, False, 123.0, True),
    # and with the floor itself not accepted
    (0, None, True, 0, False), (0, None, True, 5e-324, True), (0, None, True, float("inf"), True),
])
def test_a_range_with_one_bound_missing_checks_the_bound_it_has(lo, hi, above, value, inside):
    spec = _spec(lo, hi, above, unit="ratio")
    assert fr._in_range(spec, value) is inside
    assert qaqc_validation._in_range(spec, value) is inside
    if not above:
        # main's own expression for an ordinary range
        assert inside is ((lo is None or value >= lo) and (hi is None or value <= hi))


# The catalogue entry the model reads states the same declaration.

@pytest.mark.parametrize("lo, hi, above, want", [
    (0, 50, True, True), (2.5, 1e6, True, True), (0, None, True, True),
    (0, 50, False, False), (None, 50, True, False), (None, None, True, False),
])
def test_the_catalogue_input_says_when_the_lower_bound_is_not_accepted(lo, hi, above, want):
    got = catalogue._input("span", _spec(lo, hi, above))
    assert got["range"] == ([lo, hi] if lo is not None or hi is not None else None)
    assert (got.get(catalogue.LOWER_BOUND_EXCLUDED) is True) is want
    assert (catalogue.LOWER_BOUND_EXCLUDED in got) is want       # absent, not false, on an ordinary range


@pytest.mark.parametrize("name, excluded", [("near_critical_band_days", True), ("activities", False),
                                            ("project_start", False)])
def test_the_catalogue_entry_of_a_registered_calculator_carries_it(name, excluded):
    entry = next(e for e in catalogue.entries() if e["id"] == "formula:cpm_network")
    row = next(i for i in entry["inputs"] if i["name"] == name)
    assert (row.get(catalogue.LOWER_BOUND_EXCLUDED) is True) is excluded
    if excluded:
        assert row["range"] == [0, 100000]
    others = [i for e in catalogue.entries() if e["id"] != "formula:cpm_network" for i in e.get("inputs") or []]
    assert others and not any(catalogue.LOWER_BOUND_EXCLUDED in i for i in others)

