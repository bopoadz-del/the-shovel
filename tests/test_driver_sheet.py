"""The working sheet: the model writes givens / plan / missing; code writes
every result and resolves references; arithmetic runs in code with units.
Every input below is invented for the test."""
import pytest

from app.agents.base.formulas.arithmetic import arithmetic_step
from app.agents.driver import sheet as ws


def _sheet_with(givens, plan=None):
    s = ws.new_sheet("t-1", "q")
    s, problems = ws.apply_update(s, {"givens": givens, "plan": plan or []})
    assert problems == []
    return s


# --- who writes what -------------------------------------------------------

@pytest.mark.parametrize("foreign", [
    {"steps": [{"id": "s1", "status": "done", "result": {"value": 5}}]},
    {"done": True},
    {"givens": [], "result": 7},
])
def test_model_cannot_write_results_or_done(foreign):
    s = ws.new_sheet("t", "q")
    out, problems = ws.apply_update(s, foreign)
    assert problems and "written by code" in problems[0]
    assert out is s


@pytest.mark.parametrize("value,unit", [(20, "nr"), ("3,000", "m2")])
def test_given_is_stored_as_number_with_unit(value, unit):
    s = _sheet_with([{"id": "g1", "what": "w", "value": value, "unit": unit}])
    g = s["givens"][0]
    assert g["value"] == float(str(value).replace(",", ""))
    assert g["unit"] == unit
    assert g["source"] == "user"


@pytest.mark.parametrize("bad", [
    {"id": "g1", "value": "twenty", "unit": "nr"},   # not a number
    {"id": "g1", "value": 20, "unit": ""},           # no unit
    {"id": "x1", "value": 20, "unit": "nr"},         # wrong id shape
    {"id": "g1", "value": True, "unit": "nr"},       # wrong type
    {"id": "g1", "value": float("inf"), "unit": "nr"},
])
def test_bad_given_is_refused_and_sheet_unchanged(bad):
    s = ws.new_sheet("t", "q")
    out, problems = ws.apply_update(s, {"givens": [bad]})
    assert problems
    assert out is s and out["givens"] == []


@pytest.mark.parametrize("first,second", [(20, 24), (1.5, 0.9)])
def test_correction_replaces_given_by_id(first, second):
    s = _sheet_with([{"id": "g1", "value": first, "unit": "m"}])
    s, problems = ws.apply_update(s, {"givens": [{"id": "g1", "value": second, "unit": "m"}]})
    assert problems == []
    assert [g["value"] for g in s["givens"]] == [second]


def test_plan_reference_to_unknown_id_is_refused():
    s = _sheet_with([{"id": "g1", "value": 2, "unit": "m"}])
    out, problems = ws.apply_update(s, {"plan": [{"id": "s1", "entry": "e", "inputs": {"a": "g9"}}]})
    assert problems and "g9" in problems[0]
    out2, problems2 = ws.apply_update(s, {"plan": [{"id": "s1", "entry": "e", "inputs": {"a": "g1"}}]})
    assert problems2 == [] and out2["plan"][0]["inputs"] == {"a": "g1"}


def test_plan_step_may_not_refer_to_itself():
    s = ws.new_sheet("t", "q")
    _, problems = ws.apply_update(s, {"plan": [{"id": "s1", "entry": "e", "inputs": {"a": "s1"}}]})
    assert problems and "itself" in problems[0]


def test_plan_step_needs_an_entry():
    s = ws.new_sheet("t", "q")
    _, problems = ws.apply_update(s, {"plan": [{"id": "s1", "entry": " ", "inputs": {}}]})
    assert problems and "no catalogue entry" in problems[0]


def test_update_that_is_not_an_object_is_refused():
    s = ws.new_sheet("t", "q")
    for bad in (None, [], "givens"):
        out, problems = ws.apply_update(s, bad)
        assert problems and out is s
    out, problems = ws.apply_update(s, {"givens": {"id": "g1"}})
    assert problems == ["givens must be a list"] and out is s


# --- references and results --------------------------------------------------

@pytest.mark.parametrize("a,b", [(12.0, 450.0), (3.5, 1200.4)])
def test_resolve_given_and_earlier_step(a, b):
    s = _sheet_with([{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "-"}],
                    [{"id": "s1", "entry": "e", "inputs": {"x": "g1", "y": "g2"}},
                     {"id": "s2", "entry": "e", "inputs": {"x": "s1", "mode": "net"}}])
    vals, problems = ws.resolve_inputs(s, "s1")
    assert problems == [] and vals == {"x": {"value": a, "unit": "m"}, "y": {"value": b, "unit": "-"}}
    # s2 waits for s1: never guessed
    vals2, problems2 = ws.resolve_inputs(s, "s2")
    assert problems2 and "s1" in problems2[0] and "x" not in vals2
    s = ws.record_result(s, "s1", "c1", {"result": arithmetic_step("multiply", a, b, "m", "-"),
                                         "units": {"result": "m"}}, output="result")
    vals2, problems2 = ws.resolve_inputs(s, "s2")
    assert problems2 == []
    assert vals2["x"] == {"value": round(a * b, 10), "unit": "m"}
    assert vals2["mode"] == "net"


def test_resolve_unknown_step():
    s = ws.new_sheet("t", "q")
    vals, problems = ws.resolve_inputs(s, "s4")
    assert vals == {} and problems


@pytest.mark.parametrize("tool_result,output,expect", [
    ({"result": {"volume_m3": 900.0, "loose": 1}, "units": {"volume_m3": "m3"}}, "volume_m3", (900.0, "m3")),
    ({"result": {"total_cost": "5,400.00"}}, "total_cost", (5400.0, "-")),
    ({"result": 42.5, "unit": "d"}, "", (42.5, "d")),
])
def test_record_result_reads_the_tool_result(tool_result, output, expect):
    s = _sheet_with([], [{"id": "s1", "entry": "e", "inputs": {}}])
    s = ws.record_result(s, "s1", "c9", tool_result, output)
    step = s["steps"][0]
    assert step["status"] == "done" and step["call_id"] == "c9"
    assert (step["result"]["value"], step["result"]["unit"]) == expect
    assert s["done"] is True


@pytest.mark.parametrize("tool_result,output", [
    ({"error": "area must be > 0"}, ""),
    ({"status": "error", "error": "Unknown calculation"}, ""),
    ({"result": {"a": 1}}, "b"),            # output not in result
    ({"result": {"a": "n/a"}}, "a"),        # not numeric
    ("plain text", ""),                     # not an object
])
def test_record_result_failure_is_failed_not_guessed(tool_result, output):
    s = _sheet_with([], [{"id": "s1", "entry": "e", "inputs": {}}])
    s = ws.record_result(s, "s1", "c1", tool_result, output)
    step = s["steps"][0]
    assert step["status"] == "failed" and step["error"] and "result" not in step
    assert s["done"] is False


def test_done_needs_every_plan_step_done():
    s = ws.new_sheet("t", "q")
    assert ws.is_done(s) is False  # empty plan is never done
    s = _sheet_with([], [{"id": "s1", "entry": "e", "inputs": {}}, {"id": "s2", "entry": "e", "inputs": {}}])
    s = ws.record_result(s, "s1", "c", {"result": 1})
    assert s["done"] is False
    s = ws.record_result(s, "s2", "c", {"result": 2})
    assert s["done"] is True


# --- missing inputs: asked once ----------------------------------------------

@pytest.mark.parametrize("asked,new", [("unit_rate", "duration"), ("crew_size", "area")])
def test_missing_asked_once_across_turns(asked, new):
    s = ws.new_sheet("t1", "q")
    s, _ = ws.apply_update(s, {"missing": [{"for": "s2", "input": asked, "label": asked}]})
    assert [m["input"] for m in ws.unasked(s)] == [asked]
    s = ws.mark_asked(s)
    assert ws.unasked(s) == []
    nxt = ws.new_sheet("t2", "q2", previous=s)
    # restating the same missing input keeps its asked mark
    nxt, _ = ws.apply_update(nxt, {"missing": [{"for": "s2", "input": asked},
                                               {"for": "s3", "input": new}]})
    assert [m["input"] for m in ws.unasked(nxt)] == [new]
    # supplying it: the model leaves it off the list
    nxt, _ = ws.apply_update(nxt, {"givens": [{"id": "g1", "value": 450, "unit": "-"}], "missing": []})
    assert nxt["missing"] == []


@pytest.mark.parametrize("inputs", [["a"], ["a", "b"]])
def test_update_without_missing_keeps_missing(inputs):
    s = ws.new_sheet("t", "q")
    s, _ = ws.apply_update(s, {"missing": [{"for": "s1", "input": i} for i in inputs]})
    s, _ = ws.apply_update(s, {"givens": [{"id": "g1", "value": 1, "unit": "m"}]})
    assert [m["input"] for m in s["missing"]] == inputs


@pytest.mark.parametrize("value,unit", [(20, "nr"), (3.5, "m")])
def test_previous_turn_givens_carry_over_marked(value, unit):
    s = _sheet_with([{"id": "g1", "value": value, "unit": unit}])
    nxt = ws.new_sheet("t2", "next", previous=s)
    assert nxt["givens"][0]["value"] == value and nxt["givens"][0]["unit"] == unit
    assert nxt["givens"][0]["source"] == "user"
    assert nxt["plan"] == [] and nxt["steps"] == []
    fresh = ws.new_sheet("t3", "q", previous=None)
    assert fresh["givens"] == []


@pytest.mark.parametrize("ids", [("g2", "g1"), ("g1", "g3")])
def test_render_is_stable(ids):
    givens = [{"id": ids[0], "value": 1, "unit": "m"}, {"id": ids[1], "value": 2, "unit": "m"}]
    a, b = _sheet_with(givens), _sheet_with([dict(g) for g in givens])
    assert ws.render(a) == ws.render(b)
    assert ws.render(a).startswith('{"done":false,"givens":')


# --- the arithmetic formula ---------------------------------------------------

@pytest.mark.parametrize("op,a,b,au,bu,value,unit", [
    ("divide", 2059200, 3000, "currency", "m2", 686.4, "currency/m2"),
    ("divide", 900, 450, "m3", "m3", 2.0, "-"),
    ("multiply", 12, 450, "m", "-", 5400.0, "m"),
    ("multiply", 2.5, 4, "m", "m", 10.0, "m*m"),
    ("add", 1.5, 0.25, "m", "m", 1.75, "m"),
    ("subtract", 900, 945, "m3", "M3", -45.0, "m3"),
    ("divide", 10, 4, "-", "d", 2.5, "1/d"),
    ("multiply", 3, 7, "-", "-", 21.0, "-"),
])
def test_arithmetic(op, a, b, au, bu, value, unit):
    out = arithmetic_step(op, a, b, au, bu)
    assert "error" not in out
    assert out["result"] == pytest.approx(value) and out["result_unit"] == unit
    assert f"{out['result']:g}" in out["method"]


@pytest.mark.parametrize("op,a,b,au,bu,fragment", [
    ("add", 1, 2, "m", "mm", "convert"),
    ("subtract", 5, 2, "m2", "m3", "convert"),
    ("divide", 5, 0, "m", "m", "zero"),
    ("power", 2, 3, "-", "-", "operator"),
    ("add", "x", 2, "-", "-", "numbers"),
    ("multiply", None, 2, "-", "-", "numbers"),
    ("add", float("nan"), 2, "-", "-", "numbers"),
    ("add", True, 2, "-", "-", "numbers"),
])
def test_arithmetic_refuses(op, a, b, au, bu, fragment):
    out = arithmetic_step(op, a, b, au, bu)
    assert fragment in out["error"] and "result" not in out


def test_arithmetic_is_a_registered_base_formula():
    from app.lib import formula_registry

    spec = formula_registry.get("arithmetic_step")
    assert spec is not None and spec.owner == "base" and spec.main_output == "result"
    assert "arithmetic_step" in {s.name for s in formula_registry.specs_for(None)}


# --- run_step: code runs the planned formula and writes the result -------------

@pytest.mark.parametrize("total,area,expect,unit", [
    (2059200, 3000, 686.4, "currency/m2"),
    (450000, 1200, 375.0, "currency/m2"),
])
def test_run_step_divides_through_code(total, area, expect, unit):
    s = _sheet_with([{"id": "g1", "value": total, "unit": "currency"},
                     {"id": "g2", "value": area, "unit": "m2"}],
                    [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "divide", "a": "g1", "b": "g2"}}])
    s = ws.run_step(s, "s1", "c1")
    step = s["steps"][0]
    assert step["status"] == "done"
    assert step["result"] == {"name": "result", "value": pytest.approx(expect), "unit": unit}
    assert step["method"].endswith(f"{expect:g} {unit}")
    assert s["done"] is True and ws.open_steps(s) == []


@pytest.mark.parametrize("a,b", [(12.0, 450.0), (3.5, 8.0)])
def test_run_step_chains_a_result_into_the_next_step(a, b):
    s = _sheet_with([{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"},
                     {"id": "g3", "value": 2, "unit": "-"}],
                    [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "g1", "b": "g2"}},
                     {"id": "s2", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "divide", "a": "s1", "b": "g3"}}])
    s = ws.run_step(s, "s2", "c0")  # s1 has not run: s2 is not guessed
    assert s["steps"][0]["status"] == "failed" and "s1" in s["steps"][0]["error"]
    assert ws.open_steps(s) == ["s1", "s2"]
    s = ws.run_step(ws.run_step(s, "s1", "c1"), "s2", "c2")
    assert ws.open_steps(s) == []
    final = [st for st in s["steps"] if st["id"] == "s2"][0]["result"]
    assert final == {"name": "result", "value": pytest.approx(a * b / 2), "unit": "m*m"}


@pytest.mark.parametrize("plan,fragment,status", [
    ({"id": "s1", "entry": "formula:no_such_formula", "inputs": {}}, "not a registered formula", "failed"),
    ({"id": "s1", "entry": "arithmetic", "inputs": {}}, "formula:<name>", "failed"),
    ({"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1", "zz": 1}},
     "zz", "failed"),
    ({"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1", "b": "g2"}},
     "convert", "failed"),
    ({"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1"}},
     "second value", "waiting_input"),
])
def test_run_step_refuses_never_guesses(plan, fragment, status):
    s = _sheet_with([{"id": "g1", "value": 5, "unit": "m"}, {"id": "g2", "value": 2, "unit": "m2"}], [plan])
    s = ws.run_step(s, "s1", "c1")
    step = s["steps"][0]
    assert step["status"] == status and "result" not in step
    assert fragment in (step.get("error") or " ".join(step.get("needs") or []))
    assert s["done"] is False and ws.open_steps(s) == ["s1"]


@pytest.mark.parametrize("dims_mm", [(2000, 2500, 3000), (12000, 4000, 1500)])
def test_run_step_converts_a_written_unit_to_the_declared_one(dims_mm):
    # The volume formula declares its inputs in m; the sheet holds them in mm.
    # The registry's declared units convert them; the plan names the output.
    keys = ("length_m", "width_m", "depth_m")
    givens = [{"id": f"g{i + 1}", "value": v, "unit": "mm"} for i, v in enumerate(dims_mm)]
    s = _sheet_with(givens, [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                              "inputs": {k: f"g{i + 1}" for i, k in enumerate(keys)}}])
    s = ws.run_step(s, "s1", "c1")
    step = s["steps"][0]
    assert step["status"] == "done", step
    metres = [v / 1000 for v in dims_mm]
    assert [step["inputs"][k] for k in keys] == pytest.approx(metres)
    assert step["result"]["value"] == pytest.approx(metres[0] * metres[1] * metres[2])
    assert step["result"]["name"] == "bank_volume_m3"


def test_run_step_without_an_output_and_no_main_output_is_failed():
    keys = ("length_m", "width_m", "depth_m")
    givens = [{"id": f"g{i + 1}", "value": 2, "unit": "m"} for i in range(3)]
    s = _sheet_with(givens, [{"id": "s1", "entry": "formula:excavation_volume",
                              "inputs": {k: f"g{i + 1}" for i, k in enumerate(keys)}}])
    s = ws.run_step(s, "s1", "c1")
    assert s["steps"][0]["status"] == "failed" and "no output" in s["steps"][0]["error"]


def test_unknown_step_id_is_failed():
    s = ws.run_step(ws.new_sheet("t", "q"), "s9", "c1")
    assert s["steps"][0]["status"] == "failed" and s["done"] is False


def test_reference_to_a_failed_step_is_not_resolved():
    s = _sheet_with([], [{"id": "s1", "entry": "e", "inputs": {}},
                         {"id": "s2", "entry": "e", "inputs": {"x": "s1"}}])
    s = ws.record_result(s, "s1", "c1", {"error": "out of range"})
    vals, problems = ws.resolve_inputs(s, "s2")
    assert problems and "s1" in problems[0] and vals == {}


@pytest.mark.parametrize("payload", [{"status": "error", "result": 5}, {"status": "error", "result": 0.0}])
def test_error_status_with_a_number_is_still_failed(payload):
    s = _sheet_with([], [{"id": "s1", "entry": "e", "inputs": {}}])
    s = ws.record_result(s, "s1", "c1", payload)
    assert s["steps"][0]["status"] == "failed"
