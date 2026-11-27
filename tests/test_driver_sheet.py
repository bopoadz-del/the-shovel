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


@pytest.mark.parametrize("literal", [
    float("nan"),
    float("inf"),
    -float("inf"),
    {"value": float("inf"), "unit": "-"},
    [2.0, float("nan")],
    {"length": {"value": float("-inf"), "unit": "m"}},
])
def test_a_plan_literal_that_is_not_finite_is_refused_and_the_sheet_unchanged(literal):
    s = _sheet_with([{"id": "g1", "value": 2, "unit": "m"}])
    out, problems = ws.apply_update(s, {"plan": [
        {"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"a": "g1", "b": literal}}]})
    assert problems == ["plan step s1 input b holds a number that is not finite "
                        "(NaN or infinity); write the figure itself"]
    assert out is s and out["plan"] == []


@pytest.mark.parametrize("literal", [
    1.05, -3, 0.0, 10 ** 400, "inf", "2.5 m3",
    {"value": 1e308, "unit": "m"}, [1, {"value": 2, "unit": "m"}], True,
])
def test_a_finite_or_text_plan_literal_is_kept_as_written(literal):
    # the edge: the largest finite float, a huge int and the WORD "inf" are
    # not non-finite numbers; only a float that is NaN or infinite is refused
    s = _sheet_with([{"id": "g1", "value": 2, "unit": "m"}])
    out, problems = ws.apply_update(s, {"plan": [
        {"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"a": "g1", "b": literal}}]})
    assert problems == []
    assert out["plan"][0]["inputs"]["b"] == literal


@pytest.mark.parametrize("name,bad", [("factor", float("nan")), ("rate", {"value": float("inf"), "unit": "m"})])
def test_a_non_finite_literal_refuses_the_whole_update_beside_a_good_step(name, bad):
    s = _sheet_with([{"id": "g1", "value": 2, "unit": "m"}])
    out, problems = ws.apply_update(s, {"plan": [
        {"id": "s1", "entry": "e", "inputs": {"a": "g1", "b": 3}},
        {"id": "s2", "entry": "e", "inputs": {"a": "s1", name: bad}}]})
    assert problems == [f"plan step s2 input {name} holds a number that is not finite "
                        "(NaN or infinity); write the figure itself"]
    assert out is s


@pytest.mark.parametrize("step", ["s1", 3, ["s1", "e"], None])
def test_a_plan_step_that_is_not_an_object_is_refused_and_the_sheet_unchanged(step):
    s = _sheet_with([{"id": "g1", "value": 2, "unit": "m"}])
    out, problems = ws.apply_update(s, {"plan": [step]})
    assert problems == ["a plan step must be an object"]
    assert out is s and out["plan"] == []


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
    ("multiply", 6, 5, "-", "kg", 30.0, "kg"),
    ("multiply", 4, 2.5, "-", "m2", 10.0, "m2"),
    # a divisor that is itself a ratio or a product is bracketed: m2 over m2/day is not (m2/m2)/day
    ("divide", 150, 50, "m2", "m2/day", 3.0, "m2/(m2/day)"),
    ("divide", 1, 4, "-", "m/s", 0.25, "1/(m/s)"),
    ("divide", 90, 3, "kN", "m*m", 30.0, "kN/(m*m)"),
    ("divide", 2400, 2, "kg/m3", "m", 1200.0, "kg/m3/m"),      # a ratio divided on: read left to right, as written
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


@pytest.mark.parametrize("side", [2, 3.5])
def test_run_step_without_an_output_and_no_main_output_is_failed(side):
    keys = ("length_m", "width_m", "depth_m")
    givens = [{"id": f"g{i + 1}", "value": side, "unit": "m"} for i in range(3)]
    s = _sheet_with(givens, [{"id": "s1", "entry": "formula:excavation_volume",
                              "inputs": {k: f"g{i + 1}" for i, k in enumerate(keys)}}])
    s = ws.run_step(s, "s1", "c1")
    assert s["steps"][0]["status"] == "failed"
    assert "names none; set the step's output to one of: bank_volume_m3" in s["steps"][0]["error"]


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


# --- a carried given keeps where it came from ---------------------------------

@pytest.mark.parametrize("value,unit", [(20, "nr"), (3.5, "m")])
def test_a_carried_given_without_a_source_is_marked_earlier_turn(value, unit):
    previous = {"givens": [{"id": "g1", "value": value, "unit": unit}],
                "missing": [{"for": "s1", "input": "depth", "label": "depth", "asked": True}]}
    nxt = ws.new_sheet("t2", "next", previous=previous)
    assert nxt["givens"] == [{"id": "g1", "value": value, "unit": unit, "source": "earlier_turn"}]
    assert nxt["missing"][0]["asked"] is True


# --- a correction re-opens what depends on it -----------------------------------

def _chain(a, b, c):
    """g1 + g2 -> s1; s1 x g3 -> s2; g3 / g2 -> s3 (s3 does not use s1)."""
    s = _sheet_with([{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"},
                     {"id": "g3", "value": c, "unit": "-"}])
    s, problems = ws.apply_update(s, {"plan": [
        {"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1", "b": "g2"}},
        {"id": "s2", "entry": "formula:arithmetic_step", "inputs": {"operator": "multiply", "a": "s1", "b": "g3"}},
        {"id": "s3", "entry": "formula:arithmetic_step", "inputs": {"operator": "divide", "a": "g3", "b": "g2"}},
    ]})
    assert problems == []
    for sid in ("s1", "s2", "s3"):
        s = ws.run_step(s, sid, "c-" + sid)
    assert s["done"] is True
    return s


def _results(s):
    return {st["id"]: round(st.get("result", {}).get("value"), 6) for st in s["steps"]}


@pytest.mark.parametrize("a,b,c,new_a", [(100, 4, 2, 150), (7, 3, 5, 9)])
def test_a_corrected_given_reopens_every_step_that_depends_on_it(a, b, c, new_a):
    s = _chain(a, b, c)
    s, problems = ws.apply_update(s, {"givens": [{"id": "g1", "value": new_a, "unit": "m"}]})
    assert problems == []
    # s1 uses g1 directly, s2 through s1: both open, results cleared; s3 kept.
    assert _results(s) == {"s3": round(c / b, 6)}
    assert s["done"] is False and ws.open_steps(s) == ["s1", "s2"]
    for sid in ("s1", "s2"):
        s = ws.run_step(s, sid, "again-" + sid)
    assert _results(s) == {"s3": round(c / b, 6), "s1": new_a + b, "s2": (new_a + b) * c}
    assert s["done"] is True


@pytest.mark.parametrize("a,b,c", [(100, 4, 2), (7, 3, 5)])
def test_a_corrected_given_used_by_every_step_reopens_them_all(a, b, c):
    s = _chain(a, b, c)
    s, _ = ws.apply_update(s, {"givens": [{"id": "g2", "value": b + 1, "unit": "m"}]})
    assert s["steps"] == [] and s["done"] is False


@pytest.mark.parametrize("a,b,c", [(100, 4, 2), (7, 3, 5)])
def test_a_replanned_step_reopens_itself_and_its_dependents_only(a, b, c):
    s = _chain(a, b, c)
    s, problems = ws.apply_update(s, {"plan": [
        {"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "multiply", "a": "g1", "b": "g2"}}]})
    assert problems == []
    assert _results(s) == {"s3": round(c / b, 6)} and s["done"] is False
    s = ws.run_step(s, "s1", "x")
    assert _results(s)["s1"] == a * b


@pytest.mark.parametrize("a,b,c", [(100, 4, 2), (7, 3, 5)])
def test_the_same_given_or_step_written_again_reopens_nothing(a, b, c):
    s = _chain(a, b, c)
    before = _results(s)
    s, _ = ws.apply_update(s, {"givens": [{"id": "g1", "value": a, "unit": "m", "what": "a new label"}],
                               "plan": [{"id": "s3", "entry": "formula:arithmetic_step",
                                         "inputs": {"operator": "divide", "a": "g3", "b": "g2"}}]})
    assert _results(s) == before and s["done"] is True
    # A new given that no step uses re-opens nothing either.
    s, _ = ws.apply_update(s, {"givens": [{"id": "g9", "value": a + c, "unit": "m"}]})
    assert _results(s) == before and s["done"] is True


@pytest.mark.parametrize("unit", ["mm", "ft"])
def test_a_corrected_unit_alone_reopens_the_dependents(unit):
    s = _chain(100, 4, 2)
    s, _ = ws.apply_update(s, {"givens": [{"id": "g3", "value": 2, "unit": unit}]})
    assert sorted(_results(s)) == ["s1"]  # s2 and s3 use g3


@pytest.mark.parametrize("ids,expected", [
    ({"g1"}, {"s1", "s2"}),
    ({"g3"}, {"s2", "s3"}),
    ({"s1"}, {"s2"}),
    ({"s2"}, set()),
    (set(), set()),
])
def test_dependents_follow_the_plans_own_inputs(ids, expected):
    assert ws.dependents(_chain(100, 4, 2), ids) == expected


def _replan_s1(s, **change):
    step = {"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1", "b": "g2"}}
    s, problems = ws.apply_update(s, {"plan": [{**step, **change}]})
    assert problems == []
    return s


@pytest.mark.parametrize("output,a,b,c", [("result", 100, 4, 2), ("result_unit", 7, 3, 5)])
def test_a_replanned_step_changing_only_its_output_reopens_it_and_its_dependents(output, a, b, c):
    s = _replan_s1(_chain(a, b, c), output=output)
    assert _results(s) == {"s3": round(c / b, 6)} and s["done"] is False
    assert ws.open_steps(s) == ["s1", "s2"]


@pytest.mark.parametrize("entry,a,b,c", [("formula:another_entry", 100, 4, 2), ("formula:third_entry", 7, 3, 5)])
def test_a_replanned_step_changing_only_its_entry_reopens_it_and_its_dependents(entry, a, b, c):
    s = _replan_s1(_chain(a, b, c), entry=entry)
    assert _results(s) == {"s3": round(c / b, 6)} and s["done"] is False
    assert ws.open_steps(s) == ["s1", "s2"]


# --- a step that ran closes what it was missing --------------------------------

def _waiting_for(missing, a, b):
    s = ws.new_sheet("t-2", "q")
    s, problems = ws.apply_update(s, {"missing": missing})
    assert problems == []
    return _replan_after_finding(s, a, b)


def _replan_after_finding(s, a, b):
    """The model finds the value and plans s1, without rewriting missing."""
    s, problems = ws.apply_update(s, {
        "givens": [{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"}],
        "plan": [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]})
    assert problems == []
    return s


@pytest.mark.parametrize("a,b", [(55, 77), (12.5, 3)])
def test_a_step_that_runs_closes_the_missing_input_it_was_for(a, b):
    s = _waiting_for([{"for": "s1", "input": "b"}, {"for": "s2", "input": "rate"}], a, b)
    assert len(ws.unasked(s)) == 2  # planning alone closes nothing
    s = ws.run_step(s, "s1", "c1")
    assert s["steps"][0]["result"]["value"] == a + b and s["done"] is True
    # s1 has its result: its item is closed; s2 (no result) stays as the
    # model wrote it. (An item naming no step is refused at the update.)
    assert [(m["for"], m["input"]) for m in s["missing"]] == [("s2", "rate")]


@pytest.mark.parametrize("a,b", [(55, 77), (12.5, 3)])
def test_a_step_that_fails_or_waits_closes_nothing(a, b):
    s = _waiting_for([{"for": "s1", "input": "b"}], a, b)
    failed = ws.run_step(s, "s9", "c1")  # not a plan step: failed
    assert [m["input"] for m in failed["missing"]] == ["b"]
    s, _ = ws.apply_update(s, {"plan": [{"id": "s1", "entry": "formula:no_such_entry", "inputs": {"a": "g1"}}]})
    s = ws.run_step(s, "s1", "c2")
    assert s["steps"][0]["status"] == "failed" and [m["input"] for m in s["missing"]] == ["b"]


@pytest.mark.parametrize("value,unit", [(552.75, "m"), (48, "m*m")])
def test_close_answered_reads_only_done_steps_with_a_result(value, unit):
    s = ws.new_sheet("t-3", "q")
    s["missing"] = [{"for": "s1", "input": "x"}, {"for": "s2", "input": "y"}, {"for": "s3", "input": "z"}]
    s["steps"] = [{"id": "s1", "status": "done", "result": {"name": "result", "value": value, "unit": unit}},
                  {"id": "s2", "status": "done"},  # done but no result: not closed
                  {"id": "s3", "status": "waiting_input"}]
    assert [m["for"] for m in ws.close_answered(s)["missing"]] == ["s2", "s3"]
    assert len(s["missing"]) == 3  # the sheet passed in is not changed
    assert ws.close_answered(ws.new_sheet("t-4", "q"))["missing"] == []


@pytest.mark.parametrize("a,b,op,expected,unit", [(55, 2, "add", 57, "m"), (12, 4, "multiply", 48, "m*m")])
def test_code_results_lists_every_result_code_wrote(a, b, op, expected, unit):
    s = _sheet_with([{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"}],
                    [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": op, "a": "g1", "b": "g2"}},
                     {"id": "s2", "entry": "formula:no_such_entry", "inputs": {"a": "g1"}}])
    s = ws.run_step(ws.run_step(s, "s1", "c1"), "s2", "c2")
    assert ws.code_results(s) == [{"step": "s1", "entry": "formula:arithmetic_step", "name": "result",
                                   "value": expected, "unit": unit, "rests_on": [float(a), float(b)]}]


@pytest.mark.parametrize("sheet", [
    None, "a sheet", [], {}, {"steps": None}, {"steps": ["s1", 3]},
    {"steps": [{"id": "s1", "status": "failed", "result": {"value": 5, "unit": "m"}}]},
    {"steps": [{"id": "s1", "status": "done", "result": "5 m"}]},
    {"steps": [{"id": "s1", "status": "done", "result": {"value": float("nan"), "unit": "m"}}]},
    {"steps": [{"id": "s1", "status": "done", "result": {"value": True, "unit": "m"}}]},
    {"steps": [{"id": "s1", "status": "done", "result": {"value": None, "unit": "m"}}]},
])
def test_code_results_of_anything_without_a_done_numeric_result_is_empty(sheet):
    assert ws.code_results(sheet) == []


@pytest.mark.parametrize("a,b", [(55, 77), (12.5, 3)])
def test_a_missing_input_written_for_a_step_that_already_ran_is_closed(a, b):
    s = ws.run_step(_replan_after_finding(ws.new_sheet("t-5", "q"), a, b), "s1", "c1")
    s, problems = ws.apply_update(s, {"missing": [{"for": "s1", "input": "b"}, {"for": "s2", "input": "rate"}]})
    assert problems == [] and [m["for"] for m in s["missing"]] == ["s2"]
    # A correction that reopens s1 lets its input be missing again.
    s, _ = ws.apply_update(s, {"givens": [{"id": "g2", "value": b + 1, "unit": "m"}],
                               "missing": [{"for": "s1", "input": "b"}]})
    assert [m["for"] for m in s["missing"]] == ["s1"] and s["done"] is False


@pytest.mark.parametrize("item", [{"input": "b"}, {"input": "rate", "for": ""}, {"input": "rate", "for": "g1"}])
def test_a_missing_item_must_name_its_plan_step(item):
    start = ws.new_sheet("t1", "q")
    out, problems = ws.apply_update(start, {"missing": [item]})
    assert out == start
    assert problems and "must say which plan step it is for" in problems[0]


@pytest.mark.parametrize("a,b", [(20, 3), (7.5, 40)])
def test_a_result_lists_every_number_it_rests_on_through_earlier_steps(a, b):
    work, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "givens": [{"id": "g1", "value": a, "unit": "m"}],
        "plan": [{"id": "s1", "entry": "formula:arithmetic_step",
                  "inputs": {"operator": "multiply", "a": "g1", "b": b, "flag": True}},
                 {"id": "s2", "entry": "formula:arithmetic_step",
                  "inputs": {"operator": "add", "a": "s1", "b": "g1"}},
                 {"id": "s3", "entry": "formula:arithmetic_step",
                  "inputs": {"operator": "add", "a": "s1", "b": "s2"}}]})
    assert problems == []
    assert sorted(ws.rests_on(work, "s1")) == sorted([float(a), float(b)])
    assert sorted(ws.rests_on(work, "s2")) == sorted([float(a), float(b), float(a)])
    # s3 reads s1 directly and again through s2: s1 is followed once.
    assert sorted(ws.rests_on(work, "s3")) == sorted([float(a), float(b), float(a)])
    assert ws.rests_on(work, "s9") == []


@pytest.mark.parametrize("output,result,expected", [
    ("", {"bank_volume_m3": 120.0, "loose_volume_m3": 150.0, "note": "text"},
     "f has several outputs and the plan step names none; set the step's output to one of: "
     "bank_volume_m3, loose_volume_m3"),
    ("area", {"volume": 3, "length": 2.5},
     "f has no output 'area'; set the step's output to one of: length, volume"),
    ("area", {"note": "text"}, "f has no output 'area'"),
    ("", None, "f has several outputs and the plan step names none"),
])
def test_a_step_that_kept_no_output_is_told_which_outputs_exist(output, result, expected):
    assert ws._no_output_error("f", output, result) == expected


@pytest.mark.parametrize("a,b", [(40, 2), (7.5, 67)])
def test_code_marks_a_given_checked_and_only_an_unchanged_one_keeps_the_mark(a, b):
    s, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "givens": [{"id": "g1", "value": a, "unit": "m", "checked": True},   # the model cannot set the mark
                   {"id": "g2", "value": b, "unit": "m"}],
        "plan": [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1", "b": "g2"}}]})
    assert problems == [] and not any(g.get("checked") for g in s["givens"])
    marked = ws.mark_given(s, lambda number, unit: number == float(a))
    assert [g.get("checked") for g in marked["givens"]] == [True, None]
    assert not any(g.get("checked") for g in s["givens"])  # the sheet passed in is not changed
    assert ws.rests_on(marked, "s1") == [float(b)]         # the checked given is no longer open
    # The same value and unit written again keeps code's mark; a new value loses it.
    same, _ = ws.apply_update(marked, {"givens": [{"id": "g1", "value": a, "unit": "m"}]})
    assert same["givens"][0].get("checked") is True
    changed, _ = ws.apply_update(marked, {"givens": [{"id": "g1", "value": a + 1, "unit": "m"}]})
    assert changed["givens"][0].get("checked") is None
    assert sorted(ws.rests_on(changed, "s1")) == sorted([float(a + 1), float(b)])
    # The next turn's sheet carries the mark with the given.
    carried = ws.new_sheet("t2", "q2", marked)
    assert [g.get("checked") for g in carried["givens"]] == [True, None]


@pytest.mark.parametrize("by,kept", [("user", [40.0]), ("passage", []), (True, [])])
def test_user_givens_are_the_givens_code_marked_as_the_users_own(by, kept):
    s, _ = ws.apply_update(ws.new_sheet("t1", "q"), {"givens": [{"id": "g1", "value": 40, "unit": "m"},
                                                                {"id": "g2", "value": 3, "unit": "m"}]})
    marked = ws.mark_given(s, lambda number, unit: by if number == 40.0 else "")
    assert marked["givens"][0].get("checked") == by and marked["givens"][1].get("checked") is None
    assert ws.user_givens(marked) == kept
    assert ws.user_givens(None) == [] and ws.user_givens({"givens": ["g1"]}) == []


@pytest.mark.parametrize("literal,expected", [
    (43, [43.0]), (12.5, [12.5]), ("43 m", [43.0]), ("12.5m", [12.5]), ("-7", [-7.0]), (" -7.5 m", [-7.5]),
    ("1,250 m3", [1250.0]), ({"value": 4321, "unit": "m3"}, [4321.0]), ({"value": "987.5 m2"}, [987.5]),
    ([3, "4 m", {"value": 5}], [3.0, 4.0, 5.0]),
    (".5 m", [0.5]), ("+.75 m", [0.75]), (".5", [0.5]), ("-.25 m", [-0.25]), ("5e3 m", [5000.0]),
    ("AED 5,000", [5000.0]), ("20 ft", [20.0]), ("1,250.", [1250.0]),
    ("add", []), ("c30", []), ("m 43", []), (True, []), (None, []), ("", []), ({"unit": "m"}, []),
])
def test_written_numbers_reads_a_number_in_every_shape_a_formula_would(literal, expected):
    assert ws.written_numbers(literal) == expected


@pytest.mark.parametrize("depth,ran_on", [(".5 m", 0.5), ("+.75 m", 0.75), ("2 m", 2.0)])
def test_a_literal_the_reader_finds_no_number_in_is_taken_as_the_formula_ran_it(monkeypatch, depth, ran_on):
    """Whatever shape the reader of written numbers misses, the check reads
    no less than the step did: the number the formula ran on for that input
    stands in, so the result still rests on it."""
    work, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "givens": [{"id": "g1", "value": 40, "unit": "m"}],
        "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                  "inputs": {"length_m": "g1", "width_m": "1.5 m", "depth_m": depth}}]})
    assert problems == []
    ran = ws.run_step(work, "s1")
    assert [st["status"] for st in ran["steps"]] == ["done"]
    assert sorted(ws.rests_on(ran, "s1")) == sorted([40.0, 1.5, ran_on])
    monkeypatch.setattr(ws, "written_numbers", lambda literal: [])      # a shape the reader cannot read
    assert sorted(ws.rests_on(ran, "s1")) == sorted([40.0, 1.5, ran_on])
    assert sorted(ws.rests_on(work, "s1")) == [40.0]                     # not run: nothing to fall back on


@pytest.mark.parametrize("value,numbers", [(5.0, [5.0]), ([1, "x", 2.5], [1.0, 2.5]), ((3, 4), [3.0, 4.0]),
                                           ("add", []), (True, []), (None, []), ([], [])])
def test_ran_numbers_are_the_numbers_a_step_received_for_an_input(value, numbers):
    assert ws._ran_numbers(value) == numbers


@pytest.mark.parametrize("a,b", [("43 m", {"value": 10, "unit": "m"}), ("12.5 m", {"value": "8 m"})])
def test_a_result_rests_on_numbers_written_in_the_plan_as_text_or_as_objects(a, b):
    work, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "plan": [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": a, "b": b}}]})
    assert problems == []
    assert sorted(ws.rests_on(work, "s1")) == sorted(ws.written_numbers(a) + ws.written_numbers(b))
    assert len(ws.rests_on(work, "s1")) == 2


# --- the plan's own quantities keep the unit they were written with ---

@pytest.mark.parametrize("literal,expected", [
    (43, [(43.0, "")]), ("43 m", [(43.0, "m")]), ("12 mm", [(12.0, "mm")]), ("12.5m", [(12.5, "m")]),
    (" -7.5 m", [(-7.5, "m")]), ("-7", [(-7.0, "")]), ("1,250 m3", [(1250.0, "m3")]),
    ({"value": 4321, "unit": "m3"}, [(4321.0, "m3")]), ({"value": 12, "unit": "mm"}, [(12.0, "mm")]),
    ({"value": "987.5 m2"}, [(987.5, "m2")]), ({"value": 5}, [(5.0, "")]),
    ([3, "4 m", {"value": 5, "unit": "kg"}], [(3.0, ""), (4.0, "m"), (5.0, "kg")]),
    ("add", []), ("m 43", []), (True, []), (None, []), ({"unit": "m"}, []),
])
def test_written_quantities_reads_each_number_with_the_unit_written_on_it(literal, expected):
    assert ws.written_quantities(literal) == expected
    assert ws.written_numbers(literal) == [number for number, _unit in expected]


@pytest.mark.parametrize("unit_a,literal,expected", [
    # The bare 2 is fed to an input the formula declares in m: it ran as 2 m, and is listed so.
    ("m", "3 m", [(12.0, "m"), (3.0, "m"), (2.0, "m")]),
    ("mm", {"value": 3, "unit": "cm"}, [(12.0, "mm"), (3.0, "cm"), (2.0, "m")]),
])
def test_what_a_step_rests_on_keeps_each_unit_the_model_wrote(unit_a, literal, expected):
    work, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "givens": [{"id": "g1", "value": 12, "unit": unit_a}],
        "plan": [{"id": "s1", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
                  "inputs": {"length_m": "g1", "width_m": literal, "depth_m": 2}}]})
    assert problems == []
    assert sorted(ws.rests_quantities(work, "s1")) == sorted(expected)
    assert sorted(ws.rests_on(work, "s1")) == sorted(number for number, _unit in expected)


@pytest.mark.parametrize("unit", ["m", "mm", "-"])     # "-" is how a pure number is written on the sheet
def test_the_check_a_given_is_marked_with_receives_its_unit(unit):
    s, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "givens": [{"id": "g1", "value": 12, "unit": unit}]})
    assert problems == []
    asked = []

    def check(number, written_unit):
        asked.append((number, written_unit))
        return "user" if written_unit == "m" else ""

    marked = ws.mark_given(s, check)
    assert asked == [(12.0, unit)]
    assert [g.get("checked") for g in marked["givens"]] == (["user"] if unit == "m" else [None])


# --- a value read from the sheet keeps its own unit ----------------------------

@pytest.mark.parametrize("given_unit,beside,unit_first", [
    ("mm", "m", False), ("mm", "m", True),      # the plan writes the unit after / before the value
    ("kg", "t", False), ("kg", "t", True),
    ("-", "m", False), ("-", "m", True),         # a pure number is not given a unit by the plan
    ("m", "m", False),
])
def test_formula_values_keep_the_unit_of_a_value_read_from_the_sheet(given_unit, beside, unit_first):
    from app.lib import formula_registry

    pair = [("b", {"value": 250, "unit": given_unit}), ("b_unit", beside)]
    values = dict([("operator", "add"), ("a", {"value": -43, "unit": "m"})] + (pair[::-1] if unit_first else pair))
    out = ws._formula_values(formula_registry.get("arithmetic_step"), values)
    assert out == {"operator": "add", "a": -43, "a_unit": "m", "b": 250, "b_unit": given_unit}


@pytest.mark.parametrize("number,beside", [(10, "m"), (250, "mm"), (2.5, "kg")])
def test_formula_values_take_the_plans_unit_for_a_number_the_plan_wrote_bare(number, beside):
    from app.lib import formula_registry

    values = {"operator": "add", "a": {"value": -43, "unit": "m"}, "b": number, "b_unit": beside}
    out = ws._formula_values(formula_registry.get("arithmetic_step"), values)
    assert out == {"operator": "add", "a": -43, "a_unit": "m", "b": number, "b_unit": beside}


@pytest.mark.parametrize("level,level_unit,rise,rise_unit,unit_first", [
    (-43, "m", 250, "mm", False), (-43, "m", 250, "mm", True),
    (12, "m", 500, "mm", False), (12, "m", 500, "mm", True),
    (900, "kg", 2, "t", False), (900, "kg", 2, "t", True),
])
def test_run_step_never_runs_a_given_in_a_unit_the_plan_wrote_beside_it(level, level_unit, rise, rise_unit,
                                                                        unit_first):
    """The plan says the second value is in the first one's unit; the given
    says otherwise. The given's unit stands, so the two are not added."""
    inputs = {"operator": "add", "a": "g1"}
    inputs.update([("b_unit", level_unit), ("b", "g2")] if unit_first else [("b", "g2"), ("b_unit", level_unit)])
    s = _sheet_with([{"id": "g1", "value": level, "unit": level_unit},
                     {"id": "g2", "value": rise, "unit": rise_unit}],
                    [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": inputs}])
    step = ws.run_step(s, "s1", "c1")["steps"][0]
    assert step["status"] == "failed" and "result" not in step
    assert step["error"] == f"cannot add {level_unit} and {rise_unit}: convert one to the other's unit first"


@pytest.mark.parametrize("first,second,beside,expected", [
    (12.0, 4.0, "m", 16.0), (3.5, 8.0, "kg", 11.5),
])
def test_run_step_keeps_an_earlier_steps_unit_over_one_written_beside_it(first, second, beside, expected):
    """s1's result is in m2 (code wrote it). s2 reads it and the plan writes
    another unit beside it: the step's own unit stands."""
    s = _sheet_with([{"id": "g1", "value": first, "unit": "m2"}, {"id": "g2", "value": second, "unit": "m2"}],
                    [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}},
                     {"id": "s2", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "multiply", "a": "s1", "a_unit": beside, "b": 2}}])
    s = ws.run_step(ws.run_step(s, "s1", "c1"), "s2", "c2")
    final = [st for st in s["steps"] if st["id"] == "s2"][0]
    assert final["result"] == {"name": "result", "value": pytest.approx(expected * 2), "unit": "m2"}
    assert final["inputs"]["a_unit"] == "m2"


@pytest.mark.parametrize("level,rise,beside,total", [(-43, 10, "m", -33.0), (12, 0.5, "m", 12.5), (7, 3, "kg", 10.0)])
def test_run_step_runs_a_bare_number_of_the_plan_in_the_unit_written_beside_it(level, rise, beside, total):
    s = _sheet_with([{"id": "g1", "value": level, "unit": beside}],
                    [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": rise, "b_unit": beside}}])
    step = ws.run_step(s, "s1", "c1")["steps"][0]
    assert step["status"] == "done"
    assert step["result"] == {"name": "result", "value": pytest.approx(total), "unit": beside}


# --- the plan's bare number is checked in the unit the plan runs it in ----------

@pytest.mark.parametrize("inputs,name,expected", [
    ({"b": 250, "b_unit": "m"}, "b", "m"),
    ({"b": 250, "b_unit": " mm "}, "b", "mm"),
    ({"b": 250, "b_unit": "-"}, "b", ""),          # "-" is a pure number: no unit
    ({"b": 250}, "b", ""),
    ({"b": 250, "b_unit": 7}, "b", ""),            # not text: no unit
    ({"a": 1, "a_unit": "kg", "b": 250}, "b", ""),  # another input's unit is not this one's
])
def test_unit_beside_reads_the_unit_a_plan_writes_next_to_an_input(inputs, name, expected):
    assert ws._unit_beside(inputs, name) == expected


@pytest.mark.parametrize("b,beside,expected", [
    (250, "m", [(250.0, "m")]),
    ("250", "mm", [(250.0, "mm")]),
    (250, "-", [(250.0, "")]),
    (250, None, [(250.0, "")]),
    ("250 mm", "m", [(250.0, "mm")]),              # a unit written on the number itself stands
    ({"value": 250, "unit": "mm"}, "m", [(250.0, "mm")]),
])
def test_a_bare_number_of_the_plan_rests_on_the_unit_written_beside_it(b, beside, expected):
    inputs = {"operator": "add", "a": "g1", "b": b}
    if beside is not None:
        inputs["b_unit"] = beside
    work, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "givens": [{"id": "g1", "value": -43, "unit": "m"}],
        "plan": [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": inputs}]})
    assert problems == []
    work = ws.mark_given(work, lambda number, unit: "user")
    assert ws.rests_quantities(work, "s1") == expected
    assert ws.rests_on(work, "s1") == [number for number, _unit in expected]


@pytest.mark.parametrize("a,b,operator,unit", [
    ("m2", "m2/day", "divide", "m2/(m2/day)"), ("-", "m/s", "divide", "1/(m/s)"), ("kN", "m*m", "divide", "kN/(m*m)"),
    ("currency", "m2", "divide", "currency/m2"), ("kg/m3", "m", "divide", "kg/m3/m"), ("m3", "M3", "divide", "-"),
    ("m", "-", "divide", "m"), ("-", "d", "divide", "1/d"),
    ("m", "m", "multiply", "m*m"), ("-", "kg", "multiply", "kg"), ("m3", "-", "multiply", "m3"),
    ("m3", "currency/m3", "multiply", "m3*currency/m3"),
])
def test_compose_writes_the_unit_of_a_product_or_a_quotient_as_it_is_read(a, b, operator, unit):
    from app.agents.base.formulas.arithmetic import _compose

    assert _compose(a, b, operator) == unit


# --- a number with no unit fed to an input that declares a unit is that quantity ---

@pytest.mark.parametrize("unit,expected", [
    ("-", True), ("", True), (None, True), (" - ", True), ("  ", True),
    ("m", False), ("m3", False), ("nr", False), ("currency", False), ("--", False),
])
def test_no_unit_is_true_only_for_a_pure_number_as_the_sheet_writes_it(unit, expected):
    assert ws.no_unit(unit) is expected


@pytest.mark.parametrize("entry,name,expected", [
    ("formula:excavation_volume", "length_m", "m"),
    ("formula:excavation_volume", "bulking_factor", "fraction"),
    ("formula:unit_cost_total", "unit_rate", "currency"),
    ("formula:beam_moment_simple", "udl_w_kn_m", "kN/m"),
    ("formula:unit_cost_total", "quantity", ""),            # declared "nr": a count carries no dimension
    ("formula:arithmetic_step", "a", ""),                   # declared a bare number (its unit is beside it)
    ("formula:arithmetic_step", "operator", ""),            # a text option: "-"
    ("formula:excavation_volume", "no_such_input", ""),
    ("formula:no_such_formula", "length_m", ""),
    ("workflow:excavation_volume", "length_m", ""),         # not a formula entry
    ("excavation_volume", "length_m", ""), (None, "length_m", ""), ("", "", ""),
])
def test_declared_unit_reads_the_unit_the_registry_declares_for_an_input(entry, name, expected):
    assert ws.declared_unit(entry, name) == expected


@pytest.mark.parametrize("entry,output,expected", [
    ("formula:arithmetic_step", "result", True),            # it returns result_unit beside result
    ("formula:excavation_volume", "bank_volume_m3", False),
    ("formula:critical_path_float", "total_float", False),  # declared "-": no unit is declared, none returned
    ("formula:arithmetic_step", "", False), ("formula:arithmetic_step", "result_unit", False),
    ("formula:no_such_formula", "result", False), ("workflow:arithmetic_step", "result", False),
    (None, "result", False),
])
def test_returns_unit_is_true_only_for_a_formula_that_returns_its_results_unit(entry, output, expected):
    assert ws.returns_unit(entry, output) is expected


def _trench_plan(length="g1", width="g2", depth="g3", sid="s1"):
    return {"id": sid, "entry": "formula:excavation_volume", "output": "bank_volume_m3",
            "inputs": {"length_m": length, "width_m": width, "depth_m": depth}}


@pytest.mark.parametrize("units,plan,expected", [
    (["-", "m", "m"], [_trench_plan()], {"g1": ["m"]}),
    (["-", "-", "-"], [_trench_plan()], {"g1": ["m"], "g2": ["m"], "g3": ["m"]}),
    (["-", "m", "m"], [_trench_plan(width="g1", depth="g1")], {"g1": ["m"]}),            # each unit once
    (["-", "m", "m"], [_trench_plan(), {"id": "s2", "entry": "formula:excavation_volume",
                                        "inputs": {"length_m": "g2", "width_m": "g2", "depth_m": "g3",
                                                   "bulking_factor": "g1"}}], {"g1": ["m", "fraction"]}),
    (["m", "m", "m"], [_trench_plan()], {}),                                             # every given has its unit
    (["-", "-", "-"], [{"id": "s1", "entry": "formula:arithmetic_step",
                        "inputs": {"operator": "add", "a": "g1", "b": "g2"}}], {}),    # inputs that declare none
    (["-", "-", "-"], [{"id": "s1", "entry": "formula:no_such_formula", "inputs": {"length_m": "g1"}}], {}),
    (["-", "-", "-"], [], {}),
])
def test_fed_units_lists_the_declared_units_a_given_with_no_unit_is_fed_to(units, plan, expected):
    s = _sheet_with([{"id": f"g{i + 1}", "value": 10 + i, "unit": unit} for i, unit in enumerate(units)], plan)
    assert ws.fed_units(s) == expected


@pytest.mark.parametrize("sheet", [{}, {"givens": None, "plan": None}, {"givens": ["g1"], "plan": ["s1", {"id": "s1"}]},
                                   {"givens": [{"id": "g1", "unit": "-"}],
                                    "plan": [{"id": "s1", "entry": "formula:excavation_volume", "inputs": "g1"}]}])
def test_fed_units_of_a_sheet_with_nothing_fed_is_empty(sheet):
    assert ws.fed_units(sheet) == {}


@pytest.mark.parametrize("number,given_as,marks,asked_units", [
    (12, {"-", "m"}, ("user", ["m"]), ["-", "m"]),        # the user wrote 12 m (or 12 bare)
    (12, {"-"}, ("user", None), ["-", "m"]),              # the number is the user's; 12 m is not (it was 12 mm)
    (12, set(), (None, None), ["-", "m"]),                # nobody gave the number at all
    (250, {"-", "m"}, ("user", ["m"]), ["-", "m"]),
])
def test_mark_given_checks_a_given_with_no_unit_as_the_quantity_it_is_fed_as(number, given_as, marks, asked_units):
    s = _sheet_with([{"id": "g1", "value": number, "unit": "-"}, {"id": "g2", "value": 3, "unit": "m"},
                     {"id": "g3", "value": 2, "unit": "m"}], [_trench_plan()])
    asked = []

    def check(value, unit):
        asked.append((value, unit))
        return "user" if value != float(number) or unit in given_as else ""

    marked = ws.mark_given(s, check)
    assert (marked["givens"][0].get("checked"), marked["givens"][0].get("checked_as")) == marks
    assert [unit for value, unit in asked if value == float(number)] == asked_units
    # a given with its own unit is checked in that unit only, and gets no second mark
    assert [(g.get("checked"), g.get("checked_as")) for g in marked["givens"][1:]] == [("user", None), ("user", None)]
    assert "checked_as" not in s["givens"][0]             # the sheet passed in is not changed
    # what the step still rests on: the number when nobody gave it, and the quantity it ran as
    open_ = ([] if marks[0] else [(float(number), "-")]) + ([] if marks[1] else [(float(number), "m")])
    assert ws.rests_quantities(marked, "s1") == open_


@pytest.mark.parametrize("number", [12, 250])
def test_the_quantity_mark_is_kept_asked_once_and_lost_with_the_given(number):
    """Mark, keep, re-plan, change, mark again: the mark stays with an
    unchanged given into the next turn, a unit already held is not asked
    again, a new declared unit is asked for by itself, and a changed value
    or unit loses every mark."""
    s = _sheet_with([{"id": "g1", "value": number, "unit": "-"}, {"id": "g2", "value": 3, "unit": "m"},
                     {"id": "g3", "value": 2, "unit": "m"}], [_trench_plan()])
    asked = []

    def check(value, unit):
        asked.append((value, unit))
        return "user"

    marked = ws.mark_given(s, check)
    assert marked["givens"][0]["checked_as"] == ["m"]
    asked.clear()
    again = ws.mark_given(marked, check)
    assert asked == [] and again == marked                                    # nothing new to check
    same, _ = ws.apply_update(marked, {"givens": [{"id": "g1", "value": number, "unit": "-"}]})
    assert (same["givens"][0]["checked"], same["givens"][0]["checked_as"]) == ("user", ["m"])
    carried = ws.new_sheet("t2", "q2", marked)
    assert carried["givens"][0]["checked_as"] == ["m"]
    # the model cannot write the mark
    typed, _ = ws.apply_update(s, {"givens": [{"id": "g1", "value": number, "unit": "-", "checked_as": ["m"]}]})
    assert "checked_as" not in typed["givens"][0]
    # the plan now feeds the same given to an input declared in another unit: only that one is asked
    replanned, problems = ws.apply_update(marked, {"plan": [
        {"id": "s2", "entry": "formula:excavation_volume", "output": "bank_volume_m3",
         "inputs": {"length_m": "g2", "width_m": "g2", "depth_m": "g3", "bulking_factor": "g1"}}]})
    assert problems == []
    assert ws.rests_quantities(replanned, "s2") == [(float(number), "fraction")]
    refused = ws.mark_given(replanned, lambda value, unit: "")
    assert refused["givens"][0]["checked_as"] == ["m"]
    assert ws.rests_quantities(refused, "s2") == [(float(number), "fraction")]
    assert ws.mark_given(replanned, check)["givens"][0]["checked_as"] == ["m", "fraction"]
    assert asked == [(float(number), "fraction")]
    # a new value, or a unit written on it, is a new given: no mark is left
    for change in ({"value": number + 1, "unit": "-"}, {"value": number, "unit": "mm"}):
        changed, _ = ws.apply_update(marked, {"givens": [{"id": "g1", **change}]})
        assert "checked" not in changed["givens"][0] and "checked_as" not in changed["givens"][0]
    back, _ = ws.apply_update(changed, {"givens": [{"id": "g1", "value": number, "unit": "-"}]})
    assert "checked_as" not in back["givens"][0]                             # written again: checked again
    assert ws.rests_quantities(back, "s1") == [(float(number), "-"), (float(number), "m")]


@pytest.mark.parametrize("length,expected", [
    (12, [(12.0, "m")]), ("12", [(12.0, "m")]), ({"value": 12}, [(12.0, "m")]),
    ("12 mm", [(12.0, "mm")]), ({"value": 12, "unit": "km"}, [(12.0, "km")]),      # a unit written on it stands
    ([12, "5 mm"], [(12.0, "m"), (5.0, "mm")]),
])
def test_a_bare_number_of_the_plan_rests_on_the_unit_its_input_declares(length, expected):
    work, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "plan": [_trench_plan(length=length, width="3 m", depth="2 m")]})
    assert problems == []
    assert ws.rests_quantities(work, "s1") == expected + [(3.0, "m"), (2.0, "m")]


@pytest.mark.parametrize("a,b", [(43, 250), (12, 0.5)])
def test_a_bare_number_fed_to_an_input_that_declares_no_unit_keeps_the_number_check_alone(a, b):
    """arithmetic_step's values declare no unit (the unit is written beside
    them), and a count carries no dimension: a bare number there is listed
    with no unit, as before."""
    work, problems = ws.apply_update(ws.new_sheet("t1", "q"), {
        "givens": [{"id": "g1", "value": a, "unit": "-"}],
        "plan": [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": {"operator": "add", "a": "g1", "b": b}},
                 {"id": "s2", "entry": "formula:unit_cost_total", "inputs": {"quantity": "g1", "unit_rate": "s1"}}]})
    assert problems == []
    assert ws.rests_quantities(work, "s1") == [(float(a), "-"), (float(b), "")]
    assert ws.fed_units(work) == {}
    marked = ws.mark_given(work, lambda number, unit: "user")
    assert "checked_as" not in marked["givens"][0]
    assert ws.rests_quantities(marked, "s1") == [(float(b), "")]


@pytest.mark.parametrize("first,second,unit,extra", [
    (43, 250, "-", [(293.0, "m")]),      # a pure-number sum fed to an input in m: 293 m, which nobody gave
    (12, 500, "-", [(512.0, "m")]),
    (43, 7, "m", []),                    # the sum carries its unit: nothing more to check
])
def test_a_pure_number_result_fed_to_an_input_that_declares_a_unit_is_listed_as_that_quantity(first, second, unit,
                                                                                            extra):
    s = _sheet_with([{"id": "g1", "value": first, "unit": unit}, {"id": "g2", "value": second, "unit": unit}],
                    [{"id": "s1", "entry": "formula:arithmetic_step",
                      "inputs": {"operator": "add", "a": "g1", "b": "g2"}},
                     _trench_plan(length="s1", width="3 m", depth="2 m", sid="s2")])
    s = ws.mark_given(s, lambda number, unit: "user")                        # each number was given
    assert ws.rests_quantities(s, "s2") == [(3.0, "m"), (2.0, "m")]          # s1 has not run: no result yet
    s = ws.run_step(s, "s1", "c1")
    assert ws.rests_quantities(s, "s2") == extra + [(3.0, "m"), (2.0, "m")]
    assert ws.rests_quantities(s, "s1") == []


@pytest.mark.parametrize("output,value", [("bulking_factor", 0.25), ("bank_volume_m3", 72.0)])
def test_a_result_whose_formula_does_not_return_its_unit_is_not_listed_as_a_quantity(output, value):
    """An output the registry declares no unit for ("-") is not a pure
    number by code's arithmetic: fed on to an input in m, it adds nothing to
    check beyond what it rests on itself."""
    s = _sheet_with([{"id": "g1", "value": 12, "unit": "m"}, {"id": "g2", "value": 3, "unit": "m"},
                     {"id": "g3", "value": 2, "unit": "m"}],
                    [dict(_trench_plan(), output=output), _trench_plan(length="s1", sid="s2")])
    s = ws.mark_given(ws.run_step(s, "s1", "c1"), lambda number, unit: "user")
    assert s["steps"][0]["result"]["value"] == value
    assert ws.rests_quantities(s, "s2") == []
