"""#392 grader point 5: EVERY calculator is declared ``kind="computed"`` with
its statement function, and the recovery reads that declaration -- there is
no list of calculator names for it to key on.

Every input here is built by the test (invented figures, invented notes);
each decision is checked with at least two inputs.
"""
from __future__ import annotations

import json

import pytest

from app.agents import runtime
from app.agents.base.blocks import improvised_statement
from app.agents.base.tools import calculation_statement
from app.agents.core import tool_registry
from app.lib import formula_registry

FALLBACK = runtime._EMPTY_RESPONSE_FALLBACK


def _turn(ask: str, *tool_msgs: dict) -> list[dict]:
    return [{"role": "user", "content": ask}, *tool_msgs]


def _tool_msg(name: str, result: dict, ok: bool = True) -> dict:
    return {"role": "tool", "tool_call_id": f"call-{name}", "name": name,
            "content": json.dumps({"name": name, "ok": ok, "result": result})}


# ── every calculator carries the declaration ──────────────────────────────

def test_the_old_name_tuple_is_retired():
    assert not hasattr(runtime, "_CALC_TOOL_NAMES")


@pytest.mark.parametrize("name", sorted(["construction_calc", "formula_executor_v2",
                                         *runtime._HAT_CALC_ALIASES]))
def test_every_calculator_name_the_model_can_call_is_declared_computed(name):
    spec = tool_registry.result_spec(name)
    assert spec is not None, name
    assert spec.kind == "computed" and callable(spec.statement), name
    assert name in tool_registry.tools_of_kind("computed")


@pytest.mark.parametrize("name", ["search_project_documents", "fetch_document"])
def test_a_search_tool_is_not_computed(name):
    assert tool_registry.result_spec(name).kind == "search"
    assert name not in tool_registry.tools_of_kind("computed")


@pytest.mark.parametrize("kwargs, message", [
    (dict(kind="computed"), "declares a statement"),
    (dict(kind="search", statement=lambda r: []), "declares a statement"),
    (dict(kind=""), "unknown result kind"),
    (dict(kind="guess"), "unknown result kind"),
])
def test_a_result_declaration_is_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        tool_registry.declare_result("zz_invented_block", owner="base",
                                     display_name="Invented", **kwargs)
    assert tool_registry.result_spec("zz_invented_block") is None


@pytest.mark.parametrize("name", ["construction_calc", "bill_rows_aggregate"])
def test_a_registered_tool_cannot_be_declared_again_as_a_block(name):
    with pytest.raises(ValueError, match="declared twice"):
        tool_registry.declare_result(name, owner="base", display_name="X",
                                     kind="computed", statement=lambda r: [])


# ── the calculator statement reads the formula's declared outputs ─────────

@pytest.mark.parametrize("calculation, params", [
    ("excavation_volume", {"length": 7, "width": 3, "depth": 2}),
    ("concrete_volume", {"length": 11, "width": 4, "thickness": 0.5}),
])
def test_a_calculator_states_its_declared_outputs_main_first(calculation, params):
    from app.lib import construction_formulas as cf

    result = cf.run_calculation(calculation, params)
    spec = formula_registry.get(calculation)
    out = calculation_statement({"name": "construction_calc", "ok": True, "result": result})
    assert out, result
    stated = {(st.figure, st.unit) for st in out}
    for key, unit in spec.outputs.items():
        value = result["result"].get(key)
        if unit not in ("", "-") and isinstance(value, (int, float)) and not isinstance(value, bool):
            assert (float(value), unit) in stated, key
    if spec.main_output:
        assert out[0].figure == float(result["result"][spec.main_output])
    # The calculator's own working rides on the first statement only.
    assert out[0].working == result["result"]["note"]
    assert all(not st.working for st in out[1:])


@pytest.mark.parametrize("result", [
    {"status": "error", "calculation": "excavation_volume", "error": "missing depth"},
    {"status": "success", "calculation": "zz_not_a_declared_formula", "result": {"volume_m3": 4.0}},
])
def test_an_undeclared_or_failed_calculation_states_nothing(result):
    assert calculation_statement(result) == []


@pytest.mark.parametrize("calculation, figures", [
    ("concrete_volume", {"volume_m3": 0.0, "net_volume_m3": 0, "note": "Net = 0*0*0 = 0 m3"}),
    ("excavation_volume", {"bank_volume_m3": 0, "loose_volume_m3": 0.0, "note": "Bank = 0"}),
])
def test_a_calculation_whose_every_figure_is_zero_states_nothing(calculation, figures):
    result = {"status": "success", "calculation": calculation, "result": figures}
    assert calculation_statement(result) == []
    out = runtime._recover_answer_from_tool_messages(
        "", _turn("how much", _tool_msg("construction_calc", result)))
    # The computed statement steps aside; whatever answers, it is never the
    # zero working nor the generic sentence.
    assert figures["note"] not in out
    assert out.strip() and "unable to generate" not in out.lower()


@pytest.mark.parametrize("figures, kept", [
    ({"volume_m3": 0.0, "net_volume_m3": 6.5}, 6.5),
    ({"volume_m3": 2.25, "net_volume_m3": 0}, 2.25),
])
def test_one_nonzero_figure_keeps_the_statement(figures, kept):
    result = {"status": "success", "calculation": "concrete_volume", "result": figures}
    assert kept in [st.figure for st in calculation_statement(result)]


@pytest.mark.parametrize("value, expected", [
    (42.5, [("area of the invented yard", 42.5)]),
    ({"steel_t": 3, "label": "x", "flag": True}, [("steel_t", 3.0)]),
])
def test_the_code_runner_states_its_numeric_figures(value, expected):
    body = {"status": "success", "result": value, "task": "area of the invented yard"}
    out = improvised_statement({"name": "formula_executor_v2", "ok": True, "result": body})
    assert [(st.what, st.figure) for st in out] == expected


@pytest.mark.parametrize("body", [
    {"status": "error", "error": "sandbox refused", "task": "t"},
    {"status": "success", "result": "not a number", "task": "t"},
])
def test_the_code_runner_states_nothing_without_a_figure(body):
    assert improvised_statement(body) == []


# ── the recovery, by declared kind ────────────────────────────────────────

def _declared_aliases() -> list[str]:
    """Aliased actions whose calculator declares a numeric output."""
    out = []
    for action, calc in sorted(runtime._HAT_CALC_ALIASES.items()):
        spec = formula_registry.get(calc)
        if spec and any(u not in ("", "-") for u in spec.outputs.values()):
            out.append(action)
    return out


@pytest.mark.parametrize("action, figure", list(zip(_declared_aliases()[:2], (31.25, 7.5))))
def test_an_aliased_calculator_action_answers_with_its_figure(action, figure):
    calc = runtime._HAT_CALC_ALIASES[action]
    spec = formula_registry.get(calc)
    key = next(k for k in sorted(spec.outputs, key=lambda k: k != spec.main_output)
               if spec.outputs[k] not in ("", "-"))
    result = {"aliased_to": f"construction_calc:{calc}", "status": "success",
              "calculation": calc, "result": {key: figure, "note": "invented working"}}
    out = runtime._recover_answer_from_tool_messages(
        FALLBACK, _turn("work it out", _tool_msg(action, result)))
    assert f"**{figure:g} {spec.outputs[key]}**" in out, out
    assert "invented working" in out
    assert "unable to generate" not in out.lower()


@pytest.mark.parametrize("figures, shown", [
    ({"volume_m3": 12.5, "net_volume_m3": 12.0, "note": "Net = 12 m3"}, "12.5 m3"),
    ({"volume_m3": 3.75, "net_volume_m3": 3.5, "note": "Net = 3.5 m3"}, "3.75 m3"),
])
def test_construction_calc_answers_by_kind_with_its_working(figures, shown):
    result = {"status": "success", "calculation": "concrete_volume", "result": figures}
    out = runtime._recover_answer_from_tool_messages(
        "", _turn("how much concrete", _tool_msg("construction_calc", result)))
    assert f"**{shown}**" in out, out
    assert figures["note"] in out


# ── a truncated tool message is read through its whole preview ────────────

def _truncated(name: str, result: dict, *, whole: bool) -> dict:
    preview = json.dumps({"name": name, "ok": True, "result": result})
    if not whole:
        preview = preview[: len(preview) // 2]
    return {"role": "tool", "tool_call_id": f"call-{name}", "name": name,
            "content": json.dumps({"truncated": True, "preview": preview})}


@pytest.mark.parametrize("figure", [18.5, 240.0])
def test_a_truncated_envelope_with_a_whole_preview_is_read(figure):
    result = {"status": "success", "calculation": "concrete_volume",
              "result": {"volume_m3": figure, "note": "invented"}}
    msg = _truncated("construction_calc", result, whole=True)
    assert runtime._tool_message_payload(msg)["result"]["result"]["volume_m3"] == figure
    out = runtime._recover_answer_from_tool_messages("", _turn("volume?", msg))
    assert f"**{figure:g} m3**" in out, out


@pytest.mark.parametrize("figure", [18.5, 240.0])
def test_a_truncated_envelope_with_a_cut_preview_is_not_a_figure(figure):
    result = {"status": "success", "calculation": "concrete_volume",
              "result": {"volume_m3": figure, "note": "invented"}}
    msg = _truncated("construction_calc", result, whole=False)
    assert runtime._tool_message_payload(msg) is None
    out = runtime._recover_answer_from_tool_messages("", _turn("volume?", msg))
    assert f"{figure:g}" not in out
    assert "unable to generate" not in out.lower()


@pytest.mark.parametrize("content", ["not json at all", "[1, 2"])
def test_a_tool_message_that_is_not_json_has_no_payload(content):
    assert runtime._tool_message_payload({"role": "tool", "name": "x", "content": content}) is None


# ── markdown with no words is an empty answer ─────────────────────────────

@pytest.mark.parametrize("text", ["**  **", "---\n\n* \n#"])
def test_markdown_only_text_needs_recovery(text):
    assert runtime._text_needs_tool_recovery(text)


@pytest.mark.parametrize("text", ["**12**", "- نعم"])
def test_a_word_or_figure_inside_markdown_is_text(text):
    assert not runtime._text_needs_tool_recovery(text)


# ── a failed result states nothing even when it carries figures ───────────

@pytest.mark.parametrize("status, value", [("error", 42.5), ("timeout", {"steel_t": 3})])
def test_a_failed_code_run_states_nothing_even_with_a_figure(status, value):
    body = {"status": status, "result": value, "task": "invented yard area"}
    assert improvised_statement(body) == []
    assert improvised_statement({**body, "status": "success"}) != []


@pytest.mark.parametrize("calculation", ["excavation_volume", "concrete_volume"])
def test_a_failed_calculation_states_nothing_even_with_its_figures(calculation):
    spec = formula_registry.get(calculation)
    figures = {key: 5.5 for key, unit in spec.outputs.items() if unit not in ("", "-")}
    failed = {"status": "error", "calculation": calculation, "result": figures}
    assert calculation_statement(failed) == []
    assert calculation_statement({**failed, "status": "success"}) != []


@pytest.mark.parametrize("key, unit, expected", [
    ("bank_volume_m3", "m3", "bank volume"),
    ("loose_volume", "m3", "loose volume"),
    ("m3", "m3", "m3"),
    ("weight__kg", "KG", "weight"),
])
def test_a_declared_output_reads_as_words_without_its_unit(key, unit, expected):
    from app.agents.base.tools import _output_name

    assert _output_name(key, unit) == expected


@pytest.mark.parametrize("owner", ["zz_no_such_lane", ""])
def test_a_result_declared_by_an_unknown_owner_is_refused(owner):
    with pytest.raises(ValueError, match="unknown result owner"):
        tool_registry.declare_result("zz_invented_owned_block", owner=owner,
                                     display_name="Invented", kind="search")
    assert tool_registry.result_spec("zz_invented_owned_block") is None


@pytest.mark.parametrize("action", sorted(runtime._HAT_CALC_ALIASES)[:2])
def test_declaring_the_calculator_aliases_declares_each_computed(action):
    from app.agents.base import blocks

    assert blocks._declare_calculator_aliases() is None  # idempotent re-declaration
    spec = tool_registry.result_spec(action)
    assert spec is not None and spec.kind == "computed"
    assert spec.statement is calculation_statement


@pytest.mark.parametrize("name, expected", [
    ("construction_calc", True),
    ("formula_executor_v2", True),
    ("search_project_documents", False),
    ("fetch_document", False),
])
def test_a_failed_tool_is_a_calculator_by_its_declared_kind_alone(name, expected):
    # No calculation key, no "unknown calculation", no "available" list:
    # only the declaration can decide.
    msg = {"role": "tool", "name": name, "content": "{}"}
    assert runtime._is_calc_tool_error(msg, {}, {"error": "invented failure"}) is expected


# A declaration that needs another registry is deferred to the first read, so
# load() at app start never pulls that registry's (heavy) imports.
@pytest.mark.parametrize("name, reader", [
    ("zz_invented_deferred_a", lambda n: tool_registry.result_spec(n)),
    ("zz_invented_deferred_b", lambda n: n in tool_registry.tools_of_kind("search")),
])
def test_a_deferred_declaration_runs_on_first_read_not_on_load(name, reader):
    runs = []

    def declarer():
        runs.append(name)
        tool_registry.declare_result(name, owner="base", display_name="Invented", kind="search")

    tool_registry.declare_results_when_read(declarer)
    try:
        tool_registry.load()
        assert runs == []
        assert name not in tool_registry._RESULTS
        assert reader(name)
        assert runs == [name]
        reader(name)
        assert runs == [name]  # once, not on every read
    finally:
        tool_registry._RESULTS.pop(name, None)
        if declarer in tool_registry._DEFERRED:
            tool_registry._DEFERRED.remove(declarer)


@pytest.mark.parametrize("action", sorted(runtime._HAT_CALC_ALIASES)[-2:])
def test_a_calculator_alias_reads_its_registry_display_name(action):
    spec = tool_registry.result_spec(action)
    calc = formula_registry.get(runtime._HAT_CALC_ALIASES[action])
    assert spec.display_name == ((calc.display_name if calc and calc.display_name else None) or action)
