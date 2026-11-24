"""A knowledge input the user leaves out takes the value its course declares.

Every entry here is built by the test; nothing is read from the shipped
knowledge base. Each decision is checked on two or more inputs.
"""
import pytest

from app.blocks import _knowledge as kb


def _entry(rate_values=None, extra_ref="course text, section 9, p1"):
    rate = {"unit": "tonne/hour", "desc": "production rate"}
    if rate_values is not None:
        rate["course_values"] = rate_values
    return {
        "id": "t.hours",
        "type": "formula",
        "title": "Machine hours to place a quantity",
        "expression": "mass_placed / rate",
        "variables": {
            "mass_placed": {"unit": "tonne", "desc": "quantity"},
            "rate": rate,
        },
        "thresholds": {"unit": "hour"},
        "provenance": {"course_reference": extra_ref},
    }


_LAYERED = [
    {"value": 150, "when": ["lower", "layer"]},
    {"value": 90, "when": ["top"]},
]


@pytest.mark.parametrize(
    "ask, expected",
    [
        ("hours to place 900 t of the top layer", 90.0),
        ("hours to place 900 t of the lower layer", 150.0),
    ],
)
def test_the_value_stated_for_the_case_the_ask_names_fills_the_input(ask, expected):
    bound = kb.bind_entry_variables(_entry(_LAYERED), ask)
    assert bound == {"mass_placed": 900.0, "rate": expected}


@pytest.mark.parametrize(
    "ask, typed",
    [
        ("hours to place 900 t of the top layer at a rate of 75", 75.0),
        ("hours to place 900 t of the lower layer, rate = 160", 160.0),
    ],
)
def test_a_typed_figure_wins_over_the_course_value(ask, typed):
    bound = kb.bind_entry_variables(_entry(_LAYERED), ask)
    assert bound == {"mass_placed": 900.0, "rate": typed}


@pytest.mark.parametrize(
    "ask",
    [
        "hours to place 900 t",  # no case named: no value applies
        "hours to place 900 t of the top layer over the lower layer",  # two apply
        # a figure left over may be the user's own rate: the course does not fill
        "hours to place 900 t of the top layer with 75 loads each hour",
    ],
)
def test_no_value_or_two_values_applying_leaves_the_input_unbound(ask):
    assert kb.bind_entry_variables(_entry(_LAYERED), ask) is None


@pytest.mark.parametrize(
    "values, bound",
    [
        ([{"value": 120}], True),  # no "when": applies always
        (None, False),  # nothing declared
        ("120", False),  # not a list
        ([{"value": "120"}], False),  # not a number
    ],
)
def test_only_a_declared_numeric_value_fills_the_input(values, bound):
    got = kb.bind_entry_variables(_entry(values), "hours to place 600 t")
    assert (got == {"mass_placed": 600.0, "rate": 120.0}) is bound
    if not bound:
        assert got is None


@pytest.mark.parametrize("spec", ["not a dict", None, {"unit": "tonne/hour"}])
def test_a_spec_with_no_course_values_gives_none(spec):
    assert kb._course_value(spec, "any ask") is None


@pytest.mark.parametrize(
    "values, named",
    [
        ({"mass_placed": 900.0, "rate": 90.0}, {"rate"}),
        ({"mass_placed": 900.0, "rate": 75.0}, set()),  # typed, not the course's
        ({"mass_placed": 900.0}, set()),  # input not supplied
    ],
)
def test_course_inputs_names_only_inputs_holding_a_course_value(values, named):
    out = kb.course_inputs(_entry(_LAYERED), values)
    assert set(out) == named
    for name in named:
        assert out[name]["source"] == "course text, section 9, p1"
        assert out[name]["value"] == values[name]


@pytest.mark.parametrize(
    "ask, has_course_input",
    [
        ("hours to place 900 t of the top layer", True),
        ("hours to place 900 t of the top layer at a rate of 75", False),
    ],
)
def test_evaluation_names_the_course_as_source_of_a_filled_input(
    monkeypatch, ask, has_course_input
):
    entry = _entry(_LAYERED, extra_ref="course text, section 4, p7")
    monkeypatch.setattr(kb, "get_rule", lambda rule_id: entry)
    values = kb.bind_entry_variables(entry, ask)
    out = kb.evaluate("t.hours", **values)
    assert out["result"] == pytest.approx(900.0 / values["rate"])
    if has_course_input:
        assert out["course_inputs"] == {
            "rate": {"value": 90.0, "when": ["top"], "source": "course text, section 4, p7"}
        }
    else:
        assert "course_inputs" not in out


@pytest.mark.parametrize(
    "entry",
    [{"variables": {}}, {"variables": ["mass_placed", "rate"]}, {}],
)
def test_an_entry_with_no_variable_table_binds_nothing(entry):
    # The guard is the only thing that returns None here: with it gone
    # the binder walks an empty table and returns {} (or fails on a list).
    assert kb.bind_entry_variables(entry, "place 500 t at 150 t/hr") is None
