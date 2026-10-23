"""Every formula and tool declares the name a user reads, and answers show
that name, never the internal one (F-UI-PASS: "Source: construction_calc
delay_damages_daily" reached the chat)."""
import pytest

from app.agents.core import tool_registry
from app.lib import formula_registry
from app.lib.source_labels import calculator_label, input_phrase, tool_label


def test_every_formula_declares_a_display_name_that_is_not_its_id():
    specs = formula_registry.all_specs()
    assert specs
    for spec in specs:
        assert spec.display_name.strip(), spec.name
        assert "_" not in spec.display_name, spec.name
        assert spec.display_name != spec.name


def test_every_registered_tool_declares_a_display_name():
    tool_registry.load()
    assert tool_registry._TOOLS
    for name, spec in tool_registry._TOOLS.items():
        assert spec.display_name.strip(), name
        assert "_" not in spec.display_name, name


def test_a_formula_without_a_display_name_cannot_register():
    with pytest.raises(ValueError):
        formula_registry.formula(owner="base", description="d", display_name="",
                                 inputs={}, outputs={})
    with pytest.raises(TypeError):
        formula_registry.formula(owner="base", description="d", inputs={}, outputs={})


@pytest.mark.parametrize("key,value,unit,shown", [
    ("contract_amount", 50000000, "currency", "contract amount 50,000,000"),
    ("length_m", 12, "m", "length 12 m"),
    ("rate_percent", 0.1, "%", "rate 0.1%"),
    ("early_start", 10.0, "", "early start 10"),
])
def test_inputs_read_as_words(key, value, unit, shown):
    assert input_phrase(key, value, unit) == shown


def test_labels_carry_no_internal_names():
    for spec in formula_registry.all_specs():
        label = calculator_label(spec.name, {k: 1 for k in spec.inputs})
        assert spec.name not in label and "construction_calc" not in label
        assert label.startswith(spec.display_name + " — platform calculator")
    tool_registry.load()
    for name in tool_registry._TOOLS:
        assert name not in tool_label(name, {"gross_valuation": 1})
