"""An optional input the user did not give runs on the calculator's default.

LANE3 (b656776, 3/3 FAIL; f2de03b UNSTABLE): "Build up a selling rate for
100 m3 of concrete using your standard rates." The calculator gives 308.83
SAR/m3 on its declared defaults, but one run priced the build-up from a
project document's rates (1,023.62) and two refused for want of rates. The
model could not see which inputs were optional: the concrete build-up is
listed under "Other calculators", whose lines named labels only. The tool
contract (order 0700Z): an optional input left out runs on the calculator's
declared default, which the answer states; a figure read from a document goes
in only when the user asks for that document's or project's figures.
"""
from app.agents.driver import tools as driver_tools
from app.lib import construction_formulas as cf
from app.lib import formula_registry


def _description():
    driver_tools._SCHEMA_CACHE.clear()
    return driver_tools.calculator_schema(None)["function"]["description"]


def test_the_contract_says_an_optional_input_runs_on_its_default_and_is_stated():
    text = _description()
    assert "optional one runs on the calculator's own declared default" in text
    assert "the answer states that default" in text
    assert "only when the user asks for that document's or project's figures" in text


def test_every_listed_calculator_marks_its_optional_inputs():
    """Both the hat's own calculators and the 'Other calculators' lines: for
    every formula, each optional input is marked where the model reads it."""
    text = _description()
    for spec in formula_registry.all_specs():
        at = text.find(f"- {spec.name}:")
        if at < 0:
            continue
        line = text[at:text.find("\n", at) if text.find("\n", at) > 0 else len(text)]
        for key, p in formula_registry.parameters(spec).items():
            piece = line[line.find(f"{key} = "):] if f"{key} = " in line else ""
            if not piece:
                continue
            piece = piece.split(";")[0]
            assert ("optional" in piece) == (not p.required), (spec.name, key, piece)


def test_the_concrete_build_up_on_its_defaults_is_the_hand_figure():
    token = cf.set_current_ask("Build up a selling rate for 100 m3 of concrete using your standard rates.")
    try:
        out = cf.run_calculation("cost_buildup_concrete", {"quantity_m3": 100})
    finally:
        cf._CURRENT_ASK.reset(token)
    assert out["status"] == "success"
    assert out["result"]["selling_price_sar_m3"] == 308.83
