"""A figure whose next word names a role binds only to the input of that role.

"5 % waste" is a waste figure: it never fills a retention or a delay-rate
input just because that input is the one open percentage. The role words are
read from the names of the inputs themselves (the registry's and the
calculator's own), never from a list of cases.

Each calculator here is built in the test, so no registered formula and no
product answer supplies an expected value. Every decision has two inputs.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from app.lib.construction_formulas import extract_calculation_params_from_text as read
from app.lib.formula_registry import FormulaSpec

# The real vocabulary builder, kept before the fixture below replaces it.
_REAL_ROLE_WORDS = cf._role_words_by_kind


@pytest.fixture(autouse=True)
def _role_words(monkeypatch):
    # The registry's vocabulary is replaced by one the test owns, so the
    # decision reads only the inputs built here.
    monkeypatch.setattr(cf, "_role_words_by_kind", lambda: {"%": {"waste", "retention", "markup"}})


def _retention(valuation, retention_percent):
    return valuation * retention_percent / 100


def _waste(quantity, waste_percent):
    return quantity * (1 + waste_percent / 100)


def _two_roles(waste_percent, retention_percent):
    return waste_percent + retention_percent


@pytest.mark.parametrize("figure", [5, 7.5])
def test_a_figure_named_for_another_role_does_not_bind(figure):
    ask = f"Add {figure} % waste to the order."
    assert "retention_percent" not in read(_retention, ask)


@pytest.mark.parametrize("figure", [5, 7.5])
def test_a_figure_named_for_its_own_role_binds(figure):
    ask = f"Add {figure} % waste to the order."
    assert read(_waste, ask).get("waste_percent") == pytest.approx(figure)


@pytest.mark.parametrize("figure", [5, 10])
def test_a_figure_with_no_role_word_still_binds_by_unit(figure):
    ask = f"Take {figure} % on the valuation."
    assert read(_retention, ask).get("retention_percent") == pytest.approx(figure)


@pytest.mark.parametrize("waste, retention", [(5, 10), (3, 8)])
def test_two_figures_of_one_kind_each_bind_to_the_role_they_name(waste, retention):
    ask = f"Allow {waste} % waste and {retention} % retention."
    got = read(_two_roles, ask)
    assert got.get("waste_percent") == pytest.approx(waste)
    assert got.get("retention_percent") == pytest.approx(retention)


@pytest.mark.parametrize("word", ["waste", "markup"])
def test_the_role_word_is_read_only_straight_after_the_figure(word):
    assert cf._role_named_after(f"12 % {word} here", 4, "%") == word
    assert cf._role_named_after(f"12 % on the {word}", 4, "%") is None


@pytest.mark.parametrize("word", ["waste", "retention"])
def test_a_word_of_another_kind_is_not_a_role_of_this_kind(word):
    assert cf._role_named_after(f"12 m {word}", 4, "m") is None
    assert cf._role_named_after(f"12 m {word}", 4, "m", own_words={word}) == word


@pytest.mark.parametrize("name, row_unit, declared, want", [
    ("span_mm", "m", "mm", "mm"),
    ("load_kn", "kN", "-", "kN"),
])
def test_the_input_unit_is_the_declared_one_first_then_the_described_one(name, row_unit, declared, want):
    assert cf._input_unit_token(name, row_unit, declared) == want


@pytest.mark.parametrize("name", ["span_mm", "depth_m"])
def test_with_nothing_declared_the_input_unit_is_read_from_its_name(name):
    assert cf._input_unit_token(name, "", "") == cf._unit_from_name(name)
    assert cf._input_unit_token(name, "", "") != ""


@pytest.mark.parametrize("name, label, want", [
    ("waste_percent", "waste allowance", {"waste", "percent", "allowance"}),
    ("rate_percent", "", {"rate", "percent"}),
])
def test_an_input_is_named_by_the_words_of_its_name_and_label(name, label, want):
    # The one reader of input words (defined once, further down the module).
    assert cf._input_words(name, label) == want


def _spec(name, fn, inputs):
    return FormulaSpec(name=name, fn=fn, owner="test", description="", inputs=inputs)


def _w(waste_percent, length_m):
    return waste_percent + length_m


def _r(retention_percent, total_m, total_kg):
    return retention_percent + total_m + total_kg


def _t(total_percent):
    return total_percent


@pytest.mark.parametrize("specs", [
    [_spec("w", _w, {"waste_percent": "%", "length_m": "m"})],
    [_spec("w", _w, {"waste_percent": "%", "length_m": "m"}),
     _spec("r", _r, {"retention_percent": "%", "total_m": "m", "total_kg": "kg"}),
     _spec("t", _t, {"total_percent": "%"})],
])
def test_role_words_are_read_once_per_registry_state(monkeypatch, specs):
    from app.lib import formula_registry

    monkeypatch.setattr(formula_registry, "all_specs", lambda: specs)
    monkeypatch.setattr(cf, "_ROLE_WORDS", {"count": -1, "by_kind": {}})
    reads = []
    real_describe = cf.describe_calculation_params
    monkeypatch.setattr(cf, "describe_calculation_params", lambda fn, **kw: reads.append(fn) or real_describe(fn, **kw))
    first = _REAL_ROLE_WORDS()
    second = _REAL_ROLE_WORDS()
    assert second == first
    assert len(reads) == len(specs)
    assert "waste" in first["%"]


_THREE_KINDS = [
    _spec("r", _r, {"retention_percent": "%", "total_m": "m", "total_kg": "kg"}),
    _spec("t", _t, {"total_percent": "%"}),
]
_TWO_KINDS = [
    _spec("r", _r, {"retention_percent": "%", "total_m": "m", "total_kg": "kg"}),
]


@pytest.mark.parametrize("specs, total_is_a_role", [(_THREE_KINDS, False), (_TWO_KINDS, True)])
def test_a_word_naming_inputs_of_three_kinds_is_a_joiner_not_a_role(monkeypatch, specs, total_is_a_role):
    from app.lib import formula_registry

    monkeypatch.setattr(formula_registry, "all_specs", lambda: specs)
    monkeypatch.setattr(cf, "_ROLE_WORDS", {"count": -1, "by_kind": {}})
    words = _REAL_ROLE_WORDS()
    assert ("total" in words["m"]) is total_is_a_role
    assert "retention" in words["%"]
