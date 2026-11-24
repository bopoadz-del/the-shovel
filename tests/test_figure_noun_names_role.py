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
