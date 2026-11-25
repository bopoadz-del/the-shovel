"""Edges of the role-word bind: a role written in another unit of the same
quantity, a role two inputs share, an input that cannot hold a figure, a
role already filled, and a role figure that must not fall through to the
bare-number pass.

Each calculator is built here, so no registered formula and no product
answer supplies an expected value. Every decision has two inputs.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from app.lib.construction_formulas import extract_calculation_params_from_text as read


def _vocab(monkeypatch, by_kind):
    monkeypatch.setattr(cf, "_role_words_by_kind", lambda: by_kind)


def _retention(valuation, retention_percent):
    return valuation * retention_percent / 100


def _count_and_retention(count, retention_percent):
    return count * retention_percent


# The role word opens each input name but is no label of its own, so the
# figure reaches the role bind and not the label-after-figure pass.
def _two_waste(waste_allowance_percent, waste_reserve_percent):
    return waste_allowance_percent + waste_reserve_percent


def _waste_text(waste_allowance_percent: str, retention_percent):
    return retention_percent


def _waste_list(waste_allowance_percent: list, retention_percent):
    return retention_percent


def _waste_number(waste_allowance_percent: float, retention_percent):
    return waste_allowance_percent + retention_percent


def _waste(quantity, waste_percent):
    return quantity * (1 + waste_percent / 100)


@pytest.mark.parametrize("word, figure", [("markup", 12), ("overhead", 7.5)])
def test_a_role_declared_in_another_unit_of_the_same_quantity_is_still_a_role(monkeypatch, word, figure):
    # The registry declares this role as a fraction; the figure is written in %.
    _vocab(monkeypatch, {"fraction": {word}, "%": {"retention"}})
    assert cf._role_named_after(f"{figure} % {word} on top", len(f"{figure} %"), "%") == word
    assert "retention_percent" not in read(_retention, f"Add {figure} % {word} on top.")


@pytest.mark.parametrize("word, figure", [("markup", 12), ("overhead", 7.5)])
def test_a_role_of_another_quantity_is_not_a_role_of_a_percentage(monkeypatch, word, figure):
    # Must not fire: the word names only a length input, so the % figure
    # still binds by unit to the one open percentage.
    _vocab(monkeypatch, {"m": {word}, "%": {"retention"}})
    assert cf._role_named_after(f"{figure} % {word} on top", len(f"{figure} %"), "%") is None
    assert read(_retention, f"Take {figure} % {word} on top.").get("retention_percent") == pytest.approx(figure)


@pytest.mark.parametrize("figure", [7.5, 2.25])
def test_a_role_figure_is_spent_and_not_read_again_as_a_bare_count(monkeypatch, figure):
    # A decimal figure left unspent would be read again by the bare pass
    # as its whole part ("7" of "7.5 %").
    _vocab(monkeypatch, {"%": {"waste", "retention"}})
    got = read(_count_and_retention, f"Add {figure} % waste to the order.")
    assert "count" not in got
    assert "retention_percent" not in got


@pytest.mark.parametrize("figure", [5, 8])
def test_a_role_word_two_open_inputs_share_binds_neither(monkeypatch, figure):
    _vocab(monkeypatch, {"%": {"waste"}})
    got = read(_two_waste, f"Add {figure} % waste to the order.")
    assert "waste_allowance_percent" not in got
    assert "waste_reserve_percent" not in got


@pytest.mark.parametrize("fn", [_waste_text, _waste_list])
def test_a_role_input_that_cannot_hold_a_figure_is_not_filled(monkeypatch, fn):
    _vocab(monkeypatch, {"%": {"waste", "retention"}})
    got = read(fn, "Add 5 % waste to the order.")
    assert "waste_allowance_percent" not in got
    assert "retention_percent" not in got


@pytest.mark.parametrize("first, second", [(5, 7), (3, 11)])
def test_a_role_already_filled_is_not_overwritten_by_a_later_figure(monkeypatch, first, second):
    _vocab(monkeypatch, {"%": {"waste", "retention"}})
    got = read(_waste_number, f"Allow {first} % waste, or {second} % waste on site.")
    assert got.get("waste_allowance_percent") == pytest.approx(first)
    assert "retention_percent" not in got


@pytest.mark.parametrize("figure", [4, 6.5])
def test_a_role_input_that_holds_a_figure_is_filled(monkeypatch, figure):
    # The positive twin of the two guards above: one open input of the role,
    # typed to hold a number.
    _vocab(monkeypatch, {"%": {"waste", "retention"}})
    got = read(_waste_number, f"Allow {figure} % waste.")
    assert got.get("waste_allowance_percent") == pytest.approx(figure)
    assert "retention_percent" not in got
