"""D2: the commercial hat's rate lookup prices a bill from the rate card.

``rate_card_lookup`` (and ``construction:benchmark_lookup``) run the
container's ``benchmark_lookup``. These tests check that it reads the same
rate card as the price-boq endpoint (``app.lib.boq_pricing``), never guesses
an asset type, and falls back to the historical book only with a label.

Every expected value is computed from the card the test reads, never typed.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer, boq
from app.lib import boq_pricing


def _cards() -> list[tuple[str, str]]:
    return [(a, c) for a, ccys in boq_pricing.available_assets().items() for c in ccys]


def _row_line(asset: str, ccy: str, quantity: float) -> dict:
    """A bill line built from one of the card's own rows: a description that
    the categoriser files under that row's category, in that row's unit."""
    for row in boq_pricing._CARD["cards"][asset][ccy]:
        for name, pattern in boq_pricing.CATS:
            if name != row["cat"]:
                continue
            words = [w for w in pattern.replace("\\b", "").split("|")
                     if w.isalpha() and len(w) > 3]
            for word in words:
                if boq_pricing.categorize(word) == row["cat"]:
                    return {"description": word, "unit": row["unit"],
                            "quantity": quantity}
    raise AssertionError(f"no row of {asset}/{ccy} has a plain-word category")


@pytest.fixture
def container():
    return ConstructionContainer()


@pytest.mark.parametrize("name", ["benchmark_lookup", "_price_bill_from_rate_card"])
def test_rate_lookup_lives_on_the_boq_mixin(name):
    assert getattr(ConstructionContainer, name) is getattr(boq.ConstructionBoqMixin, name)


# ── resolve_card: two inputs per decision ───────────────────────────────


@pytest.mark.parametrize("asset,ccy", _cards())
def test_resolve_card_ignores_case_and_blanks(asset, ccy):
    assert boq_pricing.resolve_card(f"  {asset.upper()} ", ccy.lower()) == (asset, ccy)
    assert boq_pricing.resolve_card(asset.lower(), ccy) == (asset, ccy)


@pytest.mark.parametrize("asset", list(boq_pricing.available_assets()))
def test_resolve_card_without_currency_takes_the_cards_first(asset):
    first = boq_pricing.available_assets()[asset][0]
    assert boq_pricing.resolve_card(asset) == (asset, first)
    assert boq_pricing.resolve_card(asset, "") == (asset, first)


@pytest.mark.parametrize("bad", ["", None, "no such asset type"])
def test_resolve_card_never_guesses_an_asset(bad):
    assert boq_pricing.resolve_card(bad, "USD") == (None, None)


@pytest.mark.parametrize("asset", list(boq_pricing.available_assets()))
def test_resolve_card_rejects_a_currency_the_card_lacks(asset):
    assert boq_pricing.resolve_card(asset, "XXX") == (asset, None)
    assert boq_pricing.resolve_card(asset, "ZZZ") == (asset, None)


# ── a bill priced from chat ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("asset,ccy", _cards())
async def test_bill_is_priced_from_the_card(container, asset, ccy):
    lines = [_row_line(asset, ccy, 7.0), _row_line(asset, ccy, 3.0)]
    want, want_summary = boq_pricing.price_line_items(lines, asset, ccy)

    got = await container.benchmark_lookup(
        {"line_items": lines}, {"asset_type": asset.lower(), "currency": ccy},
    )

    assert got["status"] == "success"
    assert got["priced_items"] == want
    assert got["summary"] == want_summary
    assert got["summary"]["asset_type"] == asset
    assert got["summary"]["no_rate"] == 0
    assert got["summary"]["grand_total"] > 0
    assert f"{asset}/{ccy}" in got["rate_basis"]
    assert got["card_source"] == boq_pricing._CARD["_meta"]["source"]
    assert got["card_source"]


@pytest.mark.asyncio
@pytest.mark.parametrize("qty", [4.0, 11.0])
async def test_bill_reads_qty_as_quantity(container, qty):
    asset, ccy = _cards()[0]
    line = _row_line(asset, ccy, qty)
    as_qty = {k: v for k, v in line.items() if k != "quantity"} | {"qty": qty}

    got = await container.benchmark_lookup(
        {"line_items": [as_qty]}, {"asset_type": asset, "currency": ccy},
    )

    assert got["status"] == "success"
    assert got["priced_items"][0]["qty"] == qty
    assert got["summary"]["grand_total"] == pytest.approx(
        qty * got["priced_items"][0]["rate"], abs=0.01,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_arg", [None, "no such asset type"])
async def test_bill_without_a_card_asset_lists_the_options(container, asset_arg):
    asset, ccy = _cards()[0]
    params = {"asset_type": asset_arg} if asset_arg else {}

    got = await container.benchmark_lookup(
        {"line_items": [_row_line(asset, ccy, 2.0)]}, params,
    )

    assert got["status"] == "error"
    assert got["missing"] == ["asset_type"]
    assert got["rate_card_asset_types"] == boq_pricing.available_assets()
    assert "priced_items" not in got


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_ccy", ["XXX", "ZZZ"])
async def test_bill_with_a_currency_the_card_lacks_says_so(container, bad_ccy):
    asset, ccy = _cards()[0]

    got = await container.benchmark_lookup(
        {"line_items": [_row_line(asset, ccy, 2.0)]},
        {"asset_type": asset, "currency": bad_ccy},
    )

    assert got["status"] == "error"
    assert got["missing"] == ["currency"]
    assert "priced_items" not in got


@pytest.mark.asyncio
@pytest.mark.parametrize("junk", [["not a line"], [None, 3]])
async def test_bill_with_no_readable_line_is_an_error(container, junk):
    asset, ccy = _cards()[0]

    got = await container.benchmark_lookup(
        {"line_items": junk}, {"asset_type": asset, "currency": ccy},
    )

    assert got["status"] == "error"
    assert "priced_items" not in got


# ── a single item ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("asset,ccy", _cards()[:2])
async def test_single_item_with_an_asset_is_priced_from_the_card(container, asset, ccy):
    line = _row_line(asset, ccy, 1.0)
    want, _ = boq_pricing.price_line_items([line], asset, ccy)

    got = await container.benchmark_lookup(
        {"item": line["description"], "unit": line["unit"]},
        {"asset_type": asset, "currency": ccy},
    )

    assert got["status"] == "success"
    assert got["unit_rate"] == want[0]["rate"]
    assert got["rate_source"] == want[0]["rate_source"]
    assert got["asset_type"] == asset and got["currency"] == ccy
    assert "rate card" in got["rate_basis"]


class _HistoricalStub:
    """Stands in for the historical_benchmark block so the fallback is
    exercised the same way under every block profile."""

    def __init__(self, rate: float):
        self.rate = rate
        self.calls: list[dict] = []

    async def process(self, data, params):
        self.calls.append(params)
        return {"status": "success", "item": params["item"],
                "rates": {"adjusted_usd": self.rate}}


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_arg,rate", [(None, 12.5), ("no such asset type", 40.0)])
async def test_single_item_without_a_card_asset_is_labelled_historical(
    container, monkeypatch, asset_arg, rate,
):
    stub = _HistoricalStub(rate)
    monkeypatch.setattr(container, "_get_historical_benchmark_block", lambda: stub)
    params = {"asset_type": asset_arg} if asset_arg else {}

    got = await container.benchmark_lookup({"item": "an item", "unit": "m3"}, params)

    assert got["status"] == "success"
    assert got["rates"]["adjusted_usd"] == rate
    assert len(stub.calls) == 1
    assert got["rate_basis"].startswith("historical benchmark book")
    assert got["rate_card_asset_types"] == boq_pricing.available_assets()
    assert "unit_rate" not in got


@pytest.mark.asyncio
@pytest.mark.parametrize("asset,ccy", _cards()[:2])
async def test_card_rate_is_used_before_the_historical_book(
    container, monkeypatch, asset, ccy,
):
    stub = _HistoricalStub(99.0)
    monkeypatch.setattr(container, "_get_historical_benchmark_block", lambda: stub)
    line = _row_line(asset, ccy, 1.0)

    got = await container.benchmark_lookup(
        {"item": line["description"], "unit": line["unit"]},
        {"asset_type": asset, "currency": ccy},
    )

    assert stub.calls == []
    assert got["asset_type"] == asset
