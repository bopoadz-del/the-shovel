"""historical_benchmark — multi-country, basis-honest indicative rate fallback.

The block is a day-one FALLBACK (not a priced rate book): indicative
general-knowledge rates × multi-country cost factors. Every response must state
each item's cost basis (material-only vs supply-and-fix vs plant+labour) and
flag when a location isn't in the index (so an unlisted country isn't silently
priced at the US baseline). cost_estimate must warn when priced lines mix bases.
"""

from __future__ import annotations

import pytest

from app.blocks.historical_benchmark import (
    HistoricalBenchmarkBlock, supported_locations, _BASE_RATES,
)
from app.containers.construction import ConstructionContainer
from tests.conftest import requires_construction_kit


@pytest.fixture
def block():
    return HistoricalBenchmarkBlock()


@pytest.fixture
def container():
    return ConstructionContainer()


class TestMultiCountry:
    def test_broad_country_coverage(self):
        locs = supported_locations()
        # genuinely multi-country, not Saudi/GCC only
        assert len(locs) >= 80
        for country in ("germany", "singapore", "nigeria", "brazil", "japan",
                        "india", "south africa", "canada", "saudi arabia"):
            assert country in locs

    @pytest.mark.asyncio
    async def test_country_factor_applied(self, block):
        # Germany (1.30) prices concrete higher than the US base (150).
        r = await block.process({}, {"action": "lookup", "item": "concrete",
                                     "location": "Germany"})
        assert r["status"] == "success"
        assert r["factors"]["location_matched"] is True
        assert r["rates"]["adjusted_usd"] > 150.0  # 150 × 1.30

    @pytest.mark.asyncio
    async def test_unknown_location_flagged_not_silent(self, block):
        r = await block.process({}, {"action": "lookup", "item": "concrete",
                                     "location": "Narnia"})
        assert r["status"] == "success"
        # priced at US base (1.0) but the fallback is DISCLOSED, not silent
        assert r["factors"]["location_matched"] is False
        assert r["rates"]["adjusted_usd"] == 150.0
        assert "not in the location index" in r["source_note"]

    @pytest.mark.asyncio
    async def test_list_locations_action(self, block):
        r = await block.process({}, {"action": "list_locations"})
        assert r["status"] == "success"
        assert r["count"] == len(supported_locations())
        assert "dubai" in r["locations"]


class TestBasisHonesty:
    def test_every_rate_declares_a_basis(self):
        for key, spec in _BASE_RATES.items():
            assert spec.get("basis") in {"material", "all_in", "plant_labour"}, key

    @pytest.mark.asyncio
    async def test_material_only_item_flagged(self, block):
        r = await block.process({}, {"action": "lookup", "item": "concrete"})
        assert r["basis"] == "material"  # ready-mix supply, no placing
        assert "supply only" in r["basis_note"]

    @pytest.mark.asyncio
    async def test_supply_and_fix_item_flagged(self, block):
        r = await block.process({}, {"action": "lookup", "item": "tiles"})
        assert r["basis"] == "all_in"  # incl. fixing


class TestCostEstimateBasisWarning:
    # generate_cost_estimate delegates to the historical_benchmark block, which
    # is only registered when the construction kit is loaded (production-like);
    # in virgin mode it honestly errors, so gate these on the kit.
    @requires_construction_kit
    @pytest.mark.asyncio
    async def test_mixed_basis_estimate_is_flagged(self, container):
        r = await container.generate_cost_estimate({}, {
            "quantities": {
                "concrete": {"quantity": 500, "unit": "m3"},   # material-only
                "tiles": {"quantity": 300, "unit": "m2"},      # all-in
            },
            "location": "Dubai", "project_type": "commercial"})
        assert r["status"] == "success"
        assert r["mixed_basis"] is True
        assert set(r["bases_used"]) == {"material", "all_in"}
        assert r["basis_warning"] and "understates" in r["basis_warning"]
        assert r["rate_source"] == "indicative_benchmark_fallback"

    @requires_construction_kit
    @pytest.mark.asyncio
    async def test_single_basis_estimate_no_warning(self, container):
        r = await container.generate_cost_estimate({}, {
            "quantities": {
                "tiles": {"quantity": 300, "unit": "m2"},
                "formwork": {"quantity": 800, "unit": "m2"},
            },
            "location": "Riyadh", "project_type": "commercial"})
        assert r["status"] == "success"
        assert r["mixed_basis"] is False
        assert r["basis_warning"] is None
