"""construction_catalog — keyword-driven KSA materials/labour/equipment classifier
+ the GK-knowledge export. Rates live in RAG (the generated note); this module is
a deterministic classify/lookup tool. The project's priced BOQ still wins.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from app.lib import construction_catalog as cat

_NOTE = Path(__file__).resolve().parents[1] / "docs" / "knowledge" / \
    "construction_catalog_rates_ksa_2025.md"


@pytest.fixture
def catalog():
    return cat.UnifiedCatalog()


def test_classifies_material_labour_equipment(catalog):
    assert catalog.lookup("aluminium curtain wall cladding").group_type == "material"
    assert catalog.lookup("certified welder for structural steel").group_type == "labour"
    assert catalog.lookup("20-ton hydraulic excavator with breaker").group_type == "equipment"


def test_lookup_returns_rates_and_source(catalog):
    r = catalog.lookup("20-ton hydraulic excavator with breaker")
    assert r.matched_group == "excavator"
    assert r.rate_mid > 0
    assert r.source  # cited
    assert r.confidence in ("high", "medium", "low")


def test_unknown_item_returns_none(catalog):
    # Honest: no fabricated match/rate for gibberish.
    assert catalog.lookup("zzz nonsense qwerty item") is None


def test_bulk_classify_mixed_list(catalog):
    items = [
        "aluminium curtain wall cladding panels",       # material
        "20-ton hydraulic excavator dry hire",          # equipment
        "certified structural steel welder",            # labour
    ]
    results = catalog.classify_items(items)
    assert len(results) == 3
    types = {r.group_type for r in results if r is not None}
    assert types == {"material", "equipment", "labour"}


def test_estimate_cost_totals(catalog):
    est = catalog.estimate_cost("ready mix concrete cast in situ", quantity=100, spec_level="mid")
    assert est["total_sar"] == pytest.approx(est["rate_sar"] * 100, rel=1e-6)


# ── GK knowledge export ─────────────────────────────────────────────────────

def test_knowledge_markdown_has_all_three_rate_tables():
    md = cat.to_knowledge_markdown()
    assert "## Material supply rates" in md
    assert "## Labour rates" in md
    assert "## Equipment dry-hire rates" in md
    # Sourced + indicative (doctrine): rates cite a source and are flagged indicative.
    assert "Source" in md
    assert "indicative" in md.lower()
    assert "priced BOQ" in md


def test_committed_note_matches_generator():
    # Single source of truth: the committed GK note must equal the generator's
    # output. If catalog rates change, regenerate the note (this test fails until
    # you do), so RAG never drifts from the library.
    assert _NOTE.exists(), f"missing generated note: {_NOTE}"
    on_disk = io.open(_NOTE, encoding="utf-8").read()
    generated = cat.to_knowledge_markdown()
    if not generated.endswith("\n"):
        generated += "\n"
    assert on_disk == generated, (
        "docs/knowledge/construction_catalog_rates_ksa_2025.md is stale — "
        "regenerate it from construction_catalog.to_knowledge_markdown()."
    )
