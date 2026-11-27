"""D1: ONE production-rate table, read by the schedule path and by the
planning-hat calculator alike (app/lib/productivity_norms.py).

Every test builds its own rows or its own BOQ lines and computes the expected
day count from the row it read, so no shipped figure is pinned here.
"""
from __future__ import annotations

import math

import pytest

from app.agents.hats.planning.formulas import productivity_benchmarks
from app.agents.hats.planning.formulas.productivity_benchmarks import productivity_benchmark
from app.containers.construction import ConstructionContainer
from app.lib import productivity_norms
from app.lib.productivity_norms import norm_rate_for


def _row(work_item, keywords, rate, unit="m2", crew="crew of two"):
    return {"work_item": work_item, "unit": unit, "crew": crew, "rate": rate,
            "flag": "", "keywords": keywords}


# ---- one copy ---------------------------------------------------------------

@pytest.mark.parametrize("shared", ["benchmark_row", "TEMPLATE_PROVENANCE"])
def test_the_calculator_keeps_no_table_of_its_own(shared):
    """The hat module holds no rate list; what it uses is the shared object."""
    held = [k for k, v in vars(productivity_benchmarks).items()
            if not k.startswith("__") and isinstance(v, (list, dict)) and v]
    assert not held
    assert getattr(productivity_benchmarks, shared) is getattr(productivity_norms, shared)


@pytest.mark.parametrize("item,rate", [("widget laying", 17.0), ("gizmo fixing", 33.0)])
def test_the_calculator_answers_from_the_shared_table(monkeypatch, item, rate):
    monkeypatch.setattr(productivity_norms, "_TEMPLATE_RATES",
                        [_row(item, [item], rate)])
    out = productivity_benchmark(item.title())
    assert out["rate_per_crew_day"] == rate
    assert "unverified" in out["provenance"]


def test_every_template_row_carries_unit_crew_rate_and_keywords():
    for row in productivity_norms._TEMPLATE_RATES:
        assert str(row["unit"]).strip() and str(row["crew"]).strip()
        assert float(row["rate"]) > 0
        assert row["keywords"], row["work_item"]


# ---- the schedule path reads the table -----------------------------------

@pytest.mark.parametrize("desc,qty,rate", [
    ("Widget laying to level 2", 120, 40.0),
    ("Gizmo fixing, external", 90, 15.0),
])
def test_norm_rate_for_reads_the_template_and_says_so(monkeypatch, desc, qty, rate):
    monkeypatch.setattr(productivity_norms, "_TEMPLATE_RATES", [
        _row("widget laying", ["widget laying"], 40.0),
        _row("gizmo fixing", ["gizmo fixing"], 15.0),
    ])
    got = norm_rate_for(desc, "m2")
    assert got["rate"] == f"{rate:g} m2 per gang-day"
    assert got["tier"] == "unverified template"
    assert "unverified" in got["source"] and "10-hour" in got["source"]
    assert norm_rate_for(desc, "m3") is None      # unit mismatch
    assert norm_rate_for(desc, None) is None      # no unit


@pytest.mark.asyncio
@pytest.mark.parametrize("widget_rate,gizmo_rate,widget_qty,gizmo_qty", [
    (40.0, 15.0, 200, 45),
    (25.0, 60.0, 75, 600),
])
async def test_boq_rows_without_a_rate_get_a_kernel_duration_from_the_table(
        monkeypatch, widget_rate, gizmo_rate, widget_qty, gizmo_qty):
    monkeypatch.setattr(productivity_norms, "_TEMPLATE_RATES", [
        _row("widget laying", ["widget"], widget_rate),
        _row("gizmo fixing", ["gizmo"], gizmo_rate),
    ])
    r = await ConstructionContainer().generate_wbs(
        {"brief": "WBS from the BOQ scope", "boq_items": [
            {"item_key": "A1", "description": "Widget laying", "quantity": widget_qty, "unit": "m2"},
            {"item_key": "A2", "description": "Gizmo fixing", "quantity": gizmo_qty, "unit": "m2"},
            {"item_key": "A3", "description": "Sprocket polishing", "quantity": 45, "unit": "m2"},
        ]},
        {"target_count": 40},
    )
    acts = {a["boq"]["item_key"]: a for a in r["activities"] if a.get("boq")}
    assert acts["A1"]["duration_days"] == math.ceil(widget_qty / widget_rate)
    assert acts["A2"]["duration_days"] == math.ceil(gizmo_qty / gizmo_rate)
    assert "unverified" in acts["A1"]["duration_basis"]
    assert acts["A3"]["duration_days"] is None    # no row for this trade -> UNKNOWN


# ---- precedence and specificity -----------------------------------------

@pytest.mark.parametrize("cited_rate", ["25 m2 per gang-day", "80 m2 per gang-day"])
def test_a_cited_row_wins_over_the_template(monkeypatch, cited_rate):
    monkeypatch.setattr(productivity_norms, "_TEMPLATE_RATES",
                        [_row("widget laying", ["widget"], 40.0)])
    monkeypatch.setattr(productivity_norms, "_NORMS", [
        {"keywords": ["widget"], "rate": cited_rate, "unit": "m2",
         "citation": "TEST ONLY — Example Norms, 1st ed., tbl 3 (not shipped)"},
    ])
    got = norm_rate_for("Widget laying", "m2")
    assert got["rate"] == cited_rate and got["tier"] == "published"
    # The cited row is for m2 only; an m3 quantity falls to nothing, not to it.
    assert norm_rate_for("Widget laying", "m3") is None


@pytest.mark.parametrize("desc,want", [
    ("Ceramic floor tiling", "floor row"),
    ("Ceramic tiles to walls", "wall row"),
    ("Ceramic skirting", "plain row"),
])
def test_the_most_specific_keyword_wins(monkeypatch, desc, want):
    monkeypatch.setattr(productivity_norms, "_TEMPLATE_RATES", [
        _row("plain row", ["ceramic"], 10.0),
        _row("wall row", ["ceramic wall"], 20.0),
        _row("floor row", ["ceramic floor"], 30.0),
    ])
    got = norm_rate_for(desc, "m2")
    assert want in got["source"]


@pytest.mark.parametrize("desc", ["Surrender of keys", "Widgetry inspection"])
def test_keyword_words_match_only_whole_words(monkeypatch, desc):
    monkeypatch.setattr(productivity_norms, "_TEMPLATE_RATES", [
        _row("render", ["render"], 10.0), _row("widget", ["widget"], 10.0),
    ])
    assert norm_rate_for(desc, "m2") is None


# ---- helpers, directly ---------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("Blockwork (Walls)", "blockwork walls"),
    ("  Ceramic-tiles,  FLOORS ", "ceramic tiles floors"),
])
def test_normalise_work_item(text, want):
    assert productivity_norms.normalise_work_item(text) == want


@pytest.mark.parametrize("keyword", ["", "   "])
def test_an_empty_keyword_matches_nothing(keyword):
    assert productivity_norms._keyword_words_match(keyword, "widget laying") is False
    assert productivity_norms._keyword_words_match("widget", "widget laying") is True


@pytest.mark.parametrize("name", ["", "   "])
def test_an_empty_activity_name_gets_no_rate_and_is_refused_by_the_matcher(monkeypatch, name):
    monkeypatch.setattr(productivity_norms, "_TEMPLATE_RATES",
                        [_row("widget laying", ["widget"], 40.0)])
    assert norm_rate_for(name, "m2") is None
    with pytest.raises(ValueError):
        productivity_norms._best_match(productivity_norms._TEMPLATE_RATES, name, "m2")
