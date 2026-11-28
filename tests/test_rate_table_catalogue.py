"""The planner template's production rates as catalogue entries: one
knowledge entry per work item (its unit, crew and what it does not cover),
the figures held once in the shipped knowledge base, and a duration
calculator that takes the row the caller names and checks the quantity's
unit against that row's. No row is chosen from a description."""
from __future__ import annotations

import json
import os

import pytest

from app.agents.hats.planning.formulas import productivity_benchmarks as pb
from app.lib import construction_formulas as cf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(work_item, quantity, unit):
    return cf.run_calculation(
        "production_duration",
        {"work_item": work_item, "quantity": quantity, "quantity_unit": unit})


@pytest.mark.parametrize("item,qty,unit,want_days,want_crew", [
    ("blockwork walls", 600, "m2", 50.0, "mason + helper (1:1)"),
    ("Internal wall plaster", 1200, "M2", 60.0, "mason + helper"),
    ("ceramic tiles floors", 900, "m²", 22.5, "mason (tiler)"),
])
def test_duration_is_quantity_over_the_named_rows_rate(item, qty, unit, want_days, want_crew):
    out = _run(item, qty, unit)
    assert out["status"] == "success", out
    res = out["result"]
    assert res["duration_crew_days"] == pytest.approx(want_days)
    assert res["crew"] == want_crew
    assert res["unit"] == "m2"
    assert "unverified" in res["provenance"]
    assert any("demolition" in line for line in res["not_covered"])


@pytest.mark.parametrize("item,qty,unit", [
    ("blockwork walls", 30, "m3"),
    ("internal wall plaster", 50, "m"),
    ("epoxy paint floors", 10, ""),
])
def test_a_quantity_in_another_unit_is_refused_naming_both(item, qty, unit):
    with pytest.raises(ValueError) as err:
        pb.production_duration(item, qty, unit)
    msg = str(err.value)
    assert item in msg and "m2" in msg
    assert (unit or "no unit") in msg
    out = _run(item, qty, unit or "none")
    assert out["status"] != "success", out


@pytest.mark.parametrize("item", [
    "Demolish existing blockwork walls",
    "Remove wall tiles",
    "blockwork",
])
def test_a_name_that_is_not_a_row_is_refused_with_the_rows_listed(item):
    with pytest.raises(ValueError) as err:
        pb.production_duration(item, 100, "m2")
    msg = str(err.value)
    assert "not a planner template row" in msg
    assert "blockwork walls" in msg and "ceiling painting common areas" in msg


@pytest.mark.parametrize("a,b,same", [
    ("m2", "M2", True), ("m²", "m2", True), (" m 2", "m2", True),
    ("m3", "m2", False), ("", "", False), ("m2", "", False),
])
def test_same_unit(a, b, same):
    assert pb._same_unit(a, b) is same


def test_every_row_is_one_knowledge_entry_with_a_catalogue_file():
    rows = pb.template_rows()
    assert len(rows) == len(pb.ROW_IDS) and len(rows) > 1
    for rid, (item, unit, crew, rate, _flag, not_covered) in zip(pb.ROW_IDS, rows):
        path = os.path.join(_ROOT, "catalogue", "knowledge", rid + ".json")
        doc = json.load(open(path, encoding="utf-8"))
        assert doc["id"] == "knowledge:" + rid
        assert doc["hints"]["approved"] is False
        assert doc["facts"]["cites"]["entry"] == rid
        stated = doc["facts"]["described_as"]
        assert item in stated and unit in stated
        assert rate > 0 and crew and not_covered


@pytest.mark.parametrize("item,rate", [("blockwork walls", 12.0), ("ceiling plaster", 20.0)])
def test_the_benchmark_and_the_duration_read_the_same_row(item, rate):
    bench = pb.productivity_benchmark(item)
    assert bench["rate_per_crew_day"] == rate
    assert pb.production_duration(item, rate * 3, "m2")["duration_crew_days"] == pytest.approx(3.0)


def test_a_named_row_missing_from_the_knowledge_base_raises(monkeypatch):
    from app.blocks import _knowledge

    real = _knowledge.shipped_entry
    gone = pb.ROW_IDS[1]
    monkeypatch.setattr(_knowledge, "shipped_entry",
                        lambda rid: None if rid == gone else real(rid))
    with pytest.raises(LookupError, match=gone):
        pb.template_rows()
    monkeypatch.setattr(_knowledge, "shipped_entry", real)
    assert len(pb.template_rows()) == len(pb.ROW_IDS)


@pytest.mark.parametrize("qty,item,crews,want_days", [
    (600, "blockwork walls", 5, 10.0),
    (1800, "acrylic emulsion ceilings", 4, 10.0),
])
def test_chain_bill_quantity_to_row_to_programme_days(qty, item, crews, want_days):
    """A bill quantity, the row named, then the crew-days spread over crews."""
    first = _run(item, qty, "m2")
    assert first["status"] == "success", first
    crew_days = first["result"]["duration_crew_days"]
    second = cf.run_calculation("arithmetic_step", {
        "operator": "divide", "a": crew_days, "b": crews})
    assert second["status"] == "success", second
    assert float(second["result"]["result"]) == pytest.approx(want_days)
