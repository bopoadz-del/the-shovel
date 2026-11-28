"""The planner template's production rates as catalogue entries: one
knowledge entry per work item (its unit and crew),
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
        "crew_days_from_rate_row",
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
    # The template states no exclusions; none are invented for it.
    assert res["not_covered"] == pb._EXCLUSIONS_NOT_STATED


@pytest.mark.parametrize("item,qty,unit", [
    ("blockwork walls", 30, "m3"),
    ("internal wall plaster", 50, "m"),
    ("epoxy paint floors", 10, ""),
])
def test_a_quantity_in_another_unit_is_refused_naming_both(item, qty, unit):
    with pytest.raises(ValueError) as err:
        pb.crew_days_from_rate_row(item, qty, unit)
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
        pb.crew_days_from_rate_row(item, 100, "m2")
    msg = str(err.value)
    assert "not a planner template row" in msg
    assert "blockwork walls" in msg and "ceiling painting common areas" in msg


@pytest.mark.parametrize("a,b,same", [
    ("m2", "M2", True), ("m²", "m2", True), (" m 2", "m2", True), ("sqm", "m2", True),
    ("ft2", "m2", False), ("m3", "m2", False), ("", "", False), ("m2", "", False),
    ("crew", "crew", False), (["m2"], "m2", False), (None, "m2", False),
])
def test_same_unit(a, b, same):
    assert pb._same_unit(a, b) is same


@pytest.mark.parametrize("index", range(len(pb.ROW_IDS)))
def test_every_row_is_one_knowledge_entry_with_a_catalogue_file(index):
    rows = pb.template_rows()
    assert len(rows) == len(pb.ROW_IDS) and len(rows) > 1
    for rid, (item, unit, crew, rate, _flag, not_covered) in [(pb.ROW_IDS[index], rows[index])]:
        path = os.path.join(_ROOT, "catalogue", "knowledge", rid + ".json")
        doc = json.load(open(path, encoding="utf-8"))
        assert doc["id"] == "knowledge:" + rid
        assert doc["hints"]["approved"] is False
        assert doc["facts"]["cites"]["entry"] == rid
        stated = doc["facts"]["described_as"]
        assert item in stated and unit in stated
        assert rate > 0 and crew and not_covered == ()


@pytest.mark.parametrize("item,rate", [("blockwork walls", 12.0), ("ceiling plaster", 20.0)])
def test_the_benchmark_and_the_duration_read_the_same_row(item, rate):
    bench = pb.productivity_benchmark(item)
    assert bench["rate_per_crew_day"] == rate
    assert pb.crew_days_from_rate_row(item, rate * 3, "m2")["duration_crew_days"] == pytest.approx(3.0)


@pytest.mark.parametrize("index", [0, 1, -1])
def test_a_named_row_missing_from_the_knowledge_base_raises(monkeypatch, index):
    from app.blocks import _knowledge

    real = _knowledge.shipped_entry
    gone = pb.ROW_IDS[index]
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


@pytest.mark.parametrize("item,qty,unit,want_days", [
    ("blockwork walls", 600, "sqm", 50.0),
    ("ceiling plaster", 400, "sq.m", 20.0),
])
def test_another_spelling_of_the_rows_unit_is_the_same_unit(item, qty, unit, want_days):
    out = _run(item, qty, unit)
    assert out["status"] == "success", out
    assert out["result"]["duration_crew_days"] == pytest.approx(want_days)


@pytest.mark.parametrize("args,named", [
    (("blockwork walls", float("inf"), "m2"), "finite"),
    (("ceiling plaster", float("-inf"), "m2"), "finite"),
    ((["blockwork walls"], 600, "m2"), "not ['blockwork walls']"),
    (("ceiling plaster", 600, ["m2"]), "['m2']"),
    (("blockwork walls", 600, "ft2"), "ft2"),
])
def test_degenerate_inputs_are_refused_naming_the_input(args, named):
    with pytest.raises(ValueError) as err:
        pb.crew_days_from_rate_row(*args)
    assert named in str(err.value)
    item, qty, unit = args
    out = _run(item, qty, unit)
    if any(isinstance(v, list) and len(v) == 1 for v in args):
        # The calculator's front door reads a one-entry list as its entry.
        assert out["status"] == "success", out
        assert out["result"]["duration_crew_days"] == pytest.approx(qty / pb.productivity_benchmark(
            item[0] if isinstance(item, list) else item)["rate_per_crew_day"])
    else:
        assert out["status"] != "success", out


@pytest.mark.parametrize("item", [["blockwork walls", "ceiling plaster"], ["a", "b"]])
def test_more_than_one_row_named_is_refused(item):
    out = _run(item, 600, "m2")
    assert out["status"] != "success", out
    assert "one planner template row" in str(out.get("error"))


def _shipped_ids_with_prefix(prefix):
    path = os.path.join(_ROOT, "app", "knowledge", "construction_kb.json")
    data = json.load(open(path, encoding="utf-8"))
    return {e["id"] for e in data["entries"] if str(e.get("id", "")).startswith(prefix)}


@pytest.mark.parametrize("prefix", ["productivity.", "productivity.blockwork"])
def test_every_template_row_in_the_knowledge_base_is_named_by_the_table(prefix):
    shipped = _shipped_ids_with_prefix(prefix)
    assert shipped, prefix
    assert shipped <= set(pb.ROW_IDS)
    assert len(pb.ROW_IDS) == len(set(pb.ROW_IDS))
    assert len(pb.template_rows()) == len(_shipped_ids_with_prefix("productivity."))


@pytest.mark.parametrize("stated,want", [
    ((), "not stated by the planner template"),
    (("supply only",), ["supply only"]),
])
def test_exclusions_are_the_sources_own_or_said_not_stated(monkeypatch, stated, want):
    rows = [(item, unit, crew, rate, flag, stated if i == 0 else ())
            for i, (item, unit, crew, rate, flag, _nc) in enumerate(pb.template_rows())]
    monkeypatch.setattr(pb, "template_rows", lambda: rows)
    out = pb.crew_days_from_rate_row(rows[0][0], 24, rows[0][1])
    assert out["not_covered"] == want


@pytest.mark.parametrize("item,written,unit", [
    # The value's own unit differs from quantity_unit: an area in sq ft, a
    # volume and a length against an m2 row. None of them is a figure.
    ("internal wall plaster", "2000 sq ft", "m2"),
    ("blockwork walls", "30 m3", "m2"),
    ("blockwork walls", "600 m", "m2"),
    # Even the row's own unit written in the value is asked for again: the
    # unit is read from quantity_unit only, never from the value.
    ("ceiling plaster", "600 m2", "m2"),
])
def test_a_unit_written_in_the_quantity_is_refused_not_dropped(item, written, unit):
    out = _run(item, written, unit)
    assert out["status"] != "success", out
    assert "duration_crew_days" not in (out.get("result") or {})
    text = str(out.get("error") or "") + str(out.get("question") or "")
    assert "quantity of the work item" in text, out


@pytest.mark.parametrize("item,bare,want_days", [
    ("internal wall plaster", "2000", 100.0),
    ("blockwork walls", "1,200", 100.0),
])
def test_a_bare_quantity_with_its_unit_beside_it_computes(item, bare, want_days):
    out = _run(item, bare, "m2")
    assert out["status"] == "success", out
    assert out["result"]["duration_crew_days"] == pytest.approx(want_days)
