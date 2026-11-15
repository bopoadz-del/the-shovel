"""Count and sum over the derived bill sheet (app.core.bill_aggregate).

Every bill here is invented in the test and written through the same
``write_bill_workbook`` the derivation uses, so the reader is exercised on the
real sheet layout (banner path row, canonical columns, extras, Check, Source
Page). Each decision is driven by at least two inputs with different outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g


def _row(path, page, ref, desc, unit="", qty="", rate="", amount=""):
    fields = {"description": desc}
    for k, v in (("unit", unit), ("quantity", qty), ("rate", rate), ("total", amount)):
        if v:
            fields[k] = v
    return g.BillRow(section=path[-1], source_page=page, fields=fields,
                     extra={"Item": ref} if ref else {}, path=tuple(path))


GROUND = ("Groundworks", "Trenches")
OTHER = ("Groundworks", "Pits")

#: Bill one: quantities. Six items mention "backfill", five of them in the
#: trench section; one trench item is split across pages 3 and 4 (description
#: on 3, figures on 4); one trench item prints a rate but no quantity.
BILL_ONE = [
    _row(GROUND, 2, "T1", "Excavate trench backfill with selected material", "m3", "10", "5", "50"),
    _row(GROUND, 2, "T2", "Backfill around pipe with granular fill", "m3", "4.5", "8", "36"),
    _row(GROUND, 2, "T3", "Dispose of surplus spoil off site", "m3", "20", "3", "60"),
    _row(GROUND, 3, "T4", "Backfill trench with imported", "m3", "", "", ""),
    _row(GROUND, 4, "", "granular material, compacted in layers", "", "2.5", "12", "30"),
    _row(GROUND, 4, "T5", "Backfill to manhole surround", "m3", "1,250.00", "2", "2500"),
    _row(GROUND, 4, "T6", "Backfill, item rate only", "m3", "", "7", ""),
    _row(OTHER, 5, "P1", "Backfill to pit with excavated material", "m3", "100", "4", "400"),
]

#: Bill two: amounts, mixed units, a figureless row on the same page as the
#: row above it (a real row, not a continuation).
BILL_TWO = [
    _row(("Finishes", "Walls"), 7, "F1", "Emulsion paint to plaster walls", "m2", "300", "4", "1200"),
    _row(("Finishes", "Walls"), 7, "F2", "Emulsion paint to skirting", "m", "80", "2.5", "200"),
    _row(("Finishes", "Walls"), 7, "", "Gloss paint to door frames", "nr", "", "", ""),
    _row(("Finishes", "Ceilings"), 8, "F3", "Emulsion paint to soffits", "m2", "50", "5", "250"),
]


def _sheet(tmp_path, rows):
    path = tmp_path / "bill.xlsx"
    g.write_bill_workbook(rows, str(path))
    return agg.read_sheet_rows(str(path))


@pytest.mark.parametrize("bill", [BILL_ONE, BILL_TWO], ids=["one", "two"])
def test_reader_keeps_every_row_with_path_ref_and_page(tmp_path, bill):
    rows = _sheet(tmp_path, bill)
    assert [r.pages for r in rows] == [[b.source_page] for b in bill]
    assert [r.path for r in rows] == [b.path for b in bill]
    assert [r.ref for r in rows] == [b.extra.get("Item", "") for b in bill]
    assert [r.quantity for r in rows] == [b.fields.get("quantity", "") for b in bill]


@pytest.mark.parametrize("rows, expected", [
    # description at the foot of page 3, figures at the head of page 4: one item
    (BILL_ONE, [[2], [2], [2], [3, 4], [4], [4], [5]]),
    # a figureless row without a reference on the SAME page is its own row
    (BILL_TWO, [[7], [7], [7], [8]]),
    # a reference on the next page starts a new item
    ([_row(GROUND, 1, "A1", "Hardcore bed", "m3"),
      _row(GROUND, 2, "A2", "Blinding layer", "m3", "3", "1", "3")], [[1], [2]]),
    # both rows carry figures: two items, even without a reference
    ([_row(GROUND, 1, "A1", "Hardcore bed", "m3", "2", "1", "2"),
      _row(GROUND, 2, "", "Blinding layer", "m3", "3", "1", "3")], [[1], [2]]),
    # the next page's row belongs to another heading path
    ([_row(GROUND, 1, "A1", "Hardcore bed", "m3"),
      _row(OTHER, 2, "", "Blinding layer", "m3", "3", "1", "3")], [[1], [2]]),
    # a page skipped between them: not a continuation
    ([_row(GROUND, 1, "A1", "Hardcore bed", "m3"),
      _row(GROUND, 3, "", "Blinding layer", "m3", "3", "1", "3")], [[1], [3]]),
], ids=["split", "same-page", "next-ref", "both-figures", "other-path", "page-gap"])
def test_page_continuations(tmp_path, rows, expected):
    items = agg.join_page_continuations(_sheet(tmp_path, rows))
    assert [r.pages for r in items] == expected


@pytest.mark.parametrize("ref, figures, starts", [
    ("T4", ("2.5", "12", "30", "m3"), "Backfill trench with imported granular"),
    ("T5", ("1,250.00", "2", "2500", "m3"), "Backfill to manhole surround"),
])
def test_joined_item_keeps_both_halves(tmp_path, ref, figures, starts):
    item = next(r for r in agg.join_page_continuations(_sheet(tmp_path, BILL_ONE)) if r.ref == ref)
    assert (item.quantity, item.rate, item.amount, item.unit) == figures
    assert item.description.startswith(starts)


@pytest.mark.parametrize("bill, kwargs, refs, total, unit, without", [
    # section-scoped: the split item counts, the other section's row does not
    (BILL_ONE, {"terms": "backfill", "section": "trenches"},
     ["T1", "T2", "T4", "T5", "T6"], 10 + 4.5 + 2.5 + 1250, "m3", ["T6"]),
    # no section: the other section's row is included
    (BILL_ONE, {"terms": "backfill"},
     ["T1", "T2", "T4", "T5", "T6", "P1"], 10 + 4.5 + 2.5 + 1250 + 100, "m3", ["T6"]),
    # every term must appear (one of them only after the page join)
    (BILL_ONE, {"terms": "backfill granular"}, ["T2", "T4"], 7.0, "m3", []),
    (BILL_ONE, {"terms": "backfill gravel"}, [], 0.0, "", []),
    # a word, not a substring: "fill" is not "backfill"
    (BILL_ONE, {"terms": ["fill"]}, ["T2"], 4.5, "m3", []),
    # section only
    (BILL_ONE, {"terms": "", "section": "pits"}, ["P1"], 100.0, "m3", []),
    # amounts
    (BILL_TWO, {"terms": "emulsion paint", "measure": "amount"},
     ["F1", "F2", "F3"], 1650.0, None, []),
    (BILL_ONE, {"terms": "backfill", "section": "trenches", "measure": "amount"},
     ["T1", "T2", "T4", "T5", "T6"], 50 + 36 + 30 + 2500, None, ["T6"]),
    # one unit in the section: one sum
    (BILL_TWO, {"terms": "paint", "section": "ceilings"}, ["F3"], 50.0, "m2", []),
], ids=["section", "no-section", "two-terms", "no-match", "word", "section-only",
        "amount-two", "amount-one", "one-unit"])
def test_aggregate(tmp_path, bill, kwargs, refs, total, unit, without):
    out = agg.aggregate(_sheet(tmp_path, bill), **kwargs)
    assert [r["ref"] for r in out["rows"]] == refs
    assert out["count"] == len(refs)
    assert out["sum"] == pytest.approx(total)
    assert out["unit"] == unit
    assert [w["ref"] for w in out["without_figure"]] == without
    assert out["rows_scanned"] == len(agg.join_page_continuations(_sheet(tmp_path, bill)))


@pytest.mark.parametrize("section, by_unit", [
    ("walls", {"m2": 300, "m": 80}),
    ("finishes", {"m2": 350, "m": 80}),
])
def test_mixed_units_give_no_single_sum(tmp_path, section, by_unit):
    out = agg.aggregate(_sheet(tmp_path, BILL_TWO), terms="paint", section=section)
    assert out["sum"] is None and out["unit"] is None
    assert out["sum_by_unit"] == by_unit
    assert [w["description"] for w in out["without_figure"]] == ["Gloss paint to door frames"]


@pytest.mark.parametrize("kwargs", [
    {"terms": "", "section": ""},
    {"terms": [], "section": None},
    {"terms": "paint", "measure": "rate"},
])
def test_refuses_an_empty_filter_and_an_unknown_measure(tmp_path, kwargs):
    with pytest.raises(ValueError):
        agg.aggregate(_sheet(tmp_path, BILL_TWO), **kwargs)


@pytest.fixture
def two_bills(tmp_path, monkeypatch):
    from app.core import bill_derived, projects

    paths = {}
    for name, rows in (("one", BILL_ONE), ("two", BILL_TWO)):
        paths[name] = str(tmp_path / f"{name}.xlsx")
        g.write_bill_workbook(rows, paths[name])
    derived = {"derivation": bill_derived.DERIVATION}
    docs = [
        {"id": "d1", "original_name": "one.bill.xlsx", "path": paths["one"],
         "metadata": {**derived, "derived_from": "s1"}},
        {"id": "d2", "original_name": "two.bill.xlsx", "path": paths["two"],
         "metadata": {**derived, "derived_from": "s2"}},
        {"id": "s1", "original_name": "one.pdf", "path": "", "metadata": {}},
        {"id": "old", "original_name": "old.bill.xlsx", "path": paths["two"],
         "retrieval_visible": False, "metadata": {**derived, "derived_from": "s1"}},
        {"id": "gone", "original_name": "gone.bill.xlsx", "path": "",
         "metadata": {**derived, "derived_from": "s3"}},
    ]
    monkeypatch.setattr(projects, "list_documents", lambda pid: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])


@pytest.mark.parametrize("document_id, terms, expected", [
    (None, "paint", [("d1", 0), ("d2", 4), ("gone", None)]),
    ("s1", "backfill", [("d1", 6)]),
    ("d2", "emulsion", [("d2", 3)]),
    ("s3", "paint", [("gone", None)]),
], ids=["every", "by-source", "by-derived", "no-bytes"])
def test_project_scope_reads_each_derived_sheet_separately(two_bills, document_id, terms, expected):
    out = agg.aggregate_project_bills("p", terms=terms, document_id=document_id)
    assert out["status"] == "success"
    assert [(b["document_id"], b.get("count")) for b in out["bills"]] == expected
    for b in out["bills"]:
        assert b["status"] == ("success" if b.get("count") is not None else "error")


@pytest.mark.parametrize("document_id", ["s9", "s2-missing"])
def test_project_scope_without_a_derived_sheet(two_bills, document_id):
    out = agg.aggregate_project_bills("p", terms="paint", document_id=document_id)
    assert out == {"status": "error", "reason": agg.NO_DERIVED_SHEET}


def _tool_call(args):
    return {"function": {"name": "bill_rows_aggregate", "arguments": json.dumps(args)}}


@pytest.fixture
def runtime():
    from app.agents.runtime import Agent
    return Agent(name="test-agent", description="t", system_prompt="t", allowed_blocks=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("project_id, args, ok, field, value", [
    (None, {"terms": "paint"}, False, "error", "no project in scope"),
    ("p", {"terms": "backfill", "section": "trenches", "document_id": "s1"}, True, "sum", 1267.0),
    ("p", {"terms": "emulsion", "measure": "AMOUNT", "document_id": "d2"}, True, "sum", 1650.0),
    ("p", {"terms": "paint", "measure": "rate"}, False, "error",
     "measure must be one of ('quantity', 'amount')"),
    ("p", {"terms": "paint", "document_id": "s9"}, False, "reason", agg.NO_DERIVED_SHEET),
], ids=["no-project", "quantity", "amount", "bad-measure", "no-sheet"])
async def test_tool_handler(runtime, two_bills, project_id, args, ok, field, value):
    from app.agents.base.tools import handle_bill_rows_aggregate
    from app.agents.core.tool_registry import ToolCall

    direct = await handle_bill_rows_aggregate(ToolCall(
        agent=runtime, name="bill_rows_aggregate", args=args, project_id=project_id,
    ))
    out = await runtime._run_tool_call(_tool_call(args), project_id=project_id)
    assert out == direct
    assert out["ok"] is ok
    result = out["result"]
    got = result["bills"][0][field] if "bills" in result else result[field]
    assert got == value


@pytest.mark.parametrize("project_id, advertised", [("p", True), (None, False)])
def test_tool_is_advertised_only_in_a_project(runtime, project_id, advertised):
    from app.agents.core.tools import tool_definitions

    names = [t["function"]["name"] for t in tool_definitions(runtime, project_id=project_id)]
    # The bill tool joins the other project-scoped document tools.
    for scoped in ("search_project_documents", "list_project_documents",
                   "fetch_document", "bill_rows_aggregate"):
        assert (scoped in names) is advertised, scoped


def test_tool_is_in_the_base_manifest_and_registry():
    from app.agents.core import tool_registry

    manifest = json.loads(Path("app/agents/manifests/fork.base.json").read_text(encoding="utf-8"))
    assert "bill_rows_aggregate" in manifest["available_tools"]
    tool_registry.load()
    assert "bill_rows_aggregate" in tool_registry.tools_of("base")


@pytest.mark.parametrize("cell, expected", [
    (7, 7), ("12", 12), (3.0, 3), ("4.0", 4),          # a recorded page
    (None, None), ("", None), ("p7", None), ("7-8", None), ("n/a", None),  # no page recorded
])
def test_source_page_cell_is_read_without_guessing(cell, expected):
    assert agg._page(cell) == expected
