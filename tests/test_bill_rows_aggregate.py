"""Count and sum over the derived bill sheet (app.core.bill_aggregate).

Every bill here is invented in the test and written through the same
``write_bill_workbook`` the derivation uses, so the reader is exercised on the
real sheet layout (banner path row, canonical columns, extras, Check, Source
Page). Each decision is driven by at least two inputs with different outcomes.
"""

from __future__ import annotations

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

#: Bill one: quantities. Nine trench rows mention "backfill"; one of them is
#: split across pages 3 and 4 (description on 3, figures on 4); one in another
#: section mentions it too and must stay out of a section-scoped sum.
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


@pytest.fixture
def sheet(tmp_path):
    def build(rows):
        path = tmp_path / "bill.xlsx"
        g.write_bill_workbook(rows, str(path))
        return agg.read_sheet_rows(str(path))
    return build


def test_reader_keeps_every_row_with_path_ref_and_page(sheet):
    rows = sheet(BILL_ONE)
    assert len(rows) == len(BILL_ONE)
    assert rows[0].path == GROUND and rows[0].ref == "T1" and rows[0].pages == [2]
    assert rows[-1].path == OTHER and rows[-1].pages == [5]
    assert rows[4].ref == "" and rows[4].quantity == "2.5"


def test_split_item_joins_across_the_page_break(sheet):
    items = agg.join_page_continuations(sheet(BILL_ONE))
    assert len(items) == len(BILL_ONE) - 1
    t4 = next(r for r in items if r.ref == "T4")
    assert t4.pages == [3, 4]
    assert t4.quantity == "2.5"
    assert "compacted" in t4.description and "imported" in t4.description


def test_figureless_row_on_the_same_page_is_not_a_continuation(sheet):
    items = agg.join_page_continuations(sheet(BILL_TWO))
    assert len(items) == len(BILL_TWO)
    assert any(r.description.startswith("Gloss") for r in items)


def test_row_with_a_reference_on_the_next_page_starts_a_new_item(sheet):
    rows = sheet([
        _row(GROUND, 1, "A1", "Hardcore bed", "m3", "", "", ""),
        _row(GROUND, 2, "A2", "Blinding layer", "m3", "3", "1", "3"),
    ])
    assert len(agg.join_page_continuations(rows)) == 2


def test_section_sum_counts_the_split_item_and_skips_other_sections(sheet):
    out = agg.aggregate(sheet(BILL_ONE), terms="backfill", section="trenches")
    refs = [r["ref"] for r in out["rows"]]
    assert refs == ["T1", "T2", "T4", "T5", "T6"]
    assert "P1" not in refs and "T3" not in refs
    assert out["count"] == 5
    assert out["sum"] == pytest.approx(10 + 4.5 + 2.5 + 1250)
    assert out["unit"] == "m3"
    assert [w["ref"] for w in out["without_figure"]] == ["T6"]
    t4 = next(r for r in out["rows"] if r["ref"] == "T4")
    assert t4["pages"] == [3, 4] and t4["quantity"] == 2.5


def test_without_section_the_other_section_row_is_included(sheet):
    out = agg.aggregate(sheet(BILL_ONE), terms="backfill")
    assert out["count"] == 6
    assert out["sum"] == pytest.approx(10 + 4.5 + 2.5 + 1250 + 100)


def test_every_term_must_appear_as_a_word(sheet):
    rows = sheet(BILL_ONE)
    assert agg.aggregate(rows, terms="backfill granular")["count"] == 2
    assert agg.aggregate(rows, terms="backfill gravel")["count"] == 0
    # "fill" is a word of T2 ("granular fill"), not a substring match of "backfill".
    assert [r["ref"] for r in agg.aggregate(rows, terms="fill")["rows"]] == ["T2"]


def test_amount_sum_and_mixed_unit_quantities(sheet):
    rows = sheet(BILL_TWO)
    amount = agg.aggregate(rows, terms="emulsion paint", measure="amount")
    assert amount["count"] == 3 and amount["sum"] == pytest.approx(1650)
    walls = agg.aggregate(rows, terms="paint", section="walls")
    assert walls["count"] == 3
    assert walls["sum"] is None and walls["unit"] is None
    assert walls["sum_by_unit"] == {"m2": 300, "m": 80}
    assert [w["description"] for w in walls["without_figure"]] == ["Gloss paint to door frames"]
    ceilings = agg.aggregate(rows, terms="paint", section="ceilings")
    assert ceilings["sum"] == 50 and ceilings["unit"] == "m2"


def test_refuses_an_empty_filter_and_an_unknown_measure(sheet):
    rows = sheet(BILL_TWO)
    with pytest.raises(ValueError):
        agg.aggregate(rows, terms="", section="")
    with pytest.raises(ValueError):
        agg.aggregate(rows, terms="paint", measure="rate")


def test_project_scope_reads_each_derived_sheet_separately(tmp_path, monkeypatch):
    from app.core import bill_derived, projects

    paths = {}
    for name, rows in (("one", BILL_ONE), ("two", BILL_TWO)):
        paths[name] = str(tmp_path / f"{name}.xlsx")
        g.write_bill_workbook(rows, paths[name])
    docs = [
        {"id": "d1", "original_name": "one.bill.xlsx", "path": paths["one"],
         "metadata": {"derivation": bill_derived.DERIVATION, "derived_from": "s1"}},
        {"id": "d2", "original_name": "two.bill.xlsx", "path": paths["two"],
         "metadata": {"derivation": bill_derived.DERIVATION, "derived_from": "s2"}},
        {"id": "s1", "original_name": "one.pdf", "path": "", "metadata": {}},
        {"id": "old", "original_name": "old.bill.xlsx", "path": paths["two"],
         "retrieval_visible": False,
         "metadata": {"derivation": bill_derived.DERIVATION, "derived_from": "s1"}},
    ]
    monkeypatch.setattr(projects, "list_documents", lambda pid: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])

    every = agg.aggregate_project_bills("p", terms="paint")
    assert [b["document_id"] for b in every["bills"]] == ["d1", "d2"]
    assert [b["count"] for b in every["bills"]] == [0, 4]

    by_source = agg.aggregate_project_bills("p", terms="backfill", document_id="s1")
    assert [b["document_id"] for b in by_source["bills"]] == ["d1"]
    assert by_source["bills"][0]["count"] == 6

    none = agg.aggregate_project_bills("p", terms="paint", document_id="s9")
    assert none == {"status": "error", "reason": agg.NO_DERIVED_SHEET}


def test_tool_is_advertised_and_registered():
    import json
    from pathlib import Path

    from app.agents.base import tools as base_tools  # noqa: F401 -- registers
    from app.agents.core import tool_registry

    manifest = json.loads(Path("app/agents/manifests/fork.base.json").read_text(encoding="utf-8"))
    assert "bill_rows_aggregate" in manifest["available_tools"]
    tool_registry.load()
    assert "bill_rows_aggregate" in tool_registry.tools_of("base")
    assert tool_registry.get("bill_rows_aggregate") is not None
