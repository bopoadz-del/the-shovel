"""S-K3 item 4: BIM take-off grouped by storey x IFC class, with the
column mapping taken from the caller's table.

Every test builds its own elements (plain dicts, or an IFC written with
ifcopenshell.api); no column, field, class or storey name is read by
product code.
"""
from __future__ import annotations

import asyncio

import pytest

from app.blocks.bim_extractor import group_quantities


def _row(result, **key):
    rows = [
        r for r in result["groups"]
        if all(r.get(k) == v for k, v in key.items())
    ]
    assert len(rows) == 1, (key, result["groups"])
    return rows[0]


# ---- grouping keys ---------------------------------------------------------

@pytest.mark.parametrize(
    "elements, expected",
    [
        (
            [
                {"ifc_type": "IfcWall", "storey": "L1", "v": 2.0},
                {"ifc_type": "IfcWall", "storey": "L1", "v": 3.5},
                {"ifc_type": "IfcWall", "storey": "L2", "v": 1.0},
                {"ifc_type": "IfcSlab", "storey": "L1", "v": 10.0},
            ],
            {("L1", "IfcWall"): (2, 5.5), ("L1", "IfcSlab"): (1, 10.0),
             ("L2", "IfcWall"): (1, 1.0)},
        ),
        (
            [
                {"ifc_type": "IfcBeam", "storey": "Roof", "v": 0.25},
                {"ifc_type": "IfcColumn", "storey": "Roof", "v": 0.5},
                {"ifc_type": "IfcColumn", "storey": "Roof", "v": 0.75},
            ],
            {("Roof", "IfcBeam"): (1, 0.25), ("Roof", "IfcColumn"): (2, 1.25)},
        ),
    ],
)
def test_groups_by_storey_and_class(elements, expected):
    result = group_quantities(elements)
    assert result["group_by"] == ["storey", "ifc_type"]
    assert result["elements_grouped"] == len(elements)
    got = {
        (r["storey"], r["ifc_type"]): (r["count"], r["columns"]["v"]["total"])
        for r in result["groups"]
    }
    assert got == expected


@pytest.mark.parametrize(
    "elements, expected_order",
    [
        (
            [{"ifc_type": "IfcWall"}, {"ifc_type": "IfcWall", "storey": "B"},
             {"ifc_type": "IfcWall", "storey": "A"}],
            ["A", "B", None],
        ),
        (
            [{"ifc_type": "IfcSlab", "storey": "Z"}, {"ifc_type": "IfcSlab"}],
            ["Z", None],
        ),
    ],
)
def test_element_without_storey_is_its_own_group_last(elements, expected_order):
    result = group_quantities(elements)
    assert [r["storey"] for r in result["groups"]] == expected_order
    assert _row(result, storey=None)["count"] == sum(
        1 for e in elements if "storey" not in e
    )


@pytest.mark.parametrize(
    "group_by, expected",
    [
        (["category"], {("walls",): 2, ("slabs",): 1}),
        (["Pset_X.Zone"], {("N",): 2, (None,): 1}),
    ],
)
def test_group_by_is_the_callers_keys(group_by, expected):
    elements = [
        {"category": "walls", "properties": {"Pset_X.Zone": "N"}},
        {"category": "walls", "properties": {"Pset_X.Zone": "N"}},
        {"category": "slabs"},
    ]
    result = group_quantities(elements, group_by=group_by)
    assert result["group_by"] == group_by
    got = {tuple(r[k] for k in group_by): r["count"] for r in result["groups"]}
    assert got == expected


# ---- column mapping --------------------------------------------------------

@pytest.mark.parametrize(
    "table, total, frm, missing",
    [
        # net first: falls back to gross only where net is absent
        ([{"column": "Vol", "fields": ["net", "gross"], "unit": "m3"}],
         2.0 + 7.0, {"net": 1, "gross": 1}, 1),
        # gross first: gross wherever present
        ([{"column": "Vol", "fields": ["gross", "net"], "unit": "m3"}],
         3.0 + 7.0, {"gross": 2}, 1),
    ],
)
def test_column_reads_fields_in_table_order(table, total, frm, missing):
    elements = [
        {"ifc_type": "IfcWall", "storey": "L1", "net": 2.0, "gross": 3.0},
        {"ifc_type": "IfcWall", "storey": "L1", "gross": 7.0},
        {"ifc_type": "IfcWall", "storey": "L1", "name": "no figures"},
    ]
    result = group_quantities(elements, table)
    cell = _row(result, storey="L1")["columns"]["Vol"]
    assert cell["total"] == pytest.approx(total)
    assert cell["from"] == frm
    assert cell["counted"] == 3 - missing
    assert cell["missing"] == missing
    assert cell["unit"] == "m3"
    assert result["columns"] == ["Vol"]


@pytest.mark.parametrize(
    "field, value",
    [("Qto_Base.NetArea", 4.5), ("Pset_Y.Len", 12.0)],
)
def test_column_reads_flattened_properties(field, value):
    elements = [{"ifc_type": "IfcSlab", "storey": "L1",
                 "properties": {field: value}}]
    result = group_quantities(elements, [{"column": "C", "fields": field}])
    assert _row(result, storey="L1")["columns"]["C"]["total"] == value


@pytest.mark.parametrize(
    "bad_value",
    [True, "12.5", float("nan"), float("inf")],
)
def test_non_figures_are_missing_not_summed(bad_value):
    elements = [
        {"ifc_type": "IfcWall", "storey": "L1", "q": bad_value},
        {"ifc_type": "IfcWall", "storey": "L1", "q": 1.5},
    ]
    cell = _row(group_quantities(elements, [{"column": "Q", "fields": ["q"]}]),
                storey="L1")["columns"]["Q"]
    assert cell["total"] == 1.5
    assert cell["missing"] == 1


@pytest.mark.parametrize("n", [1, 3])
def test_column_with_no_figure_totals_none_not_zero(n):
    elements = [{"ifc_type": "IfcDoor", "storey": "L1"}] * n
    cell = _row(group_quantities(elements, [{"column": "A", "fields": ["area"]}]),
                storey="L1")["columns"]["A"]
    assert cell["total"] is None
    assert cell["missing"] == n
    assert cell["counted"] == 0


@pytest.mark.parametrize(
    "elements, columns",
    [
        ([{"ifc_type": "IfcWall", "a": 1.0, "name": "x", "b": 2,
           "flag": True}], ["a", "b"]),
        ([{"ifc_type": "IfcSlab", "z": 3.0}, {"ifc_type": "IfcSlab", "y": 1}],
         ["z", "y"]),
    ],
)
def test_default_columns_are_the_elements_own_numeric_keys(elements, columns):
    result = group_quantities(elements)
    assert result["columns"] == columns
    assert "unit" not in result["groups"][0]["columns"][columns[0]]


# ---- table validation ------------------------------------------------------

@pytest.mark.parametrize(
    "table, word",
    [
        ([], "non-empty"),
        ("Vol", "non-empty"),
        (["Vol"], "row 0 is not an object"),
        ([{"fields": ["a"]}], "no column name"),
        ([{"column": "  ", "fields": ["a"]}], "no column name"),
        ([{"column": "V", "fields": []}], "no fields"),
        ([{"column": "V", "fields": ["a", 3]}], "no fields"),
        ([{"column": "V", "fields": "a"}, {"column": "V", "fields": "b"}],
         "repeats column"),
    ],
)
def test_bad_table_is_rejected_whole(table, word):
    with pytest.raises(ValueError, match=word):
        group_quantities([{"ifc_type": "IfcWall", "a": 1.0}], table)


@pytest.mark.parametrize("group_by", [[""], ["storey", 3]])
def test_bad_group_by_is_rejected(group_by):
    with pytest.raises(ValueError, match="qto_group_by"):
        group_quantities([{"ifc_type": "IfcWall"}], group_by=group_by)


# ---- through the block -----------------------------------------------------

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.api  # noqa: E402

from app.blocks.bim_extractor import BIMExtractorBlock  # noqa: E402


def _model(placed):
    """placed: (ifc_class, storey_name, NetVolume or None)."""
    f = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name="P")
    ifcopenshell.api.run("unit.assign_unit", f)
    building = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name="B")
    ifcopenshell.api.run("aggregate.assign_object", f, products=[building], relating_object=project)
    storeys = {}
    for cls, storey, vol in placed:
        if storey not in storeys:
            st = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=storey)
            ifcopenshell.api.run("aggregate.assign_object", f, products=[st], relating_object=building)
            storeys[storey] = st
        el = ifcopenshell.api.run("root.create_entity", f, ifc_class=cls, name=cls)
        ifcopenshell.api.run("spatial.assign_container", f, products=[el], relating_structure=storeys[storey])
        if vol is not None:
            q = ifcopenshell.api.run("pset.add_qto", f, product=el, name="Qto_T")
            ifcopenshell.api.run("pset.edit_qto", f, qto=q, properties={"NetVolume": vol})
    return f


def _run(tmp_path, model, params):
    path = tmp_path / "m.ifc"
    model.write(str(path))
    result = asyncio.run(BIMExtractorBlock().process({"file_path": str(path)}, params))
    assert result["status"] == "success", result.get("error")
    return result


@pytest.mark.parametrize(
    "placed, expected",
    [
        ([("IfcWall", "GF", 2.0), ("IfcWall", "GF", 3.0), ("IfcSlab", "FF", 9.0)],
         {("GF", "IfcWall"): 5.0, ("FF", "IfcSlab"): 9.0}),
        ([("IfcColumn", "L3", 0.5), ("IfcColumn", "L4", None)],
         {("L3", "IfcColumn"): 0.5, ("L4", "IfcColumn"): None}),
    ],
)
def test_block_reports_grouped_takeoff_from_the_table(tmp_path, placed, expected):
    table = [{"column": "Concrete", "fields": ["Qto_T.NetVolume"], "unit": "m3"}]
    result = _run(tmp_path, _model(placed), {"qto_columns": table})
    got = {
        (r["storey"], r["ifc_type"]): r["columns"]["Concrete"]["total"]
        for r in result["qto_groups"]["groups"]
        if r["ifc_type"] != "IfcBuildingStorey"
    }
    assert got == expected
    assert result["qto_groups"]["elements_grouped"] == result["element_count"]


@pytest.mark.parametrize("table", [[{"column": "V"}], "not a table"])
def test_block_keeps_extraction_when_table_is_bad(tmp_path, table):
    result = _run(tmp_path, _model([("IfcWall", "GF", 1.0)]), {"qto_columns": table})
    assert "error" in result["qto_groups"]
    walls = [e for e in result["building_elements"] if e["ifc_type"] == "IfcWall"]
    assert len(walls) == 1


# ---- chat pre-dispatch keeps the grouped table inside the cut --------------

class _Extractor:
    def __init__(self, inner):
        self._inner = inner

    async def execute(self, input_data, params=None):
        return {"block": "bim_extractor", "status": "success", "result": self._inner}


def _predispatch(monkeypatch, filename, inner):
    from app.agents import runtime
    from app.core import projects as projects_mod

    monkeypatch.setattr(projects_mod, "list_documents",
                        lambda pid, **_kw: [{"original_name": filename}],
                        raising=False)
    monkeypatch.setattr(projects_mod, "list_document_names",
                        lambda pid: projects_mod.list_documents(pid))
    monkeypatch.setattr(runtime, "_resolve_file_path", lambda pid, raw: "/d/" + raw)
    monkeypatch.setattr(runtime, "block_instances",
                        {"bim_extractor": _Extractor(inner)}, raising=False)
    agent = runtime.Agent(name="qa", description="", system_prompt="x",
                          allowed_blocks=["bim_extractor"])
    msgs = [{"role": "user", "content": f"Take-off by level from {filename}"}]
    rec = asyncio.run(runtime._predispatch_file_tool(agent, msgs, "p1"))
    assert rec and rec["predispatched"]
    return msgs[-1]["content"]


@pytest.mark.parametrize("filename, n_elements, marker", [
    ("tower_a.ifc", 400, "GF"),
    ("annex.ifc", 900, "L07"),
])
def test_predispatch_puts_qto_groups_ahead_of_bulky_keys(
        monkeypatch, filename, n_elements, marker):
    # building_elements comes FIRST in the block's own dict and alone
    # exceeds the 6000-char cut, so only the lead-key order keeps the
    # grouped table in what the model reads.
    inner = {
        "building_elements": [{"ifc_type": "IfcWall", "guid": f"g{i:05d}"}
                              for i in range(n_elements)],
        "status": "success",
        "quantities": {"walls": {"count": n_elements}},
        "qto_groups": {"groups": [{"storey": marker, "count": n_elements}]},
    }
    injected = _predispatch(monkeypatch, filename, inner)
    assert f'"storey": "{marker}"' in injected
    assert injected.find("qto_groups") < injected.find("quantities")
    assert injected.find("qto_groups") < injected.find("building_elements")


@pytest.mark.parametrize("filename", ["tower_a.ifc", "annex.ifc"])
def test_predispatch_without_qto_groups_adds_no_key(monkeypatch, filename):
    inner = {"status": "success", "quantities": {"walls": {"count": 3}}}
    injected = _predispatch(monkeypatch, filename, inner)
    assert "qto_groups" not in injected
    assert '"walls"' in injected
