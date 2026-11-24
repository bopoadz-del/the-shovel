"""S-K3 item 3: IFC data extraction reports what it did not extract and
where each element sits.

Every test writes its own IFC with ifcopenshell.api; no fixture file, no
class or storey name is read by product code.
"""
from __future__ import annotations

import asyncio

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.api  # noqa: E402
import ifcopenshell.util.element as ifc_util  # noqa: E402

from app.blocks import bim_extractor as bx  # noqa: E402
from app.blocks.bim_extractor import BIMExtractorBlock  # noqa: E402


def _model(storeys, placed, loose=()):
    """storeys: storey names; placed: (ifc_class, storey_name) pairs contained
    in that storey; loose: ifc classes created with no spatial container."""
    f = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name="P")
    ifcopenshell.api.run("unit.assign_unit", f)
    site = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSite", name="S")
    building = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name="B")
    ifcopenshell.api.run("aggregate.assign_object", f, products=[site], relating_object=project)
    ifcopenshell.api.run("aggregate.assign_object", f, products=[building], relating_object=site)
    by_name = {}
    for name in storeys:
        st = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=name)
        ifcopenshell.api.run("aggregate.assign_object", f, products=[st], relating_object=building)
        by_name[name] = st
    for cls, storey in placed:
        el = ifcopenshell.api.run("root.create_entity", f, ifc_class=cls, name=cls)
        ifcopenshell.api.run(
            "spatial.assign_container", f, products=[el], relating_structure=by_name[storey]
        )
    for cls in loose:
        ifcopenshell.api.run("root.create_entity", f, ifc_class=cls, name=cls)
    return f


def _extract(tmp_path, model):
    path = tmp_path / "m.ifc"
    model.write(str(path))
    result = asyncio.run(BIMExtractorBlock().process({"file_path": str(path)}, {}))
    assert result["status"] == "success", result.get("error")
    return result


# ---- types_not_extracted ---------------------------------------------------

@pytest.mark.parametrize(
    "placed, expected",
    [
        (
            [("IfcWall", "A"), ("IfcFurniture", "A"), ("IfcFurniture", "A")],
            {"IfcFurniture": 2},
        ),
        (
            [("IfcSlab", "A"), ("IfcTransportElement", "A"), ("IfcFurniture", "A")],
            {"IfcFurniture": 1, "IfcTransportElement": 1},
        ),
    ],
)
def test_unmapped_classes_are_counted_not_dropped(tmp_path, placed, expected):
    result = _extract(tmp_path, _model(["A"], placed))
    assert result["types_not_extracted"] == expected
    extracted = {e["ifc_type"] for e in result["building_elements"]}
    assert not extracted & set(expected)


@pytest.mark.parametrize(
    "placed",
    [
        [("IfcWall", "A"), ("IfcColumn", "A")],
        [("IfcBeam", "A"), ("IfcDoor", "A"), ("IfcWindow", "A")],
    ],
)
def test_fully_mapped_model_reports_nothing_outside(tmp_path, placed):
    result = _extract(tmp_path, _model(["A"], placed))
    assert result["types_not_extracted"] == {}
    assert result["element_count"] == len(placed) + 1  # + the storey


@pytest.mark.parametrize(
    "mapped, subtype_cls",
    [("IfcFlowTerminal", "IfcAirTerminal"), ("IfcFlowTerminal", "IfcSanitaryTerminal")],
)
def test_subtype_of_a_mapped_class_is_covered(tmp_path, mapped, subtype_cls):
    # IFC_CATEGORY_MAP must carry the supertype for this test to mean anything.
    assert mapped in bx.IFC_CATEGORY_MAP
    result = _extract(tmp_path, _model(["A"], [(subtype_cls, "A")]))
    assert result["types_not_extracted"] == {}
    assert subtype_cls in {e["ifc_type"] for e in result["building_elements"]}


class _RaisingModel:
    def by_type(self, _cls):
        raise RuntimeError("schema lookup failed")


class _ListModel:
    def __init__(self, items):
        self._items = items

    def by_type(self, _cls):
        return self._items


class _El:
    def __init__(self, cls, supers=()):
        self._cls = cls
        self._supers = set(supers) | {cls}

    def is_a(self, name=None):
        return self._cls if name is None else name in self._supers


@pytest.mark.parametrize("model", [_RaisingModel(), _ListModel([])])
def test_types_outside_map_empty_when_nothing_readable(model):
    assert bx._types_outside_map(model) == {}


@pytest.mark.parametrize(
    "items, expected",
    [
        ([_El("IfcX"), _El("IfcX"), _El("IfcWall")], {"IfcX": 2}),
        ([_El("IfcY", supers={"IfcBeam"}), _El("IfcZ")], {"IfcZ": 1}),
    ],
)
def test_types_outside_map_counts_by_class(items, expected):
    assert bx._types_outside_map(_ListModel(items)) == expected


# ---- storey containment ----------------------------------------------------

@pytest.mark.parametrize(
    "storeys, placed",
    [
        (["A", "B"], [("IfcWall", "A"), ("IfcWall", "B"), ("IfcColumn", "B")]),
        (["Lower", "Upper", "Roof"], [("IfcSlab", "Roof"), ("IfcBeam", "Lower")]),
    ],
)
def test_each_element_carries_its_container_storey(tmp_path, storeys, placed):
    result = _extract(tmp_path, _model(storeys, placed))
    got = sorted(
        (e["ifc_type"], e["storey"])
        for e in result["building_elements"]
        if e["ifc_type"] != "IfcBuildingStorey"
    )
    assert got == sorted(placed)


@pytest.mark.parametrize("loose", [("IfcWall",), ("IfcColumn", "IfcBeam")])
def test_uncontained_element_has_no_storey_key(tmp_path, loose):
    result = _extract(tmp_path, _model(["A"], [], loose=loose))
    loose_rows = [e for e in result["building_elements"] if e["ifc_type"] in loose]
    assert len(loose_rows) == len(loose)
    assert all("storey" not in e for e in loose_rows)


class _Util:
    def __init__(self, container=None, exc=None):
        self._container = container
        self._exc = exc

    def get_container(self, el, ifc_class=None):
        assert ifc_class == "IfcBuildingStorey"
        if self._exc:
            raise self._exc
        return self._container


class _Named:
    def __init__(self, name):
        self.Name = name


@pytest.mark.parametrize(
    "util, expected",
    [
        (_Util(container=None), None),
        (_Util(exc=RuntimeError("broken")), None),
        (_Util(container=_Named("L2")), "L2"),
        (_Util(container=_Named(None)), ""),
    ],
)
def test_container_storey_name(util, expected):
    assert bx._container_storey_name(object(), util) == expected


def test_real_util_signature_accepts_ifc_class():
    # The helper calls get_container(el, ifc_class=...); the pinned
    # ifcopenshell must accept it, or every storey silently becomes None.
    import inspect

    assert "ifc_class" in inspect.signature(ifc_util.get_container).parameters
