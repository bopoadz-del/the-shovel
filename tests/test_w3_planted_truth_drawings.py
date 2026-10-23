"""W3 un-park — drawings/BIM against PLANTED ground truth.

New test shape (2026-07-25): instead of asserting the extractor "returns
something" on a sample file, we BUILD the file so the truth is known exactly:

- DXF: a 10 m x 5 m closed slab polyline (area exactly 50 m2) and a single
  25 m pipe line, drawn in metres. The QTO extractor must report those
  numbers, not merely succeed.
- IFC: a column and a beam whose boxes overlap by design (planted clash)
  plus a wall placed 50 m away (must NOT clash). The clash report must find
  the planted pair and nothing involving the distant wall.
"""
import math

import pytest


# ── DXF planted truth ────────────────────────────────────────────────────────

@pytest.fixture
def planted_dxf(tmp_path):
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.units = 6  # metres
    msp = doc.modelspace()
    # closed rectangle 10 x 5 on layer SLAB -> area 50 m2, perimeter 30 m
    msp.add_lwpolyline(
        [(0, 0), (10, 0), (10, 5), (0, 5)], close=True,
        dxfattribs={"layer": "SLAB"},
    )
    # one 25 m line on layer PIPES
    msp.add_line((0, 10), (25, 10), dxfattribs={"layer": "PIPES"})
    path = tmp_path / "planted.dxf"
    doc.saveas(path)
    return str(path)


@pytest.mark.asyncio
async def test_dxf_qto_recovers_planted_area_and_length(planted_dxf):
    from app.blocks.drawing_qto import DrawingQTOBlock

    result = await DrawingQTOBlock().process({"file_path": planted_dxf})
    assert result["status"] == "success", result
    # Planted slab: exactly 50 m2.
    assert result["total_area_m2"] == pytest.approx(50.0, rel=1e-6), result["areas"]
    # Planted lengths: the 25 m pipe + the 30 m slab perimeter polyline.
    lengths = sorted(m["length_m"] for m in result["measurements"])
    assert 25.0 in [round(x, 6) for x in lengths], lengths
    assert result["total_length_m"] == pytest.approx(55.0, rel=1e-6)
    # Layers surfaced for provenance.
    assert {"SLAB", "PIPES"} <= set(result["layers"])
    # Nothing silently skipped on a clean file.
    assert result.get("entities_skipped", 0) == 0


@pytest.mark.asyncio
async def test_dxf_qto_honours_declared_units(planted_dxf, tmp_path):
    # Same geometry drawn in MILLIMETRES must yield the same metric truth.
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.units = 4  # millimetres
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (10000, 0), (10000, 5000), (0, 5000)], close=True,
        dxfattribs={"layer": "SLAB"},
    )
    path = tmp_path / "planted_mm.dxf"
    doc.saveas(path)

    from app.blocks.drawing_qto import DrawingQTOBlock

    result = await DrawingQTOBlock().process({"file_path": str(path)})
    assert result["status"] == "success", result
    assert result["total_area_m2"] == pytest.approx(50.0, rel=1e-6)


# ── IFC planted clash ────────────────────────────────────────────────────────

def _make_ifc(tmp_path):
    """Column and beam overlapping at the origin; wall 50 m away."""
    import ifcopenshell
    import ifcopenshell.api as api

    model = api.run("project.create_file")
    api.run("root.create_entity", model, ifc_class="IfcProject", name="Planted")
    # Explicit METRE length unit: raw entity attributes (profile XDim/YDim)
    # are written in PROJECT units, and the default assign_unit is mm — which
    # silently turned a 400 mm column into a 0.4 mm sliver.
    metre = api.run("unit.add_si_unit", model, unit_type="LENGTHUNIT")
    api.run("unit.assign_unit", model, units=[metre])
    ctx = api.run("context.add_context", model, context_type="Model")
    body = api.run(
        "context.add_context", model, context_type="Model",
        context_identifier="Body", target_view="MODEL_VIEW", parent=ctx,
    )
    site = api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    storey = api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="L1")
    api.run("aggregate.assign_object", model, products=[site], relating_object=model.by_type("IfcProject")[0])
    api.run("aggregate.assign_object", model, products=[storey], relating_object=site)

    import numpy as np

    def _place(product, x=0.0, y=0.0, z=0.0):
        m = np.eye(4)
        m[0][3], m[1][3], m[2][3] = x, y, z
        api.run("geometry.edit_object_placement", model, product=product, matrix=m)

    def _box(ifc_class, name, x, y, z, dx, dy, depth):
        el = api.run("root.create_entity", model, ifc_class=ifc_class, name=name)
        pos2d = model.create_entity(
            "IfcAxis2Placement2D",
            Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0)),
        )
        profile = model.create_entity(
            "IfcRectangleProfileDef", ProfileType="AREA", Position=pos2d,
            XDim=dx, YDim=dy,
        )
        rep = api.run(
            "geometry.add_profile_representation", model,
            context=body, profile=profile, depth=depth,
        )
        api.run("geometry.assign_representation", model, product=el, representation=rep)
        _place(el, x, y, z)
        api.run("spatial.assign_container", model, products=[el], relating_structure=storey)
        return el

    # Column 0.4x0.4x3 at origin; beam 2x0.3x0.3 straight through it.
    _box("IfcColumn", "COL-PLANT", 0, 0, 0, 0.4, 0.4, 3.0)
    _box("IfcBeam", "BEAM-PLANT", 0, 0, 1.0, 2.0, 0.3, 0.3)
    # Wall far away — must not clash with anything.
    _box("IfcWall", "WALL-FAR", 50, 50, 0, 5.0, 0.2, 3.0)

    path = tmp_path / "planted_clash.ifc"
    model.write(str(path))
    return str(path)


@pytest.mark.asyncio
async def test_ifc_planted_clash_detected_and_distant_wall_clean(tmp_path):
    pytest.importorskip("ifcopenshell.geom")
    from app.blocks.bim_extractor import BIMExtractorBlock

    path = _make_ifc(tmp_path)
    result = await BIMExtractorBlock().process(
        {"file_path": path}, {"run_clash_detection": True}
    )
    assert result.get("status") == "success", result

    report = result.get("clash_report") or {}
    assert report.get("detection_method") == "aabb_intersection", report
    clashes = report.get("clashes") or []
    names = [
        (c.get("name_a", ""), c.get("name_b", ""))
        for c in clashes
    ]
    flat = " | ".join(f"{a}+{b}" for a, b in names)
    # The planted column-beam intersection must be found...
    assert any("COL-PLANT" in f"{a}{b}" and "BEAM-PLANT" in f"{a}{b}" for a, b in names), (
        f"planted clash missing: {flat or 'no clashes reported'}"
    )
    # ...and the distant wall must be clean.
    assert not any("WALL-FAR" in f"{a}{b}" for a, b in names), flat
