"""Create a minimal valid IFC sample file for bim_extractor smoke tests.

Builds a 2-storey "the client project Sample Building" with 8 walls, 2 slabs, 4 columns,
2 beams, 2 doors, 2 windows, 2 spaces, plus a pipe / duct / light fixture
so every category in IFC_CATEGORY_MAP has at least one element.

Every physical element also carries a real extruded-box Body representation in
world coordinates, so ``_geometric_clash_report`` has something to intersect.
Without them the fixture produced ``elements_analyzed=0`` /
``elements_without_geometry=27``, and every clash run was an empty scan that
could never fail -- a green result that proved nothing.

The layout below is arranged so exactly ONE cross-category AABB overlap
exists: ``Pipe-Storm-001`` is driven straight through ``Wall-GF-E``. Every
other pair is deliberately kept clear, including the ones a real model would
overlap (doors and windows sit flush against their wall rather than inside it,
and beams are offset from column tops). Those are genuine AABB false
positives -- the very thing ``_CLASH_DISCLAIMER`` warns about -- and letting
them into the fixture would make the planted-clash assertion untrappable.

IfcSpace and IfcBuildingStorey stay geometry-free on purpose: they are the
real-world reason ``elements_without_geometry`` exists, so the fixture keeps
that counter meaningful.

Usage:
    python scripts/_make_sample_ifc.py [out_path] [--version IFC4|IFC2X3]

Default ``out_path`` is ``tests/fixtures/sample_office.ifc`` (IFC4) or
``tests/fixtures/sample_office_2x3.ifc`` (when --version IFC2X3 is passed)
relative to repo root.
"""
from __future__ import annotations

import argparse
import os
import sys

import ifcopenshell
import ifcopenshell.api

# name -> (centre_x, centre_y, base_z, dx, dy, dz), all in millimetres to match
# the millimetre length unit that ``unit.assign_unit`` writes. The geometry
# engine rescales to metres on read, which is what the clash pass's ``tol_m``
# expects.
BOXES: dict[str, tuple[float, float, float, float, float, float]] = {
    # Ground floor shell: 10m x 6m footprint, 200mm walls, 3m storey.
    "Wall-GF-N": (5000, 5900, 0, 10000, 200, 3000),
    "Wall-GF-S": (5000, 100, 0, 10000, 200, 3000),
    "Wall-GF-E": (9900, 3000, 0, 200, 6000, 3000),
    "Wall-GF-W": (100, 3000, 0, 200, 6000, 3000),
    # Slabs sit strictly below/above the walls they meet, so slab-vs-wall
    # touches at a shared face and does not register as interpenetration.
    "Slab-GF": (5000, 3000, -200, 10000, 6000, 200),
    "Col-GF-1": (2000, 2000, 0, 400, 400, 3000),
    "Col-GF-2": (8000, 2000, 0, 400, 400, 3000),
    # Offset in Y from both columns: a beam bearing ON a column is an AABB
    # overlap but not a clash.
    "Beam-GF-Main": (5000, 4000, 2800, 6000, 300, 200),
    # Level 1 shell, lifted clear of the GF slab soffit.
    "Wall-L1-N": (5000, 5900, 3200, 10000, 200, 3000),
    "Wall-L1-S": (5000, 100, 3200, 10000, 200, 3000),
    "Wall-L1-E": (9900, 3000, 3200, 200, 6000, 3000),
    "Wall-L1-W": (100, 3000, 3200, 200, 6000, 3000),
    "Slab-L1": (5000, 3000, 3000, 10000, 6000, 200),
    "Col-L1-1": (2000, 2000, 3200, 400, 400, 3000),
    "Col-L1-2": (8000, 2000, 3200, 400, 400, 3000),
    "Beam-L1-Main": (5000, 4000, 6000, 6000, 300, 200),
    # Openings flush against the inner wall face, not embedded in it.
    "Main-Entrance-Door": (5000, 250, 0, 1000, 100, 2100),
    "L1-Door-A": (5000, 250, 3200, 1000, 100, 2100),
    "GF-Window-North": (3000, 5750, 900, 1200, 100, 1200),
    "L1-Window-East": (9750, 3000, 4100, 100, 1200, 1200),
    "Light-GF-01": (3000, 3000, 2900, 200, 200, 100),
    "Duct-AHU-1": (5000, 3000, 5800, 8000, 400, 300),
    # THE PLANTED CLASH: this storm pipe runs from inside the building out
    # through Wall-GF-E (x 9800..10000). 200mm of interpenetration, far above
    # the 10mm default tolerance, across two different categories
    # (pipes vs walls) so the same-category skip does not swallow it.
    "Pipe-Storm-001": (9250, 3000, 1400, 2500, 200, 200),
}

# The one pair the fixture guarantees. Tests assert on this by name.
PLANTED_CLASH = ("Pipe-Storm-001", "Wall-GF-E")


def _bootstrap_owner(m) -> None:
    """IFC2x3 requires at least one IfcPersonAndOrganization and one
    IfcApplication present before owner_history can be created. IFC4 is
    permissive. Creating both keeps the builder schema-agnostic; the
    settings module then picks them up via ``by_type`` automatically."""
    person = ifcopenshell.api.run(
        "owner.add_person", m,
        identification="bim", family_name="Sample", given_name="Builder",
    )
    org = ifcopenshell.api.run(
        "owner.add_organisation", m,
        identification="client", name="the client project Sample",
    )
    ifcopenshell.api.run(
        "owner.add_person_and_organisation", m,
        person=person, organisation=org,
    )
    ifcopenshell.api.run(
        "owner.add_application", m,
        application_developer=org,
        version="1.0",
        application_full_name="the client project Sample Generator",
        application_identifier="client-sample",
    )


def _dir(m, ratios):
    return m.create_entity(
        "IfcDirection", DirectionRatios=tuple(float(r) for r in ratios)
    )


def _pt3(m, coords):
    return m.create_entity(
        "IfcCartesianPoint", Coordinates=tuple(float(c) for c in coords)
    )


def _axis3(m, origin=(0.0, 0.0, 0.0)):
    return m.create_entity(
        "IfcAxis2Placement3D",
        Location=_pt3(m, origin),
        Axis=_dir(m, (0.0, 0.0, 1.0)),
        RefDirection=_dir(m, (1.0, 0.0, 0.0)),
    )


def _box_shape(m, body_ctx, dx: float, dy: float, dz: float):
    """A dx*dy rectangle extruded dz along +Z, centred on the local origin in XY.

    ``Position`` is set explicitly on the profile: it is optional in IFC4 but
    REQUIRED on IfcParameterizedProfileDef in IFC2X3, and this builder emits
    both schemas.
    """
    profile = m.create_entity(
        "IfcRectangleProfileDef",
        ProfileType="AREA",
        Position=m.create_entity(
            "IfcAxis2Placement2D",
            Location=m.create_entity(
                "IfcCartesianPoint", Coordinates=(0.0, 0.0)
            ),
            RefDirection=m.create_entity(
                "IfcDirection", DirectionRatios=(1.0, 0.0)
            ),
        ),
        XDim=float(dx),
        YDim=float(dy),
    )
    solid = m.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=_axis3(m),
        ExtrudedDirection=_dir(m, (0.0, 0.0, 1.0)),
        Depth=float(dz),
    )
    rep = m.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body_ctx,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid],
    )
    return m.create_entity("IfcProductDefinitionShape", Representations=[rep])


def _give_geometry(m, body_ctx, element, name: str) -> bool:
    """Attach a world-coordinate placement + box body to ``element``.

    Placement is absolute (``PlacementRelTo=None``) rather than nested under the
    storey, so USE_WORLD_COORDS is a no-op and the AABBs the clash pass reads
    are exactly the millimetre figures in ``BOXES``.
    """
    box = BOXES.get(name)
    if box is None:
        return False
    cx, cy, bz, dx, dy, dz = box
    element.ObjectPlacement = m.create_entity(
        "IfcLocalPlacement",
        PlacementRelTo=None,
        RelativePlacement=_axis3(m, (cx, cy, bz)),
    )
    element.Representation = _box_shape(m, body_ctx, dx, dy, dz)
    return True


def build(out_path: str, version: str = "IFC4") -> None:
    m = ifcopenshell.api.run("project.create_file", version=version)
    _bootstrap_owner(m)
    project = ifcopenshell.api.run("root.create_entity", m, ifc_class="IfcProject", name="the client project Sample Building")
    ifcopenshell.api.run("unit.assign_unit", m)
    ctx = ifcopenshell.api.run("context.add_context", m, context_type="Model")
    body_ctx = ifcopenshell.api.run(
        "context.add_context", m,
        context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=ctx,
    )

    site = ifcopenshell.api.run("root.create_entity", m, ifc_class="IfcSite", name="Site A")
    bldg = ifcopenshell.api.run("root.create_entity", m, ifc_class="IfcBuilding", name="Office Block 1")
    s1 = ifcopenshell.api.run("root.create_entity", m, ifc_class="IfcBuildingStorey", name="Ground Floor")
    s2 = ifcopenshell.api.run("root.create_entity", m, ifc_class="IfcBuildingStorey", name="Level 1")
    ifcopenshell.api.run("aggregate.assign_object", m, products=[site], relating_object=project)
    ifcopenshell.api.run("aggregate.assign_object", m, products=[bldg], relating_object=site)
    ifcopenshell.api.run("aggregate.assign_object", m, products=[s1, s2], relating_object=bldg)

    placed: list[str] = []

    def add_to_storey(cls_name: str, name: str, storey, optional: bool = False) -> object:
        try:
            e = ifcopenshell.api.run("root.create_entity", m, ifc_class=cls_name, name=name)
        except RuntimeError as exc:
            if optional and "not found in schema" in str(exc):
                return None
            raise
        ifcopenshell.api.run("spatial.assign_container", m, products=[e], relating_structure=storey)
        if _give_geometry(m, body_ctx, e, name):
            placed.append(name)
        return e

    for floor, storey in (("GF", s1), ("L1", s2)):
        for side in ("N", "S", "E", "W"):
            add_to_storey("IfcWall", f"Wall-{floor}-{side}", storey)
        add_to_storey("IfcSlab", f"Slab-{floor}", storey)
        for i in (1, 2):
            add_to_storey("IfcColumn", f"Col-{floor}-{i}", storey)
        add_to_storey("IfcBeam", f"Beam-{floor}-Main", storey)

    add_to_storey("IfcDoor", "Main-Entrance-Door", s1)
    add_to_storey("IfcDoor", "L1-Door-A", s2)
    add_to_storey("IfcWindow", "GF-Window-North", s1)
    add_to_storey("IfcWindow", "L1-Window-East", s2)
    # Schema-conditional: IFC2X3 lacks IfcPipeSegment/IfcDuctSegment (added
    # in IFC4) and the IfcLightFixture in some early IFC2X3 builds. Skip if
    # the active schema does not declare them -- the core fixture is still
    # valid without these MEP entities. NOTE: the planted clash lives on the
    # pipe, so the IFC2X3 fixture carries geometry but no planted clash.
    add_to_storey("IfcPipeSegment", "Pipe-Storm-001", s1, optional=True)
    add_to_storey("IfcDuctSegment", "Duct-AHU-1", s2, optional=True)
    add_to_storey("IfcLightFixture", "Light-GF-01", s1, optional=True)

    # IfcSpace uses aggregate (not spatial.assign_container) in IFC4.
    # Left geometry-free on purpose -- see module docstring.
    for storey, name in ((s1, "Office-Room-1"), (s2, "Office-Room-2")):
        sp = ifcopenshell.api.run("root.create_entity", m, ifc_class="IfcSpace", name=name)
        ifcopenshell.api.run("aggregate.assign_object", m, products=[sp], relating_object=storey)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.write(out_path)
    clash_ok = all(n in placed for n in PLANTED_CLASH)
    print(
        f"wrote IFC -> {out_path}  size={os.path.getsize(out_path)}B  schema={m.schema}  "
        f"walls={len(m.by_type('IfcWall'))} slabs={len(m.by_type('IfcSlab'))} "
        f"columns={len(m.by_type('IfcColumn'))} beams={len(m.by_type('IfcBeam'))} "
        f"with_geometry={len(placed)} "
        f"planted_clash={'yes' if clash_ok else 'no (schema lacks IfcPipeSegment)'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_path", nargs="?", default=None)
    parser.add_argument("--version", choices=["IFC4", "IFC2X3"], default="IFC4")
    args = parser.parse_args()
    out = args.out_path
    if not out:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fname = "sample_office.ifc" if args.version == "IFC4" else "sample_office_2x3.ifc"
        out = os.path.join(repo_root, "tests", "fixtures", fname)
    build(out, version=args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
