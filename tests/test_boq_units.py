"""Unit-of-measurement inference from BOQ item descriptions (CESMM4/POMI)."""
import pytest

from app.lib.boq_units import infer_unit, reconcile_unit, canon_unit


@pytest.mark.parametrize("desc,unit", [
    ("Excavation in trenches not exceeding 2m deep", "m3"),
    ("Backfilling to trenches with selected fill", "m3"),
    ("Disposal of surplus excavated material off site", "m3"),
    ("Reinforced concrete grade C40 to foundations", "m3"),
    ("Blinding concrete 50mm thick", "m3"),
    ("Granular fill to pipe surround", "m3"),
    ("Formwork to sides of concrete beams", "m2"),          # formwork beats concrete
    ("Mesh reinforcement A393 fabric to slab", "m2"),       # mesh beats reinforcement->t
    ("High yield reinforcement bar to columns", "t"),
    ("Structural steelwork universal beams", "t"),
    ("300mm dia vitrified clay gravity sewer pipe", "m"),
    ("Waste water / foul pipe, depth 1.5 to 2m", "m"),
    ("Precast concrete kerb laid to radius", "m"),          # kerb -> linear
    ("Circular manhole 1200mm dia, cover slab", "nr"),      # manhole -> nr
    ("Pipe fitting - 45 degree bend", "nr"),                # fitting/bend -> nr before pipe
    ("Supply and install gate valve", "nr"),
    ("Aluminium door single leaf 900x2100", "nr"),
    ("Two coats emulsion paint to walls", "m2"),
    ("12mm cement plaster to blockwork", "m2"),
    ("200mm thick blockwork wall", "m2"),
    ("Preliminaries and general items", "sum"),
    ("Provisional sum for testing", "sum"),
])
def test_infer_unit(desc, unit):
    assert infer_unit(desc) == unit


def test_infer_unit_none_for_unknown():
    assert infer_unit("miscellaneous allowance widget") is None
    assert infer_unit("") is None


def test_reconcile_fills_blank_from_worktype():
    unit, source, suspect, expected = reconcile_unit("", "Excavation to reduce levels")
    assert (unit, source, suspect) == ("m3", "inferred", False)


def test_reconcile_flags_contradiction_but_keeps_stated():
    # OCR read 'm' for an excavation line -> should stay 'm' but be flagged m3.
    unit, source, suspect, expected = reconcile_unit("m", "Bulk excavation")
    assert unit == "m" and suspect is True and expected == "m3"


def test_reconcile_no_flag_when_consistent_via_canon():
    # 'mq' (Italian m2) for plaster canonicalises to m2 -> consistent, no flag.
    unit, source, suspect, expected = reconcile_unit("mq", "Gypsum plaster to ceiling")
    assert suspect is False and unit == "mq"
    assert canon_unit("mq") == "m2"


def test_reconcile_keeps_unknown_worktype_unit():
    unit, source, suspect, expected = reconcile_unit("m2", "special bespoke item")
    assert (unit, source, suspect, expected) == ("m2", "parsed", False, None)
