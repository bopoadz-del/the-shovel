"""Course-material corrections to four knowledge entries.

Each entry disagreed with the owner-supplied course material it was taken
from. The course pages give the figures in full, so the entries now carry
them, and every corrected entry names the course part and page in
``provenance.course_reference``.

Every value test below uses two different inputs, and each expected value
is hand-calculated in the comment next to it.
"""

import pytest

from app.blocks import _knowledge as kb

CORRECTED = (
    "thermal.equilibrium_time",
    "foundation.dewatering_stop_floors",
    "diaphragm_wall.panel_sequence",
    "asphalt.bitumen_content",
)


@pytest.mark.parametrize("rule_id", CORRECTED)
def test_corrected_entry_names_its_course_source(rule_id):
    ref = kb.get_rule(rule_id)["provenance"]["course_reference"]
    assert ref.startswith("owner-supplied course material")
    assert "p" in ref.split(",")[-1]  # carries a page reference


# ---------------------------------------------------------------------------
# thermal.equilibrium_time: X is the FULL thickness of the pour
# ---------------------------------------------------------------------------


def test_equilibrium_time_variable_is_full_thickness():
    entry = kb.get_rule("thermal.equilibrium_time")
    desc = entry["variables"]["X"]["desc"].lower()
    assert desc.startswith("full thickness")
    assert "half-thickness x" not in entry["statement"].lower()


@pytest.mark.parametrize(
    "thickness_m, hours",
    [
        # 1.2 m slab (course worked example): (1.2/1.5)^2 = 0.64; 168 x 0.64 = 107.52 h
        (1.2, 107.52),
        # 150 m dam (course worked example): (150/1.5)^2 = 10,000; 168 x 10,000 =
        # 1,680,000 h = 10,000 weeks (1,680,000 / 168), about 192 years
        (150.0, 1_680_000.0),
        # invented: 3.0 m raft: (3/1.5)^2 = 4; 168 x 4 = 672 h (four weeks)
        (3.0, 672.0),
    ],
)
def test_equilibrium_time_worked_examples(thickness_m, hours):
    out = kb.evaluate("thermal.equilibrium_time", X=thickness_m)
    assert out["result"] == pytest.approx(hours, rel=1e-9)
    assert out["unit"] == "hour"


# ---------------------------------------------------------------------------
# foundation.dewatering_stop_floors: the full calculation from the course
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, expected",
    [
        ("type", "formula"),  # was "rule" with no expression
        ("needs_review", False),  # was True because the source looked ambiguous
    ],
)
def test_dewatering_stop_entry_is_a_formula_not_an_ambiguity(field, expected):
    entry = kb.get_rule("foundation.dewatering_stop_floors")
    assert entry[field] == expected
    assert "ambiguous" not in entry["statement"].lower()
    assert set(entry["variables"]) == {"min_fos", "uplift", "raft_weight", "floor_weight"}


@pytest.mark.parametrize(
    "min_fos, uplift, raft_weight, floor_weight, floors",
    [
        # Course worked example: head 23 m -> uplift 23 t/m2; 1.25 x 23 = 28.75;
        # raft 4 m x 2.5 t/m3 = 10; floor 0.60 m x 2.5 = 1.5 t/m2;
        # (28.75 - 10) / 1.5 = 12.5 -> say 13 floors (less 5 basements = level 8)
        (1.25, 23.0, 10.0, 1.5, 13),
        # invented: head 15 m, raft 2 m (5 t/m2): 1.25 x 15 = 18.75;
        # (18.75 - 5) / 1.5 = 9.17 -> 10 floors
        (1.25, 15.0, 5.0, 1.5, 10),
        # exact whole number is not rounded up: 1.25 x 12 = 15; (15 - 3) / 1.5 = 8.0 -> 8
        (1.25, 12.0, 3.0, 1.5, 8),
        # the required FOS is an input and is honoured: 1.2 x 23 = 27.6;
        # (27.6 - 10) / 1.5 = 11.73 -> 12 (13 at 1.25)
        (1.2, 23.0, 10.0, 1.5, 12),
    ],
)
def test_dewatering_stop_floors_worked_examples(min_fos, uplift, raft_weight, floor_weight, floors):
    out = kb.evaluate(
        "foundation.dewatering_stop_floors",
        min_fos=min_fos, uplift=uplift, raft_weight=raft_weight, floor_weight=floor_weight,
    )
    assert out["result"] == floors
    assert out["unit"] == "floors"


@pytest.mark.parametrize(
    "uplift, raft_weight",
    [
        # 1.25 x 6 = 7.5 < raft 10 -> raft alone suffices -> 0 floors (not -1)
        (6.0, 10.0),
        # 1.25 x 4 = 5 < raft 6 -> 0 floors
        (4.0, 6.0),
    ],
)
def test_dewatering_stop_floors_never_negative(uplift, raft_weight):
    out = kb.evaluate(
        "foundation.dewatering_stop_floors",
        min_fos=1.25, uplift=uplift, raft_weight=raft_weight, floor_weight=1.5,
    )
    assert out["result"] == 0


@pytest.mark.parametrize("missing", ["min_fos", "floor_weight"])
def test_dewatering_stop_floors_requires_every_input(missing):
    values = {"min_fos": 1.25, "uplift": 23.0, "raft_weight": 10.0, "floor_weight": 1.5}
    values.pop(missing)
    with pytest.raises(ValueError):
        kb.evaluate("foundation.dewatering_stop_floors", **values)


@pytest.mark.parametrize(
    "rule_id, key",
    [
        ("foundation.uplift_fos", "min_fos"),
        ("foundation.uplift_remediation_tension_piles", "trigger_below"),
    ],
)
def test_uplift_fos_threshold_follows_the_course(rule_id, key):
    # The course parts differ on this threshold (1.25 vs 1.2). Owner ruling:
    # the course governs, and the threshold is the 1.2 of section 1 p10.
    entry = kb.get_rule(rule_id)
    assert entry["thresholds"][key] == 1.2
    assert entry["provenance"]["course_reference"] == (
        "owner-supplied course material, roads and earthworks section 1, p10"
    )


@pytest.mark.parametrize(
    "counterweight, uplift, passes",
    [
        # 28 / 23 = 1.217 -> passes at 1.2 (it failed at the old 1.25)
        (28.0, 23.0, True),
        # 27 / 23 = 1.174 -> below 1.2 -> tension piles
        (27.0, 23.0, False),
    ],
)
def test_uplift_fos_judged_against_the_course_threshold(counterweight, uplift, passes):
    fos = kb.evaluate("foundation.uplift_fos", counterweight=counterweight, uplift=uplift)["result"]
    min_fos = kb.get_rule("foundation.uplift_fos")["thresholds"]["min_fos"]
    trigger = kb.get_rule("foundation.uplift_remediation_tension_piles")["thresholds"]["trigger_below"]
    assert (fos >= min_fos) is passes
    assert (fos < trigger) is (not passes)


# ---------------------------------------------------------------------------
# diaphragm_wall.panel_sequence: the five steps are itemised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position, marker",
    [
        (0, "guide walls"),
        (1, "middle bite"),
        (2, "cage"),
        (3, "tremie"),
        (4, "follower panel"),
    ],
)
def test_diaphragm_wall_sequence_is_itemised(position, marker):
    entry = kb.get_rule("diaphragm_wall.panel_sequence")
    steps = entry["thresholds"]["panel_sequence"]
    assert len(steps) == entry["thresholds"]["panel_sequence_steps"] == 5
    assert "not itemise" not in entry["statement"].lower()
    # each step sits in its own position, and the statement names it too
    assert marker in steps[position]
    assert f"({position + 1})" in entry["statement"]
    assert marker in entry["statement"].lower()


# ---------------------------------------------------------------------------
# asphalt.bitumen_content: decimals, not retyped ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, percent",
    [
        ("base_course_percent", 3.6),  # was the range [3, 6]
        ("allowance_40_50_percent", 4.8),  # was the range [4, 8]
        ("binder_course_percent", 4.15),  # unchanged neighbour
        ("wearing_course_percent", 4.45),  # unchanged neighbour
    ],
)
def test_bitumen_content_carries_the_typed_decimals(key, percent):
    th = kb.get_rule("asphalt.bitumen_content")["thresholds"]
    assert th[key] == percent
    assert not any(k.endswith("_range") for k in th)


@pytest.mark.parametrize(
    "course_key, tonnage, bitumen_t",
    [
        # base course: 1,000 t laid x 3.6 / 100 = 36 t bitumen (the retyped "3-6%" gave no single figure)
        ("base_course_percent", 1000.0, 36.0),
        # wearing course: 2,500 t x 4.45 / 100 = 111.25 t
        ("wearing_course_percent", 2500.0, 111.25),
    ],
)
def test_theoretical_bitumen_from_entry(course_key, tonnage, bitumen_t):
    pct = kb.get_rule("asphalt.bitumen_content")["thresholds"][course_key]
    assert tonnage * pct / 100 == pytest.approx(bitumen_t, rel=1e-9)
