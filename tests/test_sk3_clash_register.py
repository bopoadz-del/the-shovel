"""S-K3 item 6: clash resolution register from the caller's strategy table.

Every test builds its own clashes and its own table. Strategy words are
made up per test so no assertion can pass on a name the code knows.
"""
import math

import pytest

from app.blocks.clash_register import build_clash_register


def _clash(cat_a, cat_b, type_a=None, type_b=None, overlap=25.0, a="A", b="B"):
    return {
        "type": "aabb_overlap",
        "element_a": a,
        "element_b": b,
        "category_a": cat_a,
        "category_b": cat_b,
        "ifc_type_a": type_a,
        "ifc_type_b": type_b,
        "overlap_mm": overlap,
        "severity": "warning",
    }


def _only(register):
    assert register["clash_rows"] == 1
    return register["rows"][0]


# --- no table ---------------------------------------------------------------

@pytest.mark.parametrize("clashes", [
    [_clash("pipes", "beams")],
    [_clash("ducts", "walls"), _clash("trays", "slabs")],
])
def test_no_table_leaves_every_clash_unassigned(clashes):
    reg = build_clash_register(clashes, None)
    assert reg["table_given"] is False
    assert reg["unassigned"] == len(clashes)
    assert reg["by_strategy"] == {}
    assert reg["table_rows_unused"] == []
    for row in reg["rows"]:
        assert row["strategy"] is None
        assert row["reason"] == "no clash strategy table was given"
        assert row["status"] == "open"


def test_empty_table_is_given_but_matches_nothing():
    reg = build_clash_register([_clash("pipes", "beams")], [])
    assert reg["table_given"] is True
    assert _only(reg)["reason"] == "no table row matches this clash"


# --- matching -----------------------------------------------------------------

@pytest.mark.parametrize("cat_a,cat_b", [("pipes", "beams"), ("beams", "pipes")])
def test_category_row_matches_either_order(cat_a, cat_b):
    table = [{"a": {"category": "pipes"}, "b": {"category": "beams"},
              "strategy": "zq-offset", "responsible": "trade-1"}]
    row = _only(build_clash_register([_clash(cat_a, cat_b)], table))
    assert row["strategy"] == "zq-offset"
    assert row["responsible"] == "trade-1"
    assert row["table_row"] == 0


@pytest.mark.parametrize("cat_a,cat_b", [("pipes", "walls"), ("ducts", "beams")])
def test_category_row_does_not_match_other_pairs(cat_a, cat_b):
    table = [{"a": {"category": "pipes"}, "b": {"category": "beams"}, "strategy": "zq-offset"}]
    reg = build_clash_register([_clash(cat_a, cat_b)], table)
    assert _only(reg)["strategy"] is None
    assert reg["table_rows_unused"] == [0]


@pytest.mark.parametrize("value", ["PIPES", "  Pipes "])
def test_selector_ignores_case_and_spaces(value):
    table = [{"a": {"category": value}, "strategy": "kx"}]
    assert _only(build_clash_register([_clash("pipes", "beams")], table))["strategy"] == "kx"


@pytest.mark.parametrize("cat_a,cat_b", [("pipes", "beams"), ("walls", "ducts")])
def test_row_without_sides_matches_any_clash(cat_a, cat_b):
    table = [{"strategy": "catch-all-q"}]
    assert _only(build_clash_register([_clash(cat_a, cat_b)], table))["strategy"] == "catch-all-q"


def test_one_sided_row_needs_that_side_present():
    table = [{"b": {"category": "ducts"}, "strategy": "dz"}]
    reg = build_clash_register([_clash("ducts", "beams"), _clash("pipes", "beams")], table)
    assert [r["strategy"] for r in reg["rows"]] == ["dz", None]


def test_name_duplicate_clash_uses_its_shared_category_on_both_sides():
    dup = {"type": "name_duplicate", "element_a": "x", "element_b": "y",
           "category": "walls", "severity": "warning"}
    hit = [{"a": {"category": "walls"}, "b": {"category": "walls"}, "strategy": "merge-q"}]
    miss = [{"a": {"category": "walls"}, "b": {"category": "pipes"}, "strategy": "merge-q"}]
    assert _only(build_clash_register([dup], hit))["strategy"] == "merge-q"
    assert _only(build_clash_register([dup], miss))["strategy"] is None
    assert _only(build_clash_register([dup], hit))["overlap_mm"] is None


# --- specificity --------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1])
def test_ifc_type_row_beats_category_row_whatever_the_order(order):
    rows = [
        {"a": {"category": "pipes"}, "b": {"category": "beams"}, "strategy": "general-q"},
        {"a": {"ifc_type": "IfcPipeSegment"}, "b": {"category": "beams"}, "strategy": "special-q"},
    ]
    table = rows if order == 0 else rows[::-1]
    row = _only(build_clash_register(
        [_clash("pipes", "beams", "IfcPipeSegment", "IfcBeam")], table))
    assert row["strategy"] == "special-q"


def test_ifc_type_row_does_not_apply_to_another_type():
    table = [
        {"a": {"category": "pipes"}, "strategy": "general-q"},
        {"a": {"ifc_type": "IfcPipeSegment"}, "strategy": "special-q"},
    ]
    row = _only(build_clash_register([_clash("pipes", "beams", "IfcPipeFitting")], table))
    assert row["strategy"] == "general-q"


def test_band_row_beats_same_selector_row_without_band():
    table = [
        {"a": {"category": "pipes"}, "strategy": "plain-q"},
        {"a": {"category": "pipes"}, "max_overlap_mm": 50, "strategy": "shallow-q"},
    ]
    reg = build_clash_register(
        [_clash("pipes", "beams", overlap=10), _clash("pipes", "beams", overlap=80)], table)
    assert [r["strategy"] for r in reg["rows"]] == ["shallow-q", "plain-q"]


def test_tied_rows_with_different_strategies_assign_nothing():
    table = [
        {"a": {"category": "pipes"}, "strategy": "one-q"},
        {"b": {"category": "beams"}, "strategy": "two-q"},
    ]
    row = _only(build_clash_register([_clash("pipes", "beams")], table))
    assert row["strategy"] is None
    assert row["tied_rows"] == [0, 1]


def test_tied_rows_with_the_same_outcome_assign_it():
    table = [
        {"a": {"category": "pipes"}, "strategy": "one-q", "responsible": "t"},
        {"b": {"category": "beams"}, "strategy": "one-q", "responsible": "t"},
    ]
    row = _only(build_clash_register([_clash("pipes", "beams")], table))
    assert row["strategy"] == "one-q"
    assert row["table_row"] == 0


def test_tied_rows_differing_only_in_responsible_assign_nothing():
    table = [
        {"a": {"category": "pipes"}, "strategy": "one-q", "responsible": "t1"},
        {"b": {"category": "beams"}, "strategy": "one-q", "responsible": "t2"},
    ]
    assert _only(build_clash_register([_clash("pipes", "beams")], table))["strategy"] is None


# --- depth bands --------------------------------------------------------------

@pytest.mark.parametrize("overlap,expected", [
    (20.0, "band-q"),    # min is inclusive
    (19.9, None),
    (49.9, "band-q"),
    (50.0, None),        # max is exclusive
])
def test_band_edges(overlap, expected):
    table = [{"min_overlap_mm": 20, "max_overlap_mm": 50, "strategy": "band-q"}]
    assert _only(build_clash_register([_clash("p", "q", overlap=overlap)], table))["strategy"] == expected


@pytest.mark.parametrize("overlap", [None, "deep", float("nan")])
def test_band_row_cannot_decide_a_clash_without_depth(overlap):
    table = [
        {"strategy": "any-q"},
        {"a": {"category": "p"}, "min_overlap_mm": 5, "strategy": "band-q"},
    ]
    row = _only(build_clash_register([_clash("p", "q", overlap=overlap)], table))
    assert row["strategy"] is None
    assert row["depth_rows"] == [1]
    assert row["overlap_mm"] is None


def test_clash_without_depth_still_takes_a_row_with_no_band():
    table = [{"a": {"category": "p"}, "strategy": "any-q"},
             {"a": {"category": "other"}, "min_overlap_mm": 5, "strategy": "band-q"}]
    assert _only(build_clash_register([_clash("p", "q", overlap=None)], table))["strategy"] == "any-q"


# --- summary ------------------------------------------------------------------

def test_summary_counts_strategies_and_unused_rows():
    table = [
        {"a": {"category": "p"}, "strategy": "s-two"},
        {"a": {"category": "d"}, "strategy": "s-one"},
        {"a": {"category": "never"}, "strategy": "s-dead"},
    ]
    clashes = [_clash("p", "w"), _clash("d", "w"), _clash("p", "x"), _clash("z", "w")]
    reg = build_clash_register(clashes, table)
    assert reg["by_strategy"] == {"s-two": 2, "s-one": 1}
    assert list(reg["by_strategy"]) == ["s-two", "s-one"]
    assert reg["unassigned"] == 1
    assert reg["table_rows_unused"] == [2]
    assert [r["ref"] for r in reg["rows"]] == [1, 2, 3, 4]


def test_non_dict_clash_entries_are_skipped():
    reg = build_clash_register(["junk", _clash("p", "q")], [{"strategy": "s"}])
    assert reg["clash_rows"] == 1
    assert reg["rows"][0]["ref"] == 2


# --- table validation ---------------------------------------------------------

@pytest.mark.parametrize("table,fragment", [
    ({"strategy": "s"}, "must be a list"),
    (["s"], "row 0: must be a mapping"),
    ([{"strategy": "s"}, {"stratgy": "s"}], "row 1: unknown key(s) stratgy"),
    ([{"strategy": "  "}], "row 0: 'strategy' must be non-empty text"),
    ([{"strategy": 3}], "row 0: 'strategy' must be non-empty text"),
    ([{"strategy": "s", "responsible": 4}], "row 0: 'responsible' must be text"),
    ([{"strategy": "s", "a": {"category": "p", "ifc_type": "q"}}], "row 0: 'a' must hold exactly one"),
    ([{"strategy": "s", "b": "pipes"}], "row 0: 'b' must hold exactly one"),
    ([{"strategy": "s", "a": {"discipline": "p"}}], "row 0: 'a' key 'discipline'"),
    ([{"strategy": "s", "a": {"category": ""}}], "row 0: 'a.category' must be non-empty text"),
    ([{"strategy": "s", "min_overlap_mm": True}], "row 0: 'min_overlap_mm' must be a finite number"),
    ([{"strategy": "s", "max_overlap_mm": math.inf}], "row 0: 'max_overlap_mm' must be a finite number"),
    ([{"strategy": "s", "max_overlap_mm": "50"}], "row 0: 'max_overlap_mm' must be a finite number"),
    ([{"strategy": "s", "min_overlap_mm": 50, "max_overlap_mm": 50}], "row 0: 'min_overlap_mm' must be below"),
])
def test_bad_table_is_rejected_naming_the_row(table, fragment):
    with pytest.raises(ValueError) as err:
        build_clash_register([_clash("p", "q")], table)
    assert fragment in str(err.value)


def test_valid_band_passes_validation():
    reg = build_clash_register([_clash("p", "q", overlap=30)],
                               [{"strategy": "s", "min_overlap_mm": 10, "max_overlap_mm": 50.5}])
    assert _only(reg)["strategy"] == "s"


# --- block hook ---------------------------------------------------------------

def _block(config=None):
    from app.blocks.bim_extractor import BIMExtractorBlock
    block = BIMExtractorBlock()
    if config:
        block.config.update(config)
    return block


def _report(**extra):
    return {"clash_count": 1, "clashes": [_clash("p", "q")],
            "detection_method": "aabb_intersection", **extra}


def test_hook_puts_register_first_and_keeps_detection():
    out = _block()._with_clash_register(_report(), {"clash_strategy_table": [{"strategy": "s"}]})
    assert list(out)[0] == "register"
    assert out["register"]["by_strategy"] == {"s": 1}
    assert out["register"]["complete"] is True
    assert out["clashes"] == [_clash("p", "q")]


@pytest.mark.parametrize("extra", [{"pair_cap_reached": True}, {"timed_out": True}])
def test_hook_marks_a_cut_detection_incomplete(extra):
    out = _block()._with_clash_register(_report(**extra), {})
    assert out["register"]["complete"] is False
    assert out["register"]["table_given"] is False


def test_hook_reports_a_bad_table_and_keeps_the_clashes():
    out = _block()._with_clash_register(_report(), {"clash_strategy_table": [{"stratgy": "s"}]})
    assert "row 0" in out["register"]["error"]
    assert out["register"]["complete"] is True
    assert out["clash_count"] == 1


@pytest.mark.parametrize("params,expected", [
    ({}, "cfg-q"),
    ({"clash_strategy_table": [{"strategy": "param-q"}]}, "param-q"),
])
def test_hook_reads_the_table_from_params_then_config(params, expected):
    block = _block({"clash_strategy_table": [{"strategy": "cfg-q"}]})
    out = block._with_clash_register(_report(), params)
    assert out["register"]["rows"][0]["strategy"] == expected
