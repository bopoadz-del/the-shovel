"""S-K3 item 5: clash triage downstream of the AABB check.

Every test builds its own clash list. Each decision is pinned by two inputs
that must come out differently.
"""
import pytest

from app.blocks.clash_triage import decide_move, parse_move_priority, triage_clashes


def _c(a, b, cat_a, cat_b, overlap=50.0, type_a=None, type_b=None):
    return {
        "type": "aabb_overlap", "element_a": a, "element_b": b,
        "category_a": cat_a, "category_b": cat_b,
        "ifc_type_a": type_a, "ifc_type_b": type_b, "overlap_mm": overlap,
    }


# --- cause pattern: shared elements form one group -------------------------

def test_clashes_sharing_an_element_form_one_group_with_that_element_named():
    clashes = [_c("D1", f"B{i}", "ducts", "beams") for i in range(4)]
    t = triage_clashes(clashes)
    assert len(t["pattern_groups"]) == 1
    g = t["pattern_groups"][0]
    assert (g["clash_count"], g["element_count"]) == (4, 5)
    assert g["common_element"] == "D1"
    assert g["common_element_clashes"] == 4
    assert t["isolated_clashes"] == 0


def test_clashes_with_no_shared_element_are_isolated_not_a_group():
    clashes = [_c(f"D{i}", f"B{i}", "ducts", "beams") for i in range(4)]
    t = triage_clashes(clashes)
    assert t["pattern_groups"] == []
    assert t["isolated_clashes"] == 4


def test_tied_leaders_are_reported_not_picked():
    # a triangle: every element sits in two clashes
    clashes = [_c("P1", "W1", "pipes", "walls"), _c("W1", "B1", "walls", "beams"),
               _c("B1", "P1", "beams", "pipes")]
    g = triage_clashes(clashes)["pattern_groups"][0]
    assert g["common_element"] is None
    assert g["tied_elements"] == ["B1", "P1", "W1"]


def test_chain_has_a_unique_middle_element():
    clashes = [_c("P1", "W1", "pipes", "walls"), _c("W1", "B1", "walls", "beams")]
    g = triage_clashes(clashes)["pattern_groups"][0]
    assert g["common_element"] == "W1"
    assert g["common_element_type"] == "walls"
    assert g["tied_elements"] == []


def test_groups_order_by_size_then_deepest_overlap():
    small_deep = [_c("X1", "Y1", "pipes", "beams", 900.0), _c("X1", "Y2", "pipes", "beams", 10.0)]
    big_shallow = [_c("Z1", f"Q{i}", "ducts", "slabs", 20.0) for i in range(3)]
    t = triage_clashes(small_deep + big_shallow)
    assert [g["clash_count"] for g in t["pattern_groups"]] == [3, 2]
    same_size = [_c("A1", "A2", "pipes", "beams", 5.0), _c("A1", "A3", "pipes", "beams", 5.0),
                 _c("C1", "C2", "ducts", "beams", 300.0), _c("C1", "C3", "ducts", "beams", 1.0)]
    t2 = triage_clashes(same_size)
    assert [g["common_element"] for g in t2["pattern_groups"]] == ["C1", "A1"]
    assert t2["pattern_groups"][0]["max_overlap_mm"] == 300.0


# --- discipline pairs --------------------------------------------------------

def test_pairs_are_unordered_and_keep_the_deepest_overlap():
    clashes = [_c("D1", "B1", "ducts", "beams", 40.0), _c("B2", "D2", "beams", "ducts", 75.5),
               _c("P1", "W1", "pipes", "walls", 12.0)]
    pairs = triage_clashes(clashes)["by_discipline_pair"]
    assert pairs[0] == {"pair": ["beams", "ducts"], "clash_count": 2, "max_overlap_mm": 75.5}
    assert pairs[1] == {"pair": ["pipes", "walls"], "clash_count": 1, "max_overlap_mm": 12.0}


def test_pair_without_overlap_figure_reports_none_not_zero():
    clashes = [_c("D1", "B1", "ducts", "beams", None), _c("D2", "B2", "ducts", "beams", True)]
    assert triage_clashes(clashes)["by_discipline_pair"][0]["max_overlap_mm"] is None
    with_figure = [_c("D1", "B1", "ducts", "beams", 0.0)]
    assert triage_clashes(with_figure)["by_discipline_pair"][0]["max_overlap_mm"] == 0.0


# --- who moves comes only from the caller's table ---------------------------

def test_no_table_decides_nothing():
    c = _c("D1", "B1", "ducts", "beams")
    t = triage_clashes([c])
    assert c["moves"] is None and c["moves_reason"] == "no move-priority table supplied"
    assert (t["moves_decided"], t["move_table_supplied"]) == (0, False)


def test_higher_rank_moves_and_the_table_decides_direction():
    table = [{"category": "beams", "rank": 1}, {"category": "ducts", "rank": 3}]
    c = _c("D1", "B1", "ducts", "beams")
    triage_clashes([c], table)
    assert c["moves"] == "D1"
    flipped = [{"category": "beams", "rank": 5}, {"category": "ducts", "rank": 3}]
    c2 = _c("D1", "B1", "ducts", "beams")
    t = triage_clashes([c2], flipped)
    assert c2["moves"] == "B1"
    assert t["moves_decided"] == 1


def test_ifc_type_row_outranks_category_row():
    table = [{"category": "pipes", "rank": 4}, {"ifc_type": "IfcPipeSegment", "rank": 0},
             {"category": "walls", "rank": 2}]
    c = _c("P1", "W1", "pipes", "walls", type_a="IfcPipeSegment", type_b="IfcWall")
    assert decide_move(c, parse_move_priority(table))[0] == "W1"
    c_other_type = _c("P1", "W1", "pipes", "walls", type_a="IfcFlowSegment", type_b="IfcWall")
    assert decide_move(c_other_type, parse_move_priority(table))[0] == "P1"


def test_unranked_side_and_equal_ranks_are_undecided_with_reason():
    table = parse_move_priority([{"category": "beams", "rank": 1}, {"category": "ducts", "rank": 1}])
    mover, reason = decide_move(_c("D1", "B1", "ducts", "beams"), table)
    assert mover is None and reason == "both sides have rank 1"
    mover, reason = decide_move(_c("P1", "B1", "pipes", "beams"), table)
    assert mover is None and reason == "no rank in the table for pipes"


@pytest.mark.parametrize("bad,needle", [
    ([], "non-empty"),
    ("beams=1", "non-empty"),
    ([{"category": "beams", "rank": 1}, "ducts"], "row 1 is not an object"),
    ([{"rank": 1}], "row 0 must name exactly one"),
    ([{"category": "beams", "ifc_type": "IfcBeam", "rank": 1}], "row 0 must name exactly one"),
    ([{"category": " ", "rank": 1}], "row 0: category must be text"),
    ([{"category": "beams", "rank": True}], "row 0: rank must be an integer"),
    ([{"category": "beams", "rank": 1.5}], "row 0: rank must be an integer"),
    ([{"category": "beams", "rank": 1}, {"category": "beams", "rank": 2}], "row 1: category 'beams' already has rank 1"),
])
def test_bad_table_is_rejected_whole_naming_the_row(bad, needle):
    with pytest.raises(ValueError) as e:
        parse_move_priority(bad)
    assert needle in str(e.value)


def test_repeated_identical_row_is_accepted():
    ranks = parse_move_priority([{"category": "beams", "rank": 1}, {"category": "beams", "rank": 1}])
    assert ranks == {("category", "beams"): 1}


def test_clash_missing_an_element_id_is_not_triaged():
    partial = {"type": "aabb_overlap", "element_a": "D1", "category_a": "ducts"}
    t = triage_clashes([partial, _c("D2", "B2", "ducts", "beams")])
    assert t["clashes_triaged"] == 1
    assert "moves" not in partial


# --- block: triage leads the report; a bad table keeps the detection --------

def _block():
    from app.blocks.bim_extractor import BIMExtractorBlock
    return BIMExtractorBlock()


def test_block_puts_triage_first_and_marks_a_capped_list_incomplete():
    report = {"clash_count": 2, "clashes": [_c("D1", "B1", "ducts", "beams"), _c("D1", "B2", "ducts", "beams")],
              "pair_cap_reached": False, "timed_out": False}
    out = _block()._with_clash_triage(report, {})
    assert list(out)[0] == "triage"
    assert out["triage"]["complete"] is True
    assert out["triage"]["pattern_groups"][0]["common_element"] == "D1"
    capped = dict(report, pair_cap_reached=True)
    assert _block()._with_clash_triage(capped, {})["triage"]["complete"] is False
    timed = dict(report, timed_out=True)
    assert _block()._with_clash_triage(timed, {})["triage"]["complete"] is False


def test_block_reads_move_table_from_params_and_keeps_detection_on_bad_table():
    good = {"clashes": [_c("D1", "B1", "ducts", "beams")]}
    out = _block()._with_clash_triage(
        good, {"clash_move_priority": [{"category": "beams", "rank": 1}, {"category": "ducts", "rank": 2}]})
    assert out["clashes"][0]["moves"] == "D1"
    bad = {"clashes": [_c("D1", "B1", "ducts", "beams")], "clash_count": 1}
    out_bad = _block()._with_clash_triage(bad, {"clash_move_priority": [{"rank": 1}]})
    assert "row 0" in out_bad["triage"]["error"]
    assert out_bad["clash_count"] == 1 and len(out_bad["clashes"]) == 1


# --- container: geometric clashes keep both disciplines ---------------------

def _normalize(clash):
    from app.containers.construction import ConstructionContainer
    return ConstructionContainer._normalize_block_clash(None, clash, "m.ifc")


def test_container_files_geometric_clash_under_both_side_categories():
    n = _normalize(dict(_c("D1", "B1", "ducts", "beams", 33.0), moves="D1", moves_reason="r"))
    assert n["involved_disciplines"] == ["ducts", "beams"]
    assert (n["overlap_mm"], n["moves"]) == (33.0, "D1")
    legacy = _normalize({"type": "name_duplicate", "element_a": "x", "element_b": "y", "category": "walls"})
    assert legacy["involved_disciplines"] == ["walls"]
    same = _normalize(_c("W1", "W2", "walls", "walls"))
    assert same["involved_disciplines"] == ["walls"]
