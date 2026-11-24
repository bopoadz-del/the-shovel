"""S-K3 item 5: clash triage downstream of the AABB check.

Every test builds its own clash list. Each decision is pinned by at least two
parametrised inputs that come out differently.
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

@pytest.mark.parametrize("clashes, groups, isolated", [
    # one duct through four beams: one group led by the duct
    ([_c("D1", f"B{i}", "ducts", "beams") for i in range(4)],
     [(4, 5, "D1", 4, [])], 0),
    # four unrelated pairs: no group
    ([_c(f"D{i}", f"B{i}", "ducts", "beams") for i in range(4)], [], 4),
    # triangle: every element in two clashes, so the tie is reported, not picked
    ([_c("P1", "W1", "pipes", "walls"), _c("W1", "B1", "walls", "beams"),
      _c("B1", "P1", "beams", "pipes")],
     [(3, 3, None, 2, ["B1", "P1", "W1"])], 0),
    # chain: the middle element is the unique leader
    ([_c("P1", "W1", "pipes", "walls"), _c("W1", "B1", "walls", "beams"),
      _c("X1", "Y1", "pipes", "slabs")],
     [(2, 3, "W1", 2, [])], 1),
])
def test_shared_elements_group_and_name_the_leader(clashes, groups, isolated):
    t = triage_clashes(clashes)
    got = [(g["clash_count"], g["element_count"], g["common_element"],
            g["common_element_clashes"], g["tied_elements"]) for g in t["pattern_groups"]]
    assert got == groups
    assert t["isolated_clashes"] == isolated


@pytest.mark.parametrize("clashes, leaders, top_overlap", [
    # size first: the bigger shallow group leads the smaller deep one
    ([_c("X1", "Y1", "pipes", "beams", 900.0), _c("X1", "Y2", "pipes", "beams", 10.0)]
     + [_c("Z1", f"Q{i}", "ducts", "slabs", 20.0) for i in range(3)],
     ["Z1", "X1"], 20.0),
    # equal size: the deeper overlap leads
    ([_c("A1", "A2", "pipes", "beams", 5.0), _c("A1", "A3", "pipes", "beams", 5.0),
      _c("C1", "C2", "ducts", "beams", 300.0), _c("C1", "C3", "ducts", "beams", 1.0)],
     ["C1", "A1"], 300.0),
])
def test_groups_order_by_size_then_deepest_overlap(clashes, leaders, top_overlap):
    t = triage_clashes(clashes)
    assert [g["common_element"] for g in t["pattern_groups"]] == leaders
    assert t["pattern_groups"][0]["max_overlap_mm"] == top_overlap


@pytest.mark.parametrize("clashes, leader_type", [
    ([_c("P1", "W1", "pipes", "walls", type_b="IfcWall"), _c("W1", "B1", "walls", "beams", type_a="IfcWall")],
     "IfcWall"),
    ([_c("P1", "W1", "pipes", "walls"), _c("W1", "B1", "walls", "beams")], "walls"),
])
def test_leader_type_prefers_ifc_class_then_category(clashes, leader_type):
    assert triage_clashes(clashes)["pattern_groups"][0]["common_element_type"] == leader_type


# --- discipline pairs --------------------------------------------------------

@pytest.mark.parametrize("clashes, pairs", [
    ([_c("D1", "B1", "ducts", "beams", 40.0), _c("B2", "D2", "beams", "ducts", 75.5),
      _c("P1", "W1", "pipes", "walls", 12.0)],
     [(["beams", "ducts"], 2, 75.5), (["pipes", "walls"], 1, 12.0)]),
    # no usable figure is None, not 0; a real 0.0 stays 0.0
    ([_c("D1", "B1", "ducts", "beams", None), _c("D2", "B2", "ducts", "beams", True)],
     [(["beams", "ducts"], 2, None)]),
    ([_c("D1", "B1", "ducts", "beams", 0.0), _c("Q1", "Q2", None, "slabs", 3)],
     [(["beams", "ducts"], 1, 0.0), (["slabs", "unknown"], 1, 3.0)]),
])
def test_pairs_are_unordered_and_keep_the_deepest_overlap(clashes, pairs):
    got = [(p["pair"], p["clash_count"], p["max_overlap_mm"])
           for p in triage_clashes(clashes)["by_discipline_pair"]]
    assert got == pairs


# --- who moves comes only from the caller's table ---------------------------

@pytest.mark.parametrize("table, clash, mover, reason_part", [
    (None, _c("D1", "B1", "ducts", "beams"), None, "no move-priority table supplied"),
    ([{"category": "beams", "rank": 1}, {"category": "ducts", "rank": 3}],
     _c("D1", "B1", "ducts", "beams"), "D1", "rank 3 (ducts) moves for rank 1 (beams)"),
    ([{"category": "beams", "rank": 5}, {"category": "ducts", "rank": 3}],
     _c("D1", "B1", "ducts", "beams"), "B1", "rank 5 (beams) moves for rank 3 (ducts)"),
    ([{"category": "beams", "rank": 1}, {"category": "ducts", "rank": 1}],
     _c("D1", "B1", "ducts", "beams"), None, "both sides have rank 1"),
    ([{"category": "beams", "rank": 1}],
     _c("P1", "B1", "pipes", "beams"), None, "no rank in the table for pipes"),
])
def test_move_is_decided_by_the_table_only(table, clash, mover, reason_part):
    t = triage_clashes([clash], table)
    assert clash["moves"] == mover
    assert reason_part in clash["moves_reason"]
    assert t["moves_decided"] == (1 if mover else 0)
    assert t["move_table_supplied"] is (table is not None)


@pytest.mark.parametrize("type_a, mover", [
    ("IfcPipeSegment", "W1"),   # the ifc_type row (rank 0) holds the pipe
    ("IfcFlowSegment", "P1"),   # no ifc_type row: the category row (rank 4) moves it
])
def test_ifc_type_row_outranks_category_row(type_a, mover):
    table = parse_move_priority([
        {"category": "pipes", "rank": 4}, {"ifc_type": "IfcPipeSegment", "rank": 0},
        {"category": "walls", "rank": 2}])
    c = _c("P1", "W1", "pipes", "walls", type_a=type_a, type_b="IfcWall")
    assert decide_move(c, table)[0] == mover


@pytest.mark.parametrize("bad,needle", [
    ([], "non-empty"),
    ("beams=1", "non-empty"),
    ([{"category": "beams", "rank": 1}, "ducts"], "row 1 is not an object"),
    ([{"rank": 1}], "row 0 must name exactly one"),
    ([{"category": "beams", "ifc_type": "IfcBeam", "rank": 1}], "row 0 must name exactly one"),
    ([{"category": " ", "rank": 1}], "row 0: category must be text"),
    ([{"category": "beams", "rank": True}], "row 0: rank must be an integer"),
    ([{"category": "beams", "rank": 1.5}], "row 0: rank must be an integer"),
    ([{"category": "beams", "rank": 1}, {"category": "beams", "rank": 2}],
     "row 1: category 'beams' already has rank 1"),
])
def test_bad_table_is_rejected_whole_naming_the_row(bad, needle):
    with pytest.raises(ValueError) as e:
        parse_move_priority(bad)
    assert needle in str(e.value)


@pytest.mark.parametrize("table, ranks", [
    ([{"category": "beams", "rank": 1}, {"category": "beams", "rank": 1}],
     {("category", "beams"): 1}),
    ([{"ifc_type": " IfcBeam ", "rank": -2}, {"category": "beams", "rank": 7}],
     {("ifc_type", "IfcBeam"): -2, ("category", "beams"): 7}),
])
def test_valid_table_parses(table, ranks):
    assert parse_move_priority(table) == ranks


@pytest.mark.parametrize("clashes, triaged", [
    ([{"type": "aabb_overlap", "element_a": "D1", "category_a": "ducts"},
      _c("D2", "B2", "ducts", "beams")], 1),
    ([{"type": "aabb_overlap", "element_a": "", "element_b": "B9"},
      _c("D2", "B2", "ducts", "beams"), _c("D3", "B3", "ducts", "beams")], 2),
])
def test_clash_missing_an_element_id_is_not_triaged(clashes, triaged):
    t = triage_clashes(clashes)
    assert t["clashes_triaged"] == triaged
    assert "moves" not in clashes[0]


# --- block: triage leads the report; a bad table keeps the detection --------

def _block():
    from app.blocks.bim_extractor import BIMExtractorBlock
    return BIMExtractorBlock()


@pytest.mark.parametrize("cap, timed_out, complete", [
    (False, False, True),
    (True, False, False),
    (False, True, False),
])
def test_block_puts_triage_first_and_marks_a_cut_list_incomplete(cap, timed_out, complete):
    report = {"clash_count": 2,
              "clashes": [_c("D1", "B1", "ducts", "beams"), _c("D1", "B2", "ducts", "beams")],
              "pair_cap_reached": cap, "timed_out": timed_out}
    out = _block()._with_clash_triage(report, {})
    assert next(iter(out)) == "triage"
    assert out["triage"]["complete"] is complete
    assert out["triage"]["pattern_groups"][0]["common_element"] == "D1"


@pytest.mark.parametrize("table, mover, error_part", [
    ([{"category": "beams", "rank": 1}, {"category": "ducts", "rank": 2}], "D1", None),
    ([{"category": "beams", "rank": 2}, {"category": "ducts", "rank": 1}], "B1", None),
    ([{"rank": 1}], None, "row 0"),
])
def test_block_reads_move_table_from_params_and_keeps_detection_on_bad_table(table, mover, error_part):
    report = {"clashes": [_c("D1", "B1", "ducts", "beams")], "clash_count": 1}
    out = _block()._with_clash_triage(report, {"clash_move_priority": table})
    assert out["clash_count"] == 1 and len(out["clashes"]) == 1
    if error_part:
        assert error_part in out["triage"]["error"]
        assert "moves" not in out["clashes"][0]
    else:
        assert "error" not in out["triage"]
        assert out["clashes"][0]["moves"] == mover


# --- container: geometric clashes keep both disciplines ---------------------

@pytest.mark.parametrize("clash, disciplines, overlap, moves", [
    (dict(_c("D1", "B1", "ducts", "beams", 33.0), moves="D1", moves_reason="r"),
     ["ducts", "beams"], 33.0, "D1"),
    ({"type": "name_duplicate", "element_a": "x", "element_b": "y", "category": "walls"},
     ["walls"], None, None),
    (_c("W1", "W2", "walls", "walls", 4.0), ["walls"], 4.0, None),
])
def test_container_files_geometric_clash_under_both_side_categories(clash, disciplines, overlap, moves):
    from app.containers.construction import ConstructionContainer
    n = ConstructionContainer._normalize_block_clash(None, clash, "m.ifc")
    assert n["involved_disciplines"] == disciplines
    assert (n["overlap_mm"], n["moves"]) == (overlap, moves)


class _StubBlock:
    def __init__(self, reports):
        self.reports = reports

    async def process(self, input_data, params):
        return {"status": "success", "element_count": 3,
                "clash_report": self.reports[input_data["file_path"]]}


@pytest.mark.parametrize("reports, expected", [
    ({"a.ifc": {"triage": {"clashes_triaged": 1}, "clashes": [_c("D1", "B1", "ducts", "beams")]},
      "b.ifc": {"clashes": []}},
     {"a.ifc": {"clashes_triaged": 1}}),
    ({"a.ifc": {"clashes": []},
      "b.ifc": {"triage": {"clashes_triaged": 0}, "clashes": []}},
     {"b.ifc": {"clashes_triaged": 0}}),
])
def test_container_reports_each_models_triage(reports, expected):
    import asyncio
    from app.containers.construction import ConstructionContainer
    c = ConstructionContainer.__new__(ConstructionContainer)
    c._get_bim_extractor_block = lambda: _StubBlock(reports)
    out = asyncio.run(c.bim_clash_detection({}, {"discipline_models": ["a.ifc", "b.ifc"]}))
    assert out["status"] == "success"
    assert out["triage_by_model"] == expected
