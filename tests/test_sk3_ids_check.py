"""S-K3 item 1: IDS-style requirement checker.

Every test builds its own elements and its own specification; each
decision is shown with an input on each side of it.
"""
import pytest

from app.agents.hats.qaqc.ids_check import IdsSpecError, check_ids


def _el(eid, ifc_type, props=None, **attrs):
    el = {"id": eid, "ifc_type": ifc_type, "name": attrs.pop("name", eid)}
    if props is not None:
        el["properties"] = props
    el.update(attrs)
    return el


def _spec(requirements, applicability=None, **extra):
    spec = {
        "name": extra.pop("name", "spec"),
        "applicability": applicability or [{"entity": "IfcDoor"}],
        "requirements": requirements,
    }
    spec.update(extra)
    return [spec]


def _only(result):
    assert result["status"] == "success"
    (spec,) = result["specifications"]
    return spec


# -- required ---------------------------------------------------------------

def test_required_property_present_passes_and_missing_fails():
    els = [
        _el("d1", "IfcDoor", {"Psx.Rating": "R1"}),
        _el("d2", "IfcDoor", {"Psx.Other": "x"}),
    ]
    out = _only(check_ids(els, _spec([{"property": "Psx.Rating"}])))
    assert out["passed"] == 1 and out["failed"] == 1
    assert out["failures"][0]["id"] == "d2"
    assert "missing" in out["failures"][0]["reasons"][0]
    assert out["status"] == "fail"


def test_required_with_value_checks_the_value():
    els = [
        _el("a", "IfcDoor", {"Psx.Rating": "R1"}),
        _el("b", "IfcDoor", {"Psx.Rating": "R2"}),
    ]
    out = _only(check_ids(els, _spec([{"property": "Psx.Rating", "value": "r1"}])))
    assert [f["id"] for f in out["failures"]] == ["b"]
    assert out["passed"] == 1


def test_blank_value_counts_as_absent():
    els = [_el("a", "IfcDoor", {"Psx.Rating": ""}), _el("b", "IfcDoor", {"Psx.Rating": "R"})]
    out = _only(check_ids(els, _spec([{"property": "Psx.Rating"}])))
    assert [f["id"] for f in out["failures"]] == ["a"]


# -- optional ---------------------------------------------------------------

def test_optional_absent_passes_but_present_must_conform():
    els = [
        _el("absent", "IfcDoor", {}),
        _el("bad", "IfcDoor", {"Psx.Width": 700}),
        _el("good", "IfcDoor", {"Psx.Width": 900}),
    ]
    req = [{"property": "Psx.Width", "cardinality": "optional", "min": 800}]
    out = _only(check_ids(els, _spec(req)))
    assert [f["id"] for f in out["failures"]] == ["bad"]
    assert out["passed"] == 2


# -- prohibited -------------------------------------------------------------

def test_prohibited_without_value_fails_on_presence():
    els = [_el("p", "IfcDoor", {"Psx.Temp": "1"}), _el("q", "IfcDoor", {})]
    req = [{"property": "Psx.Temp", "cardinality": "prohibited"}]
    out = _only(check_ids(els, _spec(req)))
    assert [f["id"] for f in out["failures"]] == ["p"]


def test_prohibited_with_value_fails_only_on_that_value():
    els = [
        _el("p", "IfcDoor", {"Psx.Status": "Draft"}),
        _el("q", "IfcDoor", {"Psx.Status": "Issued"}),
    ]
    req = [{"property": "Psx.Status", "cardinality": "prohibited", "value": "draft"}]
    out = _only(check_ids(els, _spec(req)))
    assert [f["id"] for f in out["failures"]] == ["p"]


# -- value constraints ------------------------------------------------------

def test_pattern_is_a_full_match():
    els = [_el("a", "IfcDoor", {"Psx.Code": "D-101"}), _el("b", "IfcDoor", {"Psx.Code": "XD-101"})]
    out = _only(check_ids(els, _spec([{"property": "Psx.Code", "pattern": r"D-\d+"}])))
    assert [f["id"] for f in out["failures"]] == ["b"]


def test_enumeration_is_case_insensitive_membership():
    els = [_el("a", "IfcDoor", {"Psx.Kind": "Single"}), _el("b", "IfcDoor", {"Psx.Kind": "Triple"})]
    req = [{"property": "Psx.Kind", "enumeration": ["single", "double"]}]
    out = _only(check_ids(els, _spec(req)))
    assert [f["id"] for f in out["failures"]] == ["b"]


def test_bounds_are_inclusive_and_refuse_non_numbers():
    els = [
        _el("lo", "IfcDoor", {"Psx.W": 800}),
        _el("hi", "IfcDoor", {"Psx.W": 1200.5}),
        _el("txt", "IfcDoor", {"Psx.W": "wide"}),
        _el("ok", "IfcDoor", {"Psx.W": "1000"}),
    ]
    req = [{"property": "Psx.W", "min": 800, "max": 1200}]
    out = _only(check_ids(els, _spec(req)))
    assert sorted(f["id"] for f in out["failures"]) == ["hi", "txt"]


def test_numeric_value_compares_as_number():
    els = [_el("a", "IfcDoor", {"Psx.N": "2.0"}), _el("b", "IfcDoor", {"Psx.N": 3})]
    out = _only(check_ids(els, _spec([{"property": "Psx.N", "value": 2}])))
    assert [f["id"] for f in out["failures"]] == ["b"]


# -- applicability ----------------------------------------------------------

def test_applicability_selects_by_entity_and_property():
    els = [
        _el("w1", "IfcWall", {"Psx.External": True}),
        _el("w2", "IfcWall", {"Psx.External": False}),
        _el("d1", "IfcDoor", {"Psx.External": True}),
    ]
    app = [{"entity": "IfcWall"}, {"property": "Psx.External", "value": True}]
    out = _only(check_ids(els, _spec([{"property": "Psx.U"}], applicability=app)))
    assert out["applicable"] == 1
    assert [f["id"] for f in out["failures"]] == ["w1"]


def test_attribute_and_material_facets():
    els = [
        _el("a", "IfcSlab", {}, object_type="FL-01", materials=["Concrete"]),
        _el("b", "IfcSlab", {}, object_type="", materials=[]),
    ]
    req = [{"attribute": "object_type"}, {"material": True}]
    out = _only(check_ids(els, _spec(req, applicability=[{"entity": "IfcSlab"}])))
    assert [f["id"] for f in out["failures"]] == ["b"]
    assert len(out["failures"][0]["reasons"]) == 2


# -- specification cardinality ----------------------------------------------

def test_required_specification_fails_on_empty_scope_optional_does_not():
    els = [_el("w", "IfcWall", {})]
    req = [{"property": "Psx.A"}]
    required = _only(check_ids(els, _spec(req, cardinality="required")))
    optional = _only(check_ids(els, _spec(req)))
    assert required["status"] == "fail" and required["scope_issue"]
    assert optional["status"] == "pass" and optional["scope_issue"] is None


def test_prohibited_specification_fails_when_anything_is_in_scope():
    req = [{"property": "Psx.A", "cardinality": "optional"}]
    hit = _only(check_ids([_el("d", "IfcDoor", {})], _spec(req, cardinality="prohibited")))
    miss = _only(check_ids([_el("w", "IfcWall", {})], _spec(req, cardinality="prohibited")))
    assert hit["status"] == "fail"
    assert miss["status"] == "pass"


# -- unreadable elements ----------------------------------------------------

def test_unreadable_element_is_not_checked_not_failed():
    els = [_el("u", "IfcDoor"), _el("r", "IfcDoor")]
    req = [{"property": "Psx.A"}]
    marked = _only(check_ids(els, _spec(req), unreadable_ids=["u"]))
    unmarked = _only(check_ids(els, _spec(req)))
    assert marked["not_checked"] == ["u"] and marked["failed"] == 1
    assert unmarked["not_checked"] == [] and unmarked["failed"] == 2


def test_incomplete_status_when_only_unreadable_elements_remain():
    els = [_el("u", "IfcDoor"), _el("r", "IfcDoor", {"Psx.A": 1})]
    out = _only(check_ids(els, _spec([{"property": "Psx.A"}]), unreadable_ids=["u"]))
    assert out["status"] == "incomplete"


# -- refusals: no starter set, no silent repair -------------------------------

@pytest.mark.parametrize("specs", [[], None, {}])
def test_no_specification_is_refused(specs):
    with pytest.raises(IdsSpecError):
        check_ids([_el("d", "IfcDoor", {})], specs)


@pytest.mark.parametrize("facet,needle", [
    ({"property": "NoDot"}, "PropertySet.Property"),
    ({"property": "P.A", "cardinality": "sometimes"}, "cardinality"),
    ({"property": "P.A", "pattern": "("}, "pattern"),
    ({"property": "P.A", "min": 5, "max": 1}, "min is greater"),
    ({"property": "P.A", "min": "lots"}, "number"),
    ({"property": "P.A", "enumeration": []}, "enumeration"),
    ({"attribute": "colour"}, "attribute"),
    ({"property": "P.A", "attribute": "name"}, "exactly one"),
])
def test_malformed_requirement_is_refused(facet, needle):
    with pytest.raises(IdsSpecError, match=needle):
        check_ids([_el("d", "IfcDoor", {})], _spec([facet]))


def test_empty_requirements_or_applicability_are_refused():
    with pytest.raises(IdsSpecError, match="requirements"):
        check_ids([], [{"applicability": [{"entity": "IfcDoor"}], "requirements": []}])
    with pytest.raises(IdsSpecError, match="applicability"):
        check_ids([], [{"applicability": [], "requirements": [{"property": "P.A"}]}])


def test_summary_counts_each_status():
    els = [_el("d", "IfcDoor", {"Psx.A": 1}), _el("w", "IfcWall", {})]
    specs = [
        {"name": "doors", "applicability": [{"entity": "IfcDoor"}],
         "requirements": [{"property": "Psx.A"}]},
        {"name": "walls", "applicability": [{"entity": "IfcWall"}],
         "requirements": [{"property": "Psx.A"}]},
    ]
    out = check_ids(els, specs)
    assert out["summary"] == {
        "specifications": 2, "passing": 1, "failing": 1, "incomplete": 0, "elements": 2,
    }


def test_entity_facet_is_constrained_by_the_entity_it_names():
    els = [_el("d", "IfcDoor", {}), _el("w", "IfcWall", {})]
    doors = _only(check_ids(els, _spec([{"property": "P.A"}], applicability=[{"entity": "ifcdoor"}])))
    walls = _only(check_ids(els, _spec([{"property": "P.A"}], applicability=[{"entity": "IfcWall"}])))
    assert [f["id"] for f in doors["failures"]] == ["d"]
    assert [f["id"] for f in walls["failures"]] == ["w"]


def test_named_material_must_match_any_material_does_not_care():
    els = [
        _el("c", "IfcSlab", {}, materials=["Concrete"]),
        _el("s", "IfcSlab", {}, materials=["Steel"]),
    ]
    app = [{"entity": "IfcSlab"}]
    named = _only(check_ids(els, _spec([{"material": "concrete"}], applicability=app)))
    anything = _only(check_ids(els, _spec([{"material": True}], applicability=app)))
    assert [f["id"] for f in named["failures"]] == ["s"]
    assert anything["failed"] == 0


def test_number_parsing_refuses_text_with_digits():
    els = [_el("a", "IfcDoor", {"P.W": "900mm"}), _el("b", "IfcDoor", {"P.W": "9e2"})]
    out = _only(check_ids(els, _spec([{"property": "P.W", "min": 800}])))
    assert [f["id"] for f in out["failures"]] == ["a"]
