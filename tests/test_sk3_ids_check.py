"""S-K3 item 1: IDS-style requirement checker.

Every test builds its own elements and its own specification; each
decision is parametrized with an input on each side of it.
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


def _door_verdict(props, requirement):
    """Check one door against one requirement: 'pass' or 'fail'."""
    out = _only(check_ids([_el("d", "IfcDoor", props)], _spec([requirement])))
    assert out["applicable"] == 1
    return "fail" if out["failed"] else "pass"


# -- required ---------------------------------------------------------------

@pytest.mark.parametrize("props,expected", [
    ({"Psx.Rating": "R1"}, "pass"),
    ({"Psx.Other": "x"}, "fail"),
])
def test_required_property_present_passes_and_missing_fails(props, expected):
    assert _door_verdict(props, {"property": "Psx.Rating"}) == expected


@pytest.mark.parametrize("requirement,reason", [
    ({"property": "Psx.Rating"}, "property Psx.Rating is missing"),
    ({"attribute": "object_type"}, "attribute object_type is missing"),
    ({"material": True}, "material is missing"),
])
def test_failure_reason_names_the_facet(requirement, reason):
    out = _only(check_ids([_el("d", "IfcDoor", {})], _spec([requirement])))
    assert out["failures"][0]["reasons"] == [reason]


@pytest.mark.parametrize("rating,expected", [("R1", "pass"), ("r1", "pass"), ("R2", "fail")])
def test_required_with_value_checks_the_value(rating, expected):
    req = {"property": "Psx.Rating", "value": "r1"}
    assert _door_verdict({"Psx.Rating": rating}, req) == expected


@pytest.mark.parametrize("rating,expected", [("", "fail"), (None, "fail"), ("R", "pass")])
def test_blank_value_counts_as_absent(rating, expected):
    assert _door_verdict({"Psx.Rating": rating}, {"property": "Psx.Rating"}) == expected


# -- optional ---------------------------------------------------------------

@pytest.mark.parametrize("props,expected", [
    ({}, "pass"),
    ({"Psx.Width": 700}, "fail"),
    ({"Psx.Width": 900}, "pass"),
])
def test_optional_absent_passes_but_present_must_conform(props, expected):
    req = {"property": "Psx.Width", "cardinality": "optional", "min": 800}
    assert _door_verdict(props, req) == expected


# -- prohibited -------------------------------------------------------------

@pytest.mark.parametrize("props,expected", [({"Psx.Temp": "1"}, "fail"), ({}, "pass")])
def test_prohibited_without_value_fails_on_presence(props, expected):
    req = {"property": "Psx.Temp", "cardinality": "prohibited"}
    assert _door_verdict(props, req) == expected


@pytest.mark.parametrize("status,expected", [("Draft", "fail"), ("Issued", "pass")])
def test_prohibited_with_value_fails_only_on_that_value(status, expected):
    req = {"property": "Psx.Status", "cardinality": "prohibited", "value": "draft"}
    assert _door_verdict({"Psx.Status": status}, req) == expected


# -- value constraints ------------------------------------------------------

@pytest.mark.parametrize("code,expected", [("D-101", "pass"), ("XD-101", "fail"), ("D-101a", "fail")])
def test_pattern_is_a_full_match(code, expected):
    assert _door_verdict({"Psx.Code": code}, {"property": "Psx.Code", "pattern": r"D-\d+"}) == expected


@pytest.mark.parametrize("kind,expected", [("Single", "pass"), ("DOUBLE", "pass"), ("Triple", "fail")])
def test_enumeration_is_case_insensitive_membership(kind, expected):
    req = {"property": "Psx.Kind", "enumeration": ["single", "double"]}
    assert _door_verdict({"Psx.Kind": kind}, req) == expected


@pytest.mark.parametrize("width,expected", [
    (800, "pass"),
    (1200, "pass"),
    ("1000", "pass"),
    (799.9, "fail"),
    (1200.5, "fail"),
    ("wide", "fail"),
])
def test_bounds_are_inclusive_and_refuse_non_numbers(width, expected):
    req = {"property": "Psx.W", "min": 800, "max": 1200}
    assert _door_verdict({"Psx.W": width}, req) == expected


@pytest.mark.parametrize("flag,expected", [(True, "fail"), (False, "fail"), (1, "pass")])
def test_a_boolean_is_not_a_number_for_a_bound(flag, expected):
    # Python counts True as 1; an IDS bound reads only a written number.
    assert _door_verdict({"Psx.N": flag}, {"property": "Psx.N", "min": 0, "max": 1}) == expected


@pytest.mark.parametrize("n,expected", [("2.0", "pass"), (2, "pass"), (3, "fail"), ("2.5", "fail")])
def test_numeric_value_compares_as_number(n, expected):
    assert _door_verdict({"Psx.N": n}, {"property": "Psx.N", "value": 2}) == expected


@pytest.mark.parametrize("width,expected", [("900mm", "fail"), ("9e2", "pass"), ("1,000", "fail")])
def test_number_parsing_refuses_text_with_digits(width, expected):
    assert _door_verdict({"P.W": width}, {"property": "P.W", "min": 800}) == expected


# -- applicability ----------------------------------------------------------

@pytest.mark.parametrize("element,in_scope", [
    (_el("w1", "IfcWall", {"Psx.External": True}), 1),
    (_el("w2", "IfcWall", {"Psx.External": False}), 0),
    (_el("d1", "IfcDoor", {"Psx.External": True}), 0),
])
def test_applicability_selects_by_entity_and_property(element, in_scope):
    app = [{"entity": "IfcWall"}, {"property": "Psx.External", "value": True}]
    out = _only(check_ids([element], _spec([{"property": "Psx.U"}], applicability=app)))
    assert out["applicable"] == in_scope


@pytest.mark.parametrize("entity,failed_ids", [
    ("ifcdoor", ["d"]),
    ("IfcWall", ["w"]),
])
def test_entity_facet_is_constrained_by_the_entity_it_names(entity, failed_ids):
    els = [_el("d", "IfcDoor", {}), _el("w", "IfcWall", {})]
    out = _only(check_ids(els, _spec([{"property": "P.A"}], applicability=[{"entity": entity}])))
    assert [f["id"] for f in out["failures"]] == failed_ids


@pytest.mark.parametrize("attrs,reasons", [
    ({"object_type": "FL-01", "materials": ["Concrete"]}, 0),
    ({"object_type": "", "materials": []}, 2),
    ({"object_type": "FL-01", "materials": []}, 1),
])
def test_attribute_and_material_facets(attrs, reasons):
    req = [{"attribute": "object_type"}, {"material": True}]
    out = _only(check_ids([_el("s", "IfcSlab", {}, **attrs)],
                          _spec(req, applicability=[{"entity": "IfcSlab"}])))
    got = out["failures"][0]["reasons"] if out["failures"] else []
    assert len(got) == reasons


@pytest.mark.parametrize("material,requirement,expected", [
    ("Concrete", "concrete", "pass"),
    ("Steel", "concrete", "fail"),
    ("Steel", True, "pass"),
])
def test_named_material_must_match_any_material_does_not_care(material, requirement, expected):
    els = [_el("s", "IfcSlab", {}, materials=[material])]
    out = _only(check_ids(els, _spec([{"material": requirement}], applicability=[{"entity": "IfcSlab"}])))
    assert ("fail" if out["failed"] else "pass") == expected


# -- specification cardinality ----------------------------------------------

def test_required_specification_fails_on_empty_scope_optional_does_not():
    els = [_el("w", "IfcWall", {})]
    req = [{"property": "Psx.A"}]
    required = _only(check_ids(els, _spec(req, cardinality="required")))
    optional = _only(check_ids(els, _spec(req)))
    assert required["status"] == "fail" and required["scope_issue"]
    assert optional["status"] == "pass" and optional["scope_issue"] is None


@pytest.mark.parametrize("element,status", [
    (_el("d", "IfcDoor", {}), "fail"),
    (_el("w", "IfcWall", {}), "pass"),
])
def test_prohibited_specification_fails_when_anything_is_in_scope(element, status):
    req = [{"property": "Psx.A", "cardinality": "optional"}]
    out = _only(check_ids([element], _spec(req, cardinality="prohibited")))
    assert out["status"] == status


# -- unreadable elements ----------------------------------------------------

def test_unreadable_element_is_not_checked_not_failed():
    els = [_el("u", "IfcDoor"), _el("r", "IfcDoor")]
    req = [{"property": "Psx.A"}]
    marked = _only(check_ids(els, _spec(req), unreadable_ids=["u"]))
    unmarked = _only(check_ids(els, _spec(req)))
    assert marked["not_checked"] == ["u"] and marked["failed"] == 1
    assert unmarked["not_checked"] == [] and unmarked["failed"] == 2


@pytest.mark.parametrize("unreadable,status", [(["u"], "incomplete"), ([], "fail")])
def test_incomplete_status_when_only_unreadable_elements_remain(unreadable, status):
    els = [_el("u", "IfcDoor"), _el("r", "IfcDoor", {"Psx.A": 1})]
    out = _only(check_ids(els, _spec([{"property": "Psx.A"}]), unreadable_ids=unreadable))
    assert out["status"] == status


# -- the specification's own shape ------------------------------------------

_ONE_SPEC = {"name": "doors", "applicability": [{"entity": "IfcDoor"}],
             "requirements": [{"property": "Psx.A"}]}


@pytest.mark.parametrize("specs", [
    _ONE_SPEC,
    {"specifications": [_ONE_SPEC]},
    [_ONE_SPEC],
])
def test_a_single_specification_may_be_passed_bare_or_wrapped(specs):
    out = _only(check_ids([_el("d", "IfcDoor", {"Psx.A": 1})], specs))
    assert out["name"] == "doors"
    assert out["status"] == "pass"


# -- refusals: no starter set, no silent repair -------------------------------

@pytest.mark.parametrize("specs", [[], None, {}])
def test_no_specification_is_refused(specs):
    with pytest.raises(IdsSpecError):
        check_ids([_el("d", "IfcDoor", {})], specs)


@pytest.mark.parametrize("specs", [["not a spec"], [7]])
def test_a_specification_that_is_not_an_object_is_refused(specs):
    with pytest.raises(IdsSpecError, match="must be an object"):
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
    ("Psx.A", "a facet must be an object"),
    (["property"], "a facet must be an object"),
])
def test_malformed_requirement_is_refused(facet, needle):
    with pytest.raises(IdsSpecError, match=needle):
        check_ids([_el("d", "IfcDoor", {})], _spec([facet]))


@pytest.mark.parametrize("spec,needle", [
    ({"applicability": [{"entity": "IfcDoor"}], "requirements": []}, "requirements"),
    ({"applicability": [], "requirements": [{"property": "P.A"}]}, "applicability"),
    ({"applicability": [{"entity": "IfcDoor"}], "requirements": [{"property": "P.A"}],
      "cardinality": "often"}, "cardinality"),
])
def test_empty_requirements_or_applicability_are_refused(spec, needle):
    with pytest.raises(IdsSpecError, match=needle):
        check_ids([], [spec])


@pytest.mark.parametrize("door_props,summary", [
    ({"Psx.A": 1}, {"specifications": 2, "passing": 1, "failing": 1, "incomplete": 0, "elements": 2}),
    ({}, {"specifications": 2, "passing": 0, "failing": 2, "incomplete": 0, "elements": 2}),
])
def test_summary_counts_each_status(door_props, summary):
    els = [_el("d", "IfcDoor", door_props), _el("w", "IfcWall", {})]
    specs = [
        {"name": "doors", "applicability": [{"entity": "IfcDoor"}],
         "requirements": [{"property": "Psx.A"}]},
        {"name": "walls", "applicability": [{"entity": "IfcWall"}],
         "requirements": [{"property": "Psx.A"}]},
    ]
    assert check_ids(els, specs)["summary"] == summary
