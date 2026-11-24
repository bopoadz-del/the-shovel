"""S-K3 item 2: BIM validation rule engine over extracted elements.

Every element and rule is built here; none is read from a project.
"""
from __future__ import annotations

import pytest

from app.agents.hats.qaqc.bim_rules import (
    BimRuleError,
    check_bim_rules,
    compile_rules,
    duplicate_ids,
    field_value,
)


def _el(gid, ifc_type="IfcWall", **kw):
    props = kw.pop("props", None)
    el = {"id": gid, "ifc_type": ifc_type, "name": kw.pop("name", ""), **kw}
    if props is not None:
        el["properties"] = props
    return el


# ---- field_value -----------------------------------------------------------

@pytest.mark.parametrize("element,field,expected", [
    (_el("a", name="W-1"), "name", "W-1"),
    (_el("a", name="  "), "name", None),
    (_el("a", props={"Px.Mark": "M1"}), "Px.Mark", "M1"),
    (_el("a", props={"Px.Mark": "M1"}), "px.mark", "M1"),
    (_el("a", props={"Px.Mark": "M1"}), "Px.Other", None),
    (_el("a", materials=["Steel", "Glass"]), "materials", "Steel, Glass"),
    (_el("a", materials=[]), "materials", None),
    (_el("a", height=3.0), "height", 3.0),
])
def test_field_value_reads_attribute_property_material(element, field, expected):
    assert field_value(element, field) == expected


# ---- refusal ---------------------------------------------------------------

@pytest.mark.parametrize("rules,message", [
    ([], "rule table is required"),
    (None, "rule table is required"),
    ({"rules": []}, "rule table is required"),
    (["unique name"], "a rule must be an object"),
    ([7], "a rule must be an object"),
    ([{"kind": "sum", "field": "name"}], "kind must be one of"),
    ([{"kind": "unique"}], "field is required"),
    ([{"kind": "consistent", "field": "Px.R"}], "needs group_by"),
    ([{"kind": "compare", "field": "a", "other": "b", "op": "=<"}], "op must be one of"),
    ([{"kind": "compare", "field": "a", "op": "<"}], "needs other"),
    ([{"kind": "unique", "field": "a", "applies_to": "IfcWall"}], "applies_to must be an object"),
    ([{"kind": "unique", "field": "a", "applies_to": {"ifc_type": []}}], "must name a type"),
])
def test_malformed_table_is_refused(rules, message):
    with pytest.raises(BimRuleError, match=message):
        compile_rules(rules)


@pytest.mark.parametrize("rules", [
    {"kind": "unique", "field": "name"},
    {"rules": [{"kind": "unique", "field": "name"}]},
    [{"kind": "unique", "field": "name"}],
])
def test_one_rule_bare_wrapped_or_listed_reads_the_same(rules):
    out = check_bim_rules([_el("a", name="X"), _el("b", name="Y")], rules)
    assert out["status"] == "pass"
    assert [r["name"] for r in out["rules"]] == ["unique name"]


# ---- unique ----------------------------------------------------------------

@pytest.mark.parametrize("marks,status,dup_ids", [
    (["D1", "D2", "D3"], "pass", []),
    (["D1", "D2", "D1"], "fail", ["a", "c"]),
    (["7", "7.0", "8"], "fail", ["a", "b"]),
])
def test_unique_finds_repeated_value(marks, status, dup_ids):
    els = [_el(g, props={"Px.Mark": m}) for g, m in zip("abc", marks)]
    out = check_bim_rules(els, [{"kind": "unique", "field": "Px.Mark"}])
    rule = out["rules"][0]
    assert rule["status"] == status
    assert [v["elements"] for v in rule["violations"]] == ([dup_ids] if dup_ids else [])


@pytest.mark.parametrize("storeys,status", [
    (["L1", "L2"], "pass"),
    (["L1", "L1"], "fail"),
])
def test_unique_within_group(storeys, status):
    els = [
        _el("a", props={"Px.Mark": "M", "Px.Storey": storeys[0]}),
        _el("b", props={"Px.Mark": "M", "Px.Storey": storeys[1]}),
    ]
    out = check_bim_rules(els, [{"kind": "unique", "field": "Px.Mark", "group_by": "Px.Storey"}])
    assert out["rules"][0]["status"] == status


@pytest.mark.parametrize("second,checked,not_checked", [
    ({"Px.Mark": "M2"}, 2, []),
    ({}, 1, ["b"]),
])
def test_unique_missing_value_is_not_checked_not_a_fault(second, checked, not_checked):
    els = [_el("a", props={"Px.Mark": "M1"}), _el("b", props=second)]
    rule = check_bim_rules(els, [{"kind": "unique", "field": "Px.Mark"}])["rules"][0]
    assert rule["status"] == "pass"
    assert (rule["checked"], rule["not_checked"]) == (checked, not_checked)


@pytest.mark.parametrize("group,not_checked", [
    ("L1", []),
    (None, ["b"]),
])
def test_unique_element_without_group_is_not_checked(group, not_checked):
    second = {"Px.Mark": "M2"} if group is None else {"Px.Mark": "M2", "Px.Storey": group}
    els = [_el("a", props={"Px.Mark": "M1", "Px.Storey": "L1"}), _el("b", props=second)]
    rule = check_bim_rules(
        els, [{"kind": "unique", "field": "Px.Mark", "group_by": "Px.Storey"}]
    )["rules"][0]
    assert rule["not_checked"] == not_checked


# ---- consistent ------------------------------------------------------------

@pytest.mark.parametrize("ratings,status", [
    (["60", "60"], "pass"),
    (["60", "90"], "fail"),
    (["60", "60.0"], "pass"),
])
def test_consistent_within_type(ratings, status):
    els = [_el(g, object_type="T1", props={"Px.Rating": r}) for g, r in zip("ab", ratings)]
    out = check_bim_rules(
        els, [{"kind": "consistent", "field": "Px.Rating", "group_by": "object_type"}]
    )
    assert out["rules"][0]["status"] == status
    assert out["status"] == status


@pytest.mark.parametrize("types,status", [
    (["T1", "T2"], "pass"),
    (["T1", "T1"], "fail"),
])
def test_consistent_compares_only_inside_a_group(types, status):
    els = [
        _el("a", object_type=types[0], props={"Px.Rating": "60"}),
        _el("b", object_type=types[1], props={"Px.Rating": "90"}),
    ]
    rule = check_bim_rules(
        els, [{"kind": "consistent", "field": "Px.Rating", "group_by": "object_type"}]
    )["rules"][0]
    assert rule["status"] == status


@pytest.mark.parametrize("ratings,expected", [
    (["60", "90", "60"], {"60.0": ["a", "c"], "90.0": ["b"]}),
    (["EI30", "EI30", "EI60"], {"EI30": ["a", "b"], "EI60": ["c"]}),
])
def test_consistent_violation_lists_each_value_with_its_elements(ratings, expected):
    els = [_el(g, object_type="T1", props={"Px.Rating": r}) for g, r in zip("abc", ratings)]
    rule = check_bim_rules(
        els, [{"kind": "consistent", "field": "Px.Rating", "group_by": "object_type"}]
    )["rules"][0]
    assert rule["violations"] == [{"group": "T1", "values": expected}]


# ---- compare ---------------------------------------------------------------

@pytest.mark.parametrize("op,left,right,status", [
    ("<=", 2.0, 3.0, "pass"),
    ("<=", 3.0, 3.0, "pass"),
    ("<=", 3.5, 3.0, "fail"),
    ("<", 3.0, 3.0, "fail"),
    (">", 4.0, 3.0, "pass"),
    (">=", 2.9, 3.0, "fail"),
    ("==", 3, "3.0", "pass"),
    ("!=", 3, "3.0", "fail"),
])
def test_compare_two_fields_of_one_element(op, left, right, status):
    els = [_el("a", props={"Px.A": left, "Px.B": right})]
    rule = check_bim_rules(
        els, [{"kind": "compare", "field": "Px.A", "op": op, "other": "Px.B"}]
    )["rules"][0]
    assert rule["status"] == status


@pytest.mark.parametrize("right,status,not_checked", [
    ("wide", "not_applicable", ["a"]),
    (5, "pass", []),
])
def test_compare_non_number_is_not_checked(right, status, not_checked):
    els = [_el("a", props={"Px.A": 1, "Px.B": right})]
    rule = check_bim_rules(
        els, [{"kind": "compare", "field": "Px.A", "op": "<", "other": "Px.B"}]
    )["rules"][0]
    assert (rule["status"], rule["not_checked"]) == (status, not_checked)


@pytest.mark.parametrize("left,right", [
    (9, 3),
    ("4.5", "1"),
])
def test_compare_violation_names_both_values(left, right):
    els = [_el("a", props={"Px.A": left, "Px.B": right})]
    rule = check_bim_rules(
        els, [{"kind": "compare", "field": "Px.A", "op": "<", "other": "Px.B"}]
    )["rules"][0]
    assert rule["violations"] == [{"element": "a", "Px.A": float(left), "Px.B": float(right)}]


# ---- scope -----------------------------------------------------------------

@pytest.mark.parametrize("applies,in_scope,status", [
    ({"ifc_type": "IfcDoor"}, 2, "fail"),
    ({"ifc_type": ["ifcwall"]}, 1, "pass"),
    (None, 3, "fail"),
])
def test_applies_to_selects_by_ifc_type(applies, in_scope, status):
    els = [
        _el("a", "IfcDoor", name="X"),
        _el("b", "IfcDoor", name="X"),
        _el("c", "IfcWall", name="X"),
    ]
    rule = {"kind": "unique", "field": "name"}
    if applies is not None:
        rule["applies_to"] = applies
    out = check_bim_rules(els, [rule])["rules"][0]
    assert (out["in_scope"], out["status"]) == (in_scope, status)


# ---- model integrity and overall ------------------------------------------

@pytest.mark.parametrize("ids,expected", [
    (["a", "b", "c"], {}),
    (["a", "b", "a"], {"a": 2}),
    (["a", "", ""], {}),
])
def test_duplicate_global_ids(ids, expected):
    assert duplicate_ids([_el(g) for g in ids]) == expected


@pytest.mark.parametrize("ids,status", [
    (["a", "b"], "pass"),
    (["a", "a"], "fail"),
])
def test_repeated_global_id_fails_the_model_even_when_rules_pass(ids, status):
    els = [_el(ids[0], name="X"), _el(ids[1], name="Y")]
    out = check_bim_rules(els, [{"kind": "unique", "field": "name"}])
    assert out["rules"][0]["status"] == "pass"
    assert out["status"] == status


@pytest.mark.parametrize("props,status", [
    ({}, "not_applicable"),
    ({"Px.Mark": "M"}, "pass"),
])
def test_nothing_readable_is_not_applicable_not_pass(props, status):
    out = check_bim_rules([_el("a", props=props)], [{"kind": "unique", "field": "Px.Mark"}])
    assert out["status"] == status
    assert out["elements_read"] == 1
