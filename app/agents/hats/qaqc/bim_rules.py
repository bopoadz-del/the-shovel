"""BIM validation rule engine over extracted elements (S-K3 item 2).

The IDS checker (S-K3 item 1) asks what each element must carry on its
own. This engine asks what elements must satisfy *together* or between
two of their own fields, the model checks a BIM coordinator runs before
issue:

* ``unique``     - a field's value occurs once (optionally once per group,
  e.g. once per storey or per type);
* ``consistent`` - every element of a group carries the same value of a
  field (e.g. one value per type name);
* ``compare``    - one numeric field of an element stands in a stated
  relation (``<``, ``<=``, ``==``, ``!=``, ``>=``, ``>``) to another.

Nothing here names a property set, an entity, a field or a value. Every
rule comes from the caller's table (a project's own BIM execution plan or
the user's request); an empty or malformed table is refused, never filled
with a starter set. The one check that runs without a table is GlobalId
uniqueness, which the IFC schema itself requires of every rooted entity.

Elements are the dicts produced by the ``bim_extractor`` block: ``id``,
``ifc_type``, ``category``, ``name``, ``description``, ``object_type``,
``materials``, lower-case quantity keys and ``properties`` keyed
``"Pset.Prop"``.
"""
from __future__ import annotations

import operator
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

RULE_KINDS = ("unique", "consistent", "compare")

_OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}

_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class BimRuleError(ValueError):
    """The supplied rule table cannot be checked as written."""


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not _NUMBER.fullmatch(text):
        return None
    return float(text)


def field_value(element: Mapping[str, Any], field: str) -> Any:
    """Read ``field`` off an extracted element; ``None`` when it is absent.

    A top-level key wins (attributes, quantities); ``materials`` reads as
    the joined list; anything else is a ``"Pset.Prop"`` property, matched
    exactly first and then without regard to case.
    """
    if field == "materials":
        mats = element.get("materials") or []
        joined = ", ".join(str(m) for m in mats if str(m).strip())
        return joined or None
    if field in element and not isinstance(element[field], (dict, list)):
        value = element[field]
    else:
        props = element.get("properties") or {}
        if field in props:
            value = props[field]
        else:
            wanted = field.casefold()
            value = next(
                (v for k, v in props.items() if str(k).casefold() == wanted),
                None,
            )
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


def _key(value: Any) -> Any:
    """Comparison key: numbers by value, text as written (trimmed)."""
    num = _as_number(value)
    if num is not None:
        return num
    return str(value).strip()


def _types(applies_to: Any, where: str) -> Optional[set]:
    if applies_to is None:
        return None
    if not isinstance(applies_to, Mapping):
        raise BimRuleError(f"{where}: applies_to must be an object")
    raw = applies_to.get("ifc_type")
    if raw is None:
        return None
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, (list, tuple)) or not names:
        raise BimRuleError(f"{where}: applies_to.ifc_type must name a type")
    return {str(n).strip().casefold() for n in names if str(n).strip()}


def _compile_rule(rule: Any, index: int) -> Dict[str, Any]:
    where = f"rule {index + 1}"
    if not isinstance(rule, Mapping):
        raise BimRuleError(f"{where}: a rule must be an object")
    kind = str(rule.get("kind", "")).strip().casefold()
    if kind not in RULE_KINDS:
        raise BimRuleError(f"{where}: kind must be one of {', '.join(RULE_KINDS)}")
    field = str(rule.get("field") or "").strip()
    if not field:
        raise BimRuleError(f"{where}: field is required")
    group_by = str(rule.get("group_by") or "").strip() or None
    if kind == "consistent" and group_by is None:
        raise BimRuleError(f"{where}: a consistent rule needs group_by")
    compiled: Dict[str, Any] = {
        "name": str(rule.get("name") or f"{kind} {field}"),
        "kind": kind,
        "field": field,
        "group_by": group_by,
        "types": _types(rule.get("applies_to"), where),
    }
    if kind == "compare":
        op = str(rule.get("op") or "").strip()
        if op not in _OPERATORS:
            raise BimRuleError(f"{where}: op must be one of {' '.join(_OPERATORS)}")
        other = str(rule.get("other") or "").strip()
        if not other:
            raise BimRuleError(f"{where}: a compare rule needs other")
        compiled["op"] = op
        compiled["other"] = other
    return compiled


def compile_rules(rules: Any) -> List[Dict[str, Any]]:
    """Validate the whole table before any element is read."""
    if isinstance(rules, Mapping):
        rules = rules.get("rules", [rules] if "kind" in rules else None)
    if not isinstance(rules, (list, tuple)) or not rules:
        raise BimRuleError("no rules supplied; a rule table is required")
    return [_compile_rule(rule, i) for i, rule in enumerate(rules)]


def _in_scope(elements: Sequence[Mapping[str, Any]], types: Optional[set]) -> List[Mapping[str, Any]]:
    if types is None:
        return list(elements)
    return [e for e in elements if str(e.get("ifc_type", "")).casefold() in types]


def _group_of(element: Mapping[str, Any], group_by: Optional[str]) -> Any:
    if group_by is None:
        return None
    value = field_value(element, group_by)
    return None if value is None else _key(value)


def _check_unique(rule: Dict[str, Any], scope: Sequence[Mapping[str, Any]]):
    seen: Dict[Any, List[str]] = {}
    not_checked: List[str] = []
    for el in scope:
        value = field_value(el, rule["field"])
        if value is None or (rule["group_by"] and _group_of(el, rule["group_by"]) is None):
            not_checked.append(str(el.get("id", "")))
            continue
        seen.setdefault((_group_of(el, rule["group_by"]), _key(value)), []).append(str(el.get("id", "")))
    violations = [
        {"group": group, "value": value, "elements": ids}
        for (group, value), ids in seen.items()
        if len(ids) > 1
    ]
    checked = sum(len(ids) for ids in seen.values())
    return checked, not_checked, violations


def _check_consistent(rule: Dict[str, Any], scope: Sequence[Mapping[str, Any]]):
    groups: Dict[Any, Dict[Any, List[str]]] = {}
    not_checked: List[str] = []
    for el in scope:
        group = _group_of(el, rule["group_by"])
        value = field_value(el, rule["field"])
        if group is None or value is None:
            not_checked.append(str(el.get("id", "")))
            continue
        groups.setdefault(group, {}).setdefault(_key(value), []).append(str(el.get("id", "")))
    violations = [
        {"group": group, "values": {str(v): ids for v, ids in by_value.items()}}
        for group, by_value in groups.items()
        if len(by_value) > 1
    ]
    checked = sum(len(ids) for by_value in groups.values() for ids in by_value.values())
    return checked, not_checked, violations


def _check_compare(rule: Dict[str, Any], scope: Sequence[Mapping[str, Any]]):
    relation = _OPERATORS[rule["op"]]
    checked = 0
    not_checked: List[str] = []
    violations: List[Dict[str, Any]] = []
    for el in scope:
        left = _as_number(field_value(el, rule["field"]))
        right = _as_number(field_value(el, rule["other"]))
        if left is None or right is None:
            not_checked.append(str(el.get("id", "")))
            continue
        checked += 1
        if not relation(left, right):
            violations.append({
                "element": str(el.get("id", "")),
                rule["field"]: left,
                rule["other"]: right,
            })
    return checked, not_checked, violations


def duplicate_ids(elements: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """GlobalIds that occur more than once (the IFC schema forbids it)."""
    counts: Dict[str, int] = {}
    for el in elements:
        gid = str(el.get("id") or "").strip()
        if gid:
            counts[gid] = counts.get(gid, 0) + 1
    return {gid: n for gid, n in counts.items() if n > 1}


def check_bim_rules(elements: Sequence[Mapping[str, Any]], rules: Any) -> Dict[str, Any]:
    """Apply a rule table to extracted elements.

    A rule is ``fail`` when it finds a violation, ``pass`` when it read at
    least one element and found none, ``not_applicable`` when no element
    in its scope carried the fields it reads. The overall status is
    ``fail`` if any rule fails or the model repeats a GlobalId, ``pass``
    if at least one rule passed, otherwise ``not_applicable`` (nothing in
    the model could be read against the table).
    """
    compiled = compile_rules(rules)
    results: List[Dict[str, Any]] = []
    for rule in compiled:
        scope = _in_scope(elements, rule["types"])
        if rule["kind"] == "unique":
            checked, not_checked, violations = _check_unique(rule, scope)
        elif rule["kind"] == "consistent":
            checked, not_checked, violations = _check_consistent(rule, scope)
        else:
            checked, not_checked, violations = _check_compare(rule, scope)
        if violations:
            status = "fail"
        elif checked:
            status = "pass"
        else:
            status = "not_applicable"
        results.append({
            "name": rule["name"],
            "kind": rule["kind"],
            "status": status,
            "in_scope": len(scope),
            "checked": checked,
            "not_checked": not_checked,
            "violations": violations,
        })
    dupes = duplicate_ids(elements)
    if dupes or any(r["status"] == "fail" for r in results):
        overall = "fail"
    elif any(r["status"] == "pass" for r in results):
        overall = "pass"
    else:
        overall = "not_applicable"
    return {
        "status": overall,
        "elements_read": len(elements),
        "duplicate_ids": dupes,
        "rules": results,
    }
