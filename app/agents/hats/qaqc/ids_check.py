"""IDS-style requirement checker over extracted BIM elements (S-K3 item 1).

The method follows the buildingSMART Information Delivery Specification:
a specification selects the elements it applies to (its applicability)
and states what each of them must carry (its requirements). Every
requirement has a cardinality:

* ``required``   - the facet must be present and, if a value constraint is
  given, satisfy it;
* ``optional``   - the facet may be absent; if present it must satisfy the
  value constraint;
* ``prohibited`` - the facet must be absent; with a value constraint, it
  must not carry a value that satisfies it.

The specification itself carries a cardinality too: ``required`` means
at least one element must be in scope, ``prohibited`` means none may be,
``optional`` (the default) means scope may be empty.

Nothing here knows a property set, an entity or a value. Every name and
constraint comes from the specification the caller supplies (a project's
own requirement table); an empty or malformed specification is refused,
never filled with a starter set. Elements are the dicts produced by the
``bim_extractor`` block: ``id``, ``ifc_type``, ``name``, ``description``,
``object_type``, ``materials`` and ``properties`` keyed ``"Pset.Prop"``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

CARDINALITIES = ("required", "optional", "prohibited")

# Element attributes a facet may name directly (bim_extractor keys).
ATTRIBUTE_KEYS = ("name", "description", "object_type")


_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class IdsSpecError(ValueError):
    """The supplied specification cannot be checked as written."""


def _norm(text: Any) -> str:
    return str(text).strip().casefold()


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not _NUMBER.fullmatch(text):
        return None
    return float(text)


def _compile_constraint(facet: Mapping[str, Any], where: str) -> Dict[str, Any]:
    """Validate a facet's value constraint once, before any element is read."""
    out: Dict[str, Any] = {}
    if "value" in facet:
        out["value"] = facet["value"]
    if "enumeration" in facet:
        enum = facet["enumeration"]
        if not isinstance(enum, (list, tuple)) or not enum:
            raise IdsSpecError(f"{where}: enumeration must be a non-empty list")
        out["enumeration"] = list(enum)
    if "pattern" in facet:
        try:
            out["pattern"] = re.compile(str(facet["pattern"]))
        except re.error as exc:
            raise IdsSpecError(f"{where}: pattern does not compile: {exc}") from exc
    for bound in ("min", "max"):
        if bound in facet:
            num = _as_number(facet[bound])
            if num is None:
                raise IdsSpecError(f"{where}: {bound} must be a number")
            out[bound] = num
    if "min" in out and "max" in out and out["min"] > out["max"]:
        raise IdsSpecError(f"{where}: min is greater than max")
    return out


def _satisfies(value: Any, constraint: Mapping[str, Any]) -> bool:
    """True when ``value`` meets every constraint stated (none stated: True)."""
    if "value" in constraint:
        want = constraint["value"]
        a, b = _as_number(value), _as_number(want)
        if a is not None and b is not None:
            if a != b:
                return False
        elif _norm(value) != _norm(want):
            return False
    if "enumeration" in constraint:
        if _norm(value) not in {_norm(v) for v in constraint["enumeration"]}:
            return False
    if "pattern" in constraint:
        if not constraint["pattern"].fullmatch(str(value)):
            return False
    if "min" in constraint or "max" in constraint:
        num = _as_number(value)
        if num is None:
            return False
        if "min" in constraint and num < constraint["min"]:
            return False
        if "max" in constraint and num > constraint["max"]:
            return False
    return True


def _facet_values(element: Mapping[str, Any], facet: Mapping[str, Any]) -> List[Any]:
    """Values the element carries for the facet; empty when absent."""
    if "property" in facet:
        props = element.get("properties") or {}
        want = _norm(facet["property"])
        return [v for k, v in props.items() if _norm(k) == want and v not in (None, "")]
    if "attribute" in facet:
        value = element.get(str(facet["attribute"]).strip().lower())
        return [] if value in (None, "") else [value]
    if "material" in facet:
        return [m for m in (element.get("materials") or []) if m not in (None, "")]
    if "entity" in facet:
        return [element.get("ifc_type")] if element.get("ifc_type") else []
    return []


def _facet_label(facet: Mapping[str, Any]) -> str:
    for kind in ("property", "attribute", "material", "entity"):
        if kind in facet:
            return f"{kind} {facet[kind]}" if kind != "material" else "material"
    return "facet"


def _compile_facet(facet: Any, where: str) -> Dict[str, Any]:
    if not isinstance(facet, Mapping):
        raise IdsSpecError(f"{where}: a facet must be an object")
    kinds = [k for k in ("property", "attribute", "material", "entity") if k in facet]
    if len(kinds) != 1:
        raise IdsSpecError(
            f"{where}: name exactly one of property, attribute, material, entity"
        )
    kind = kinds[0]
    if kind == "property" and "." not in str(facet["property"]):
        raise IdsSpecError(f"{where}: a property is named as PropertySet.Property")
    if kind == "attribute" and str(facet["attribute"]).strip().lower() not in ATTRIBUTE_KEYS:
        raise IdsSpecError(
            f"{where}: attribute must be one of {', '.join(ATTRIBUTE_KEYS)}"
        )
    card = _norm(facet.get("cardinality", "required"))
    if card not in CARDINALITIES:
        raise IdsSpecError(f"{where}: cardinality must be one of {', '.join(CARDINALITIES)}")
    constraint = _compile_constraint(facet, where)
    # An entity facet is constrained by the entity it names; a material
    # facet that names a material is constrained by that name. ``True``
    # for a material means "any material".
    if kind == "entity" and "value" not in constraint:
        constraint["value"] = facet["entity"]
    if kind == "material" and isinstance(facet["material"], str) and "value" not in constraint:
        constraint["value"] = facet["material"]
    return {
        "raw": dict(facet),
        "label": _facet_label(facet),
        "cardinality": card,
        "constraint": constraint,
    }


def _compile_spec(spec: Any, index: int) -> Dict[str, Any]:
    where = f"specification {index + 1}"
    if not isinstance(spec, Mapping):
        raise IdsSpecError(f"{where}: must be an object")
    name = str(spec.get("name") or where)
    applicability = spec.get("applicability")
    if isinstance(applicability, Mapping):
        applicability = [applicability]
    if not isinstance(applicability, (list, tuple)) or not applicability:
        raise IdsSpecError(f"{name}: applicability names no facet")
    requirements = spec.get("requirements")
    if not isinstance(requirements, (list, tuple)) or not requirements:
        raise IdsSpecError(f"{name}: requirements are empty")
    card = _norm(spec.get("cardinality", "optional"))
    if card not in CARDINALITIES:
        raise IdsSpecError(f"{name}: cardinality must be one of {', '.join(CARDINALITIES)}")
    return {
        "name": name,
        "cardinality": card,
        "applicability": [
            _compile_facet(f, f"{name} applicability {i + 1}")
            for i, f in enumerate(applicability)
        ],
        "requirements": [
            _compile_facet(f, f"{name} requirement {i + 1}")
            for i, f in enumerate(requirements)
        ],
    }


def _applies(element: Mapping[str, Any], facets: Sequence[Mapping[str, Any]]) -> bool:
    for facet in facets:
        values = _facet_values(element, facet["raw"])
        if not values or not any(_satisfies(v, facet["constraint"]) for v in values):
            return False
    return True


def _check_requirement(
    element: Mapping[str, Any], facet: Mapping[str, Any]
) -> Optional[str]:
    """None when the element meets the requirement, else the reason it fails."""
    values = _facet_values(element, facet["raw"])
    constraint = facet["constraint"]
    card = facet["cardinality"]
    meeting = [v for v in values if _satisfies(v, constraint)]
    if card == "prohibited":
        if constraint:
            if meeting:
                return f"{facet['label']} carries a prohibited value {meeting[0]!r}"
            return None
        if values:
            return f"{facet['label']} is present but prohibited"
        return None
    if not values:
        return f"{facet['label']} is missing" if card == "required" else None
    if not meeting:
        return f"{facet['label']} value {values[0]!r} does not meet the requirement"
    return None


def check_ids(
    elements: Iterable[Mapping[str, Any]],
    specifications: Any,
    *,
    unreadable_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """Check every element against every specification.

    ``unreadable_ids`` are elements whose property sets could not be read
    (``bim_extractor``'s ``psets_failed``). They are listed as not checked,
    never as passing or failing, because their properties are unknown.
    Raises :class:`IdsSpecError` when the specification is empty or invalid.
    """
    if isinstance(specifications, Mapping):
        specifications = specifications.get("specifications", [specifications])
    if not isinstance(specifications, (list, tuple)) or not specifications:
        raise IdsSpecError("no specification was supplied; nothing to check against")
    compiled = [_compile_spec(s, i) for i, s in enumerate(specifications)]
    unreadable: Set[str] = {str(x) for x in unreadable_ids}
    pool = [e for e in elements if isinstance(e, Mapping)]

    results: List[Dict[str, Any]] = []
    for spec in compiled:
        in_scope = [e for e in pool if _applies(e, spec["applicability"])]
        passed: List[str] = []
        failures: List[Dict[str, Any]] = []
        not_checked: List[str] = []
        for element in in_scope:
            eid = str(element.get("id", ""))
            if eid in unreadable:
                not_checked.append(eid)
                continue
            reasons = [
                r for r in (_check_requirement(element, f) for f in spec["requirements"]) if r
            ]
            if reasons:
                failures.append({
                    "id": eid,
                    "ifc_type": element.get("ifc_type", ""),
                    "name": element.get("name", ""),
                    "reasons": reasons,
                })
            else:
                passed.append(eid)
        scope_issue = None
        if spec["cardinality"] == "required" and not in_scope:
            scope_issue = "no element is in scope, but the specification is required"
        elif spec["cardinality"] == "prohibited" and in_scope:
            scope_issue = f"{len(in_scope)} element(s) in scope, but the specification is prohibited"
        if scope_issue or failures:
            status = "fail"
        elif not_checked:
            status = "incomplete"
        else:
            status = "pass"
        results.append({
            "name": spec["name"],
            "cardinality": spec["cardinality"],
            "status": status,
            "applicable": len(in_scope),
            "passed": len(passed),
            "failed": len(failures),
            "not_checked": not_checked,
            "scope_issue": scope_issue,
            "failures": failures,
        })
    return {
        "status": "success",
        "specifications": results,
        "summary": {
            "specifications": len(results),
            "passing": sum(1 for r in results if r["status"] == "pass"),
            "failing": sum(1 for r in results if r["status"] == "fail"),
            "incomplete": sum(1 for r in results if r["status"] == "incomplete"),
            "elements": len(pool),
        },
    }
