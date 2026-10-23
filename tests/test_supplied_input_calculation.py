"""A calculation whose operands the user typed must run the registered calculator.

The same rule for every formula: when the ask names that formula and supplies
its required inputs, the calculator runs, the answer states its result, and
the answer and Sources credit it in plain words with those inputs. A percent
sign is hundredths. A default the user did not supply is the calculator's
default, not a figure from the project documents.
"""
from __future__ import annotations

import inspect
import math

import pytest

from app.agents.runtime import (
    _build_sources_from_audit,
    _forced_specific_tool,
    _message_is_formula_style_ask,
    _postprocess_answer,
    _wants_user_supplied_arithmetic_directive,
    derivation_mismatches,
)
from app.lib.construction_formulas import (
    _missing_required,
    describe_calculation_params,
    extract_calculation_params_from_text,
)
from app.lib.formula_registry import all_specs, parameters

DELAY = (
    "Calculate delay damages per day: contract amount 50,000,000, "
    "rate 0.1% per day."
)
SLAB = (
    "Calculate the volume of concrete for a slab 20 m long, 10 m wide "
    "and 0.25 m thick."
)
REBAR = "Calculate the area of rebar needed: 12 bars of 16 mm diameter."
LOOKUP = "What are the delay damages under this contract?"


def _scalar_number(param: inspect.Parameter) -> bool:
    ann = getattr(param.annotation, "__name__", "") or str(param.annotation)
    low = ann.lower()
    if any(tok in low for tok in ("str", "list", "dict", "bool", "tuple")):
        return False
    if param.default is not inspect.Parameter.empty and isinstance(
        param.default, (str, list, dict, tuple, bool)
    ):
        return False
    return True


def _in_declared_range(spec, name: str, value: float) -> float:
    """``value`` when the registry accepts it for ``name``, else a round figure inside that range."""
    declared = parameters(spec).get(name)
    if declared is None or declared.lo is None or declared.hi is None:
        return value
    lo, hi = declared.lo, declared.hi
    whole = declared.kind == "integer"
    if lo <= value <= hi and (not whole or float(value).is_integer()):
        return value
    floor = lo if lo > 0 else hi * 1e-3
    middle = math.sqrt(floor * hi) if floor > 0 else (lo + hi) / 2
    middle = float(f"{middle:.2g}")
    return float(max(math.ceil(lo), round(middle))) if whole else middle


def _phrase(spec, name: str, unit: str, index: int) -> str:
    if unit == "%" or name.endswith(("_percent", "_pct")):
        number, unit_txt = _in_declared_range(spec, name, 0.4), "%"
    elif "ratio" in name or name.endswith("_factor"):
        number, unit_txt = _in_declared_range(spec, name, 0.2), ""
    else:
        number = _in_declared_range(spec, name, 12 + index * 3)
        unit_txt = "" if unit in ("", "-", "currency", "ratio", "count", "no", "nr") else unit
    value = f"{int(number)}" if float(number).is_integer() else f"{number:g}"
    if len(name.replace("_", "")) == 1:
        return f"{name} = {value} {unit_txt}".strip()
    return f"{name} = {value} {unit_txt}".strip()


def _registry_cases() -> list[tuple[str, str, str]]:
    """One ask per formula the extractor can complete from that formula's name."""
    cases: list[tuple[str, str, str]] = []
    for spec in all_specs():
        rows = describe_calculation_params(spec.fn, name=spec.name)
        sig = inspect.signature(spec.fn)
        bits: list[str] = []
        numeric = 0
        blocked = False
        for index, row in enumerate(rows):
            if not row["required"]:
                continue
            param = sig.parameters[row["name"]]
            if row["name"] == "code":
                bits.append("ACI")
                continue
            if not _scalar_number(param):
                blocked = True
                break
            unit = (spec.inputs or {}).get(row["name"]) or row.get("unit") or ""
            bits.append(_phrase(spec, row["name"], unit, index))
            numeric += 1
        if blocked:
            continue
        if numeric == 0:
            for index, row in enumerate(rows):
                param = sig.parameters[row["name"]]
                if row["name"] == "code" or not _scalar_number(param):
                    continue
                unit = (spec.inputs or {}).get(row["name"]) or row.get("unit") or ""
                bits.append(_phrase(spec, row["name"], unit, index))
                numeric += 1
                break
        if numeric == 0:
            continue
        ask = f"Calculate {spec.display_name}: " + ", ".join(bits) + "."
        bound = extract_calculation_params_from_text(spec.fn, ask)
        if _missing_required(spec.fn, bound, spec.name):
            continue
        if not any(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in bound.values()
        ):
            continue
        cases.append((spec.name, ask, spec.owner))
    return cases


CASES = _registry_cases()


def test_registry_cases_span_the_formula_owners():
    owners = {owner for _name, _ask, owner in CASES}
    assert owners >= {
        "base", "design", "quantities", "commercial", "contracts",
        "planning", "procurement", "qaqc", "safety",
    }
    assert len(CASES) >= 70


@pytest.mark.parametrize(
    "name,ask,owner",
    CASES,
    ids=[name for name, _ask, _owner in CASES],
)
def test_user_supplied_inputs_select_that_formula(name, ask, owner):
    from app.lib.construction_formulas import formula_completed_by_user_text

    got = formula_completed_by_user_text(ask)
    assert got is not None, ask
    assert got[0] == name
    assert owner


def test_successful_runs_are_credited_with_the_users_inputs():
    from app.lib.construction_formulas import (
        format_user_calculation_answer,
        formula_completed_by_user_text,
        run_calculation,
    )
    from app.lib.source_labels import CALCULATOR_SUFFIX, calculator_label

    credited = 0
    for name, ask, _owner in CASES:
        got = formula_completed_by_user_text(ask)
        assert got is not None and got[0] == name
        env = run_calculation(name, dict(got[1]))
        if env.get("status") != "success":
            continue
        answer = format_user_calculation_answer(name, got[1], env)
        assert CALCULATOR_SUFFIX in answer
        assert calculator_label(name, got[1]) in answer
        credited += 1
    assert credited >= 60


def test_percent_hundredths_disagree_with_the_unscaled_product():
    found = derivation_mismatches("0.1% × 50,000,000 = 5,000,000")
    assert found, "a percent sign between the factor and the operator was ignored"
    _expr, stated, computed = found[0]
    assert stated == pytest.approx(5_000_000)
    assert computed == pytest.approx(50_000)


def test_percent_hundredths_agree_when_the_result_is_scaled():
    assert derivation_mismatches("0.1% × 50,000,000 = 50,000") == []


def test_supplied_operands_are_a_formula_ask_not_a_document_lookup():
    assert _message_is_formula_style_ask(DELAY) is True
    assert _wants_user_supplied_arithmetic_directive(DELAY) is True
    assert _forced_specific_tool(
        [{"role": "user", "content": DELAY}], {"construction_calc"},
    ) == "construction_calc"


def test_delay_damages_without_numbers_stays_a_document_lookup():
    assert _message_is_formula_style_ask(LOOKUP) is False
    assert _forced_specific_tool(
        [{"role": "user", "content": LOOKUP}], {"construction_calc"},
    ) is None


def test_rebar_area_has_no_registered_formula():
    from app.lib.construction_formulas import formula_completed_by_user_text

    assert formula_completed_by_user_text(REBAR) is None


def _pp(ask: str, answer: str, retrieval: str = "Curing notes. Keep the surface damp.") -> str:
    messages = [{"role": "user", "content": ask}]
    return _postprocess_answer(
        answer,
        {"content": retrieval},
        messages,
        project_id=None,
        audit_rec={"user_message_preview": ask, "chunks": []},
    )


def test_wrong_percent_product_is_replaced_by_the_calculator():
    out = _pp(
        DELAY,
        "The rate isn't in the retrieved excerpts. "
        "0.1% × 50,000,000 = 5,000,000 per day. "
        "0.1% × 50,000,000 = 5,000,000 per day. "
        "I can run the calculator if you ask again.",
    )
    assert "platform calculator" in out
    assert "50,000" in out
    assert "5,000,000" not in out
    assert "contract amount" in out
    assert "0.1%" in out


def test_invented_project_waste_is_the_calculator_default():
    out = _pp(
        SLAB,
        "The volume is 52.5 m³, using the project's documented waste factor of 5%. "
        "Source: curing-notes.pdf",
        retrieval="[doc_id=d1 chunk=0 src=curing-notes.pdf] Keep the surface damp.",
    )
    assert "platform calculator" in out
    assert "52.5" in out
    assert "project's documented" not in out.lower()
    assert "not a figure from the project documents" in out


def test_calculator_credit_is_the_source_not_an_unrelated_page(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "curing-notes.pdf"},
    )
    out = _pp(DELAY, "Daily delay damages are 50,000.")
    audit = {
        "chunks": [
            {"doc_id": "d1", "chunk_index": 0, "score": 0.91, "chunk_id": "c0"},
            {"doc_id": "d2", "chunk_index": 3, "score": 0.8, "chunk_id": "c1"},
            {"doc_id": "d3", "chunk_index": 1, "score": 0.7, "chunk_id": "c2"},
        ],
        "user_message_preview": DELAY,
    }
    sources = _build_sources_from_audit(audit, out)
    names = " ".join(s.get("doc_name") or "" for s in sources)
    assert "platform calculator" in names
    assert "curing-notes.pdf" not in names
    assert any(s.get("layer_label") == "Platform calculator" for s in sources)
    assert all(not s.get("source_class") for s in sources if "platform calculator" in (s.get("doc_name") or ""))
