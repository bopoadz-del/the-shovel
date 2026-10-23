"""What a person reads after a calculator answers.

A registered formula or tool id is an internal name. The answer shows that
registration's display name, for every id in both registries, including when
the id is written as ``key=value``. A file name and an engineering symbol that
is not a registered id stay as written.

When the user already supplied the inputs, the answer does not attach a
currency or unit that is absent from the request, the calculator result and
the retrieved excerpts, does not name a clause or standard as the basis
unless those excerpts or that result state it, and does not compare the
figure with a limit the text never states.
"""
from __future__ import annotations

import pytest

from app.agents.core.tool_registry import OWNERS, get, tools_of
from app.agents.runtime import _apply_rag_context, _postprocess_answer
from app.lib.formula_registry import all_specs
from app.lib.source_labels import input_phrase

DELAY = (
    "Calculate delay damages per day: contract amount 50,000,000, "
    "rate 0.1% per day."
)
DELAY_USD = (
    "Calculate delay damages per day in USD: contract amount 50,000,000, "
    "rate 0.1% per day."
)
SLAB = (
    "Calculate the volume of concrete for a slab 20 m long, 10 m wide "
    "and 0.25 m thick."
)
NOTE = "Please confirm the following note."
RETRIEVAL = "Curing notes. Keep the surface damp."


def _pp(ask: str, answer: str, retrieval: str = RETRIEVAL) -> str:
    return _postprocess_answer(
        answer,
        {"content": retrieval},
        [{"role": "user", "content": ask}],
        project_id=None,
        audit_rec={"user_message_preview": ask, "chunks": []},
    )


def _registry_ids() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for spec in all_specs():
        rows.append(("formula", spec.name, spec.display_name))
    for owner in OWNERS:
        for name in tools_of(owner):
            spec = get(name)
            assert spec is not None and spec.display_name
            rows.append(("tool", spec.name, spec.display_name))
    return rows


def _parameter_assignments() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in all_specs():
        for key, unit in (spec.inputs or {}).items():
            if "_" not in key or key in seen:
                continue
            seen.add(key)
            rows.append((key, unit or ""))
    return rows


IDS = _registry_ids()
ASSIGNMENTS = _parameter_assignments()


@pytest.mark.parametrize(
    "kind,name,display",
    IDS,
    ids=[f"{kind}:{name}" for kind, name, _display in IDS],
)
def test_registry_id_is_shown_as_its_display_name(kind, name, display):
    answer = f"The note names `{name}` and also {name} in the working."
    out = _pp(NOTE, answer)
    assert name not in out
    assert display in out
    assert kind


@pytest.mark.parametrize(
    "kind,name,display",
    IDS,
    ids=[f"{kind}:{name}" for kind, name, _display in IDS],
)
def test_assignment_of_a_registry_id_reads_as_the_display_name(kind, name, display):
    answer = f"The working shows slot={name} before the result."
    out = _pp(NOTE, answer)
    assert f"slot={name}" not in out
    assert name not in out
    assert display in out
    assert kind


@pytest.mark.parametrize(
    "key,unit",
    ASSIGNMENTS,
    ids=[key for key, _unit in ASSIGNMENTS],
)
def test_registered_parameter_assignment_reads_as_words(key, unit):
    shown = input_phrase(key, 12, unit)
    answer = f"The working shows {key}=12 before the result."
    out = _pp(NOTE, answer)
    assert f"{key}=" not in out
    assert key not in out
    assert shown in out


def test_filenames_and_unregistered_notation_are_not_rewritten():
    ids = {name for _kind, name, _display in IDS}
    params = {key for key, _unit in ASSIGNMENTS}
    filename = "fenwick_waterproofing_spec.docx"
    stem = "fenwick_waterproofing_spec"
    symbol = "fy_k"
    assert stem not in ids and stem not in params
    assert symbol not in ids and symbol not in params
    named = IDS[0][1]
    registered_file = f"{named}.docx"
    answer = (
        f"See {filename} and {registered_file}. "
        f"The note records {symbol}=500 MPa and N/mm2."
    )
    out = _pp(NOTE, answer)
    assert filename in out
    assert registered_file in out
    assert f"{symbol}=500" in out
    assert "N/mm2" in out


def test_unsupplied_currency_clause_and_dangling_comparison_are_removed():
    answer = (
        "Delay damages per day: daily amount 50,000.\n"
        "0.1% of SAR 50,000,000.00 = SAR 50,000.00 per calendar day.\n"
        "This was computed on the FIDIC Sub-Clause 8.8 basis.\n"
        "Your 0.1% per day sits below that band.\n"
        "Source: Delay damages per day — platform calculator "
        "(contract amount 50,000,000, rate 0.1%)"
    )
    out = _pp(DELAY, answer)
    assert "SAR" not in out
    assert "FIDIC" not in out
    assert "Sub-Clause" not in out
    assert "that band" not in out.lower()
    assert "50,000" in out
    assert "0.1%" in out
    assert "platform calculator" in out
    assert "per calendar day" in out


def test_same_guard_on_another_formula_drops_its_unsupplied_qualifiers():
    answer = (
        "Concrete volume: volume 52.5 m3.\n"
        "Breakdown from the platform calculator (`concrete_volume`):\n"
        "The figure is AED 52.5 kg on the ACI 318 basis.\n"
        "This thickness sits above that threshold.\n"
        "Source: Concrete volume — platform calculator "
        "(length 20 m, width 10 m, thickness 0.25 m)"
    )
    out = _pp(SLAB, answer)
    assert "concrete_volume" not in out
    assert "Concrete volume" in out
    assert "AED" not in out
    assert "kg" not in out.lower()
    assert "ACI" not in out
    assert "that threshold" not in out.lower()
    assert "52.5" in out
    assert "m3" in out
    assert "platform calculator" in out


def test_stated_currency_clause_and_comparison_stay():
    answer = (
        "Delay damages per day: daily amount 50,000.\n"
        "0.1% of USD 50,000,000 = USD 50,000 per calendar day, "
        "also written EUR 50,000.\n"
        "Computed on the FIDIC Sub-Clause 8.8 basis.\n"
        "The contract states a band of 0.05% to 0.5% per day. "
        "Your 0.1% per day sits below that band.\n"
        "It also sits under that ceiling."
    )
    out = _pp(
        DELAY_USD,
        answer,
        retrieval="The contract applies FIDIC Sub-Clause 8.8.",
    )
    assert "USD" in out
    assert "FIDIC Sub-Clause 8.8" in out
    assert "that band" in out.lower()
    assert "EUR" not in out
    assert "that ceiling" not in out.lower()
    assert "SAR" not in out
    assert "50,000" in out


def test_replaced_calculator_statement_omits_the_unsupplied_currency():
    out = _pp(DELAY, "0.1% × 50,000,000 = 5,000,000 per day.")
    assert "50,000" in out
    assert "5,000,000" not in out
    assert "SAR" not in out
    assert "platform calculator" in out


def test_document_lookup_is_not_rewritten_as_a_supplied_input():
    ask = "What are the delay damages under this contract?"
    answer = "Delay damages are SAR 50,000 per day under FIDIC Sub-Clause 8.8."
    out = _pp(ask, answer)
    assert "SAR" in out
    assert "Sub-Clause" in out


def test_arithmetic_directive_forbids_unsupplied_basis_and_internal_ids():
    messages = [{"role": "user", "content": DELAY}]
    assert _apply_rag_context(messages, {"content": RETRIEVAL}) is True
    out = messages[-1]["content"]
    assert "CALCULATION" in out
    assert "Do NOT refuse" in out
    assert "they came from the user" in out
    assert "never invent" in out
    assert "Never write an internal id" in out
    assert "display name" in out
    assert "Do not attach a currency or a unit" in out
    assert "Do not cite a clause or a standard as the basis" in out
    assert "Do not compare the result with a limit" in out
    assert DELAY in out
