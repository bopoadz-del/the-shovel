"""Cycle 2: percentage-of-ACA VAT hedge, and default-rate cost calculators.

Two live classes on theshovel.ai build 0b1d13a (synthetic fixtures only).

1. A percentage of the Accepted Contract Amount (Advance Payment, and the
   same shape for Retention Money and the Maximum Amount of Delay Damages)
   is one figure on the excluding-VAT base. A model reply that states that
   figure and then offers a VAT-inclusive alternative must not keep the
   alternative.

2. When the operator asks for a deterministic cost calculator's default
   rates, the answer states that tool result. A project-document cost
   total is not the answer. A successful cost-calculator result is never
   replaced by the no-rate refusal.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _CG_REFUSAL,
    _cost_grounding_gate,
    _postprocess_answer,
)
from app.lib.construction_formulas import run_calculation
from app.lib.construction_formulas_commercial import (
    parse_accepted_contract_amount,
)


EXCL = "SAR 1,754,504,456.25"
INCL = "SAR 2,017,680,124.69"
INCL_AMOUNT = 2_017_680_124.69

LABEL = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage "
    "[SYN-2099-001_Contract Data.pdf].\nCONTRACT DATA\n"
)
CD_ACA = LABEL + (
    "1.1.1: | | Accepted Contract Amount: "
    f"{EXCL} excluding VAT |\n"
    "1.1.1: | | Accepted Contract Amount including VAT: "
    f"{INCL} |\n"
)
CD_ADVANCE = LABEL + (
    "14.2.1: | | Advance Payment: 10% of the Accepted Contract Amount | |\n"
)
CD_RETENTION = LABEL + (
    "14.3: | | Retention Money: 5% of the Accepted Contract Amount | |\n"
)
CD_CAP = LABEL + (
    "8.8.1: | | Maximum Amount of Delay Damages: "
    "10% of the Accepted Contract Amount | |\n"
    "8.8.1: | | Delay Damages (for the whole of the Works): "
    "0.1% of the Contract Price per calendar day |\n"
)

AP_ASKS = (
    "Calculate the Advance Payment in SAR.",
    "Under the Contract Data, what percentage of the Accepted Contract Amount "
    "is paid as the Advance Payment, and how much is that in SAR?",
    "Work out the Advance Payment in SAR.",
)
RETAIN_ASK = "Calculate the Retention Money in SAR."
CAP_ASK = "Calculate the Maximum Amount of Delay Damages in SAR."

CONCRETE_ASKS = (
    "Run the concrete cost build-up for 100 m3 using the calculator's "
    "default rates and give me the selling price per m3 and the total.",
    "What is the all-in selling rate in SAR/m3 and the total value for "
    "100 m3 of concrete from the standard concrete cost build-up with "
    "default inputs?",
    "Using the calculator's default rates, what is the selling price per "
    "m3 and the total for 100 m3 of concrete?",
)
REBAR_ASK = (
    "Run the rebar cost build-up for 1000 kg using the calculator's "
    "default rates and give me the selling price per tonne and the total."
)

# Unrelated project-document cost a reader can take as the total.
DOC_TOTAL = "800,000 Dhs on 16,000 m³"
DOC_CHUNK = (
    "Reference context:\n"
    "[doc_id=sched1 chunk=0 score=0.42] "
    "Concrete supply priced schedule: 800,000 Dhs on 16,000 m³."
)


def _sys(*texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=cd{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def _hedge(label: str, percent: float, correct: float) -> str:
    alt = round(INCL_AMOUNT * (percent / 100.0), 2)
    return (
        f"{label} = SAR {correct:,.2f} "
        f"({percent:g}% of the Accepted Contract Amount). "
        f"Contract Data 14.2.\n"
        f"If the {label} is to be calculated on the VAT-inclusive figure, "
        f"it would be SAR {alt:,.2f} — the Contract Data does not state "
        f"which base applies, so confirm against the Letter of Award. "
        f"The VAT-inclusive contract amount is SAR {INCL_AMOUNT:,.2f}."
    )


def _assert_one_base(out: str, correct: float, percent: float) -> None:
    alt = round(INCL_AMOUNT * (percent / 100.0), 2)
    assert f"{correct:,.2f}" in out
    assert f"{alt:,.2f}" not in out
    assert f"{INCL_AMOUNT:,.2f}" not in out
    low = out.lower()
    assert "vat-inclusive" not in low
    assert "including vat" not in low
    assert "which base applies" not in low
    assert "letter of award" not in low


def _product(excerpts: str, percent: float) -> float:
    parsed = parse_accepted_contract_amount(excerpts)
    assert parsed is not None
    amount, _currency = parsed
    return round(amount * (percent / 100.0), 2)


@pytest.mark.parametrize("ask", AP_ASKS)
def test_advance_payment_has_one_amount_and_no_vat_alternative(ask: str):
    rag = _sys(CD_ADVANCE, CD_ACA)
    correct = _product(rag["content"], 10)
    reply = _hedge("Advance Payment", 10, correct)
    out = _postprocess_answer(reply, rag, [{"role": "user", "content": ask}])
    _assert_one_base(out, correct, 10)


def test_retention_money_has_no_vat_inclusive_alternative():
    """Sibling: percentage of the Accepted Contract Amount, not Advance Payment."""
    rag = _sys(CD_RETENTION, CD_ACA)
    correct = _product(rag["content"], 5)
    reply = _hedge("Retention Money", 5, correct)
    out = _postprocess_answer(
        reply, rag, [{"role": "user", "content": RETAIN_ASK}],
    )
    _assert_one_base(out, correct, 5)
    assert "per calendar day" not in out.lower()


def test_maximum_delay_damages_cap_has_no_vat_inclusive_alternative():
    """Sibling: the cap is a percentage of ACA, not the daily rate."""
    rag = _sys(CD_CAP, CD_ACA)
    correct = _product(rag["content"], 10)
    reply = _hedge("Maximum Amount of Delay Damages", 10, correct)
    out = _postprocess_answer(
        reply, rag, [{"role": "user", "content": CAP_ASK}],
    )
    _assert_one_base(out, correct, 10)
    assert "per calendar day" not in out.lower()


def _calc_tool(name: str, params: dict) -> tuple[dict, dict]:
    env = run_calculation(name, params)
    assert env["status"] == "success", env
    return env, {
        "role": "tool",
        "name": "construction_calc",
        "content": json.dumps(env),
    }


def _money_blob(text: str) -> str:
    return (text or "").replace(",", "").replace(" ", "")


def _assert_calc_not_doc_total(out: str, env: dict) -> None:
    result = env["result"]
    selling = next(
        result[k] for k in (
            "selling_price_sar_m3",
            "selling_price_sar_t",
            "selling_price_sar_m2",
        )
        if result.get(k) is not None
    )
    total = result.get("total_project_value_sar")
    if total is None:
        total = result.get("total_for_area_sar")
    blob = _money_blob(out)
    sell = f"{float(selling):.2f}".rstrip("0").rstrip(".")
    assert sell in blob
    assert str(int(round(float(total)))) in blob
    assert "800,000" not in out
    assert "800000" not in blob
    assert "16,000" not in out
    assert "rate on file" not in out.lower()
    assert out.strip() != _CG_REFUSAL


def _doc_rag() -> dict:
    return {"role": "system", "content": DOC_CHUNK}


@pytest.mark.parametrize("ask", CONCRETE_ASKS)
def test_concrete_default_rates_state_the_calculator_not_a_document_total(ask):
    env, tool = _calc_tool("cost_buildup_concrete", {"quantity_m3": 100})
    selling = env["result"]["selling_price_sar_m3"]
    total = env["result"]["total_project_value_sar"]
    reply = (
        f"The selling price is {selling:.2f} SAR/m3 and the total for "
        f"100 m3 is {total:,.0f} SAR.\n"
        f"The project documents also record {DOC_TOTAL}."
    )
    out = _postprocess_answer(
        reply, _doc_rag(),
        [{"role": "user", "content": ask}, tool],
    )
    _assert_calc_not_doc_total(out, env)


def test_concrete_default_rates_refusal_does_not_replace_the_tool_result():
    """The tool ran. The no-rate refusal must not be the answer."""
    ask = CONCRETE_ASKS[1]
    env, tool = _calc_tool("cost_buildup_concrete", {"quantity_m3": 100})
    out = _postprocess_answer(
        _CG_REFUSAL, _doc_rag(),
        [{"role": "user", "content": ask}, tool],
    )
    _assert_calc_not_doc_total(out, env)


def test_cost_gate_does_not_refuse_a_successful_concrete_cost_calc():
    """Guard path: an ungrounded document total must not wipe the tool."""
    ask = CONCRETE_ASKS[2]
    env, tool = _calc_tool("cost_buildup_concrete", {"quantity_m3": 100})
    selling = env["result"]["selling_price_sar_m3"]
    total = env["result"]["total_project_value_sar"]
    reply = (
        f"Selling price {selling:.2f} SAR/m3, total {total:,.0f} SAR. "
        "The schedule total is SAR 800,000."
    )
    out = _cost_grounding_gate(
        reply, None,
        [{"role": "user", "content": ask}, tool],
    )
    _assert_calc_not_doc_total(out, env)


def test_rebar_default_rates_sibling_keeps_the_calculator():
    """Same shape for another deterministic cost calculator."""
    env, tool = _calc_tool("cost_buildup_rebar", {"quantity_kg": 1000})
    selling = env["result"]["selling_price_sar_t"]
    total = env["result"]["total_project_value_sar"]
    injected = (
        f"Selling price {selling:.0f} SAR/t and total {total:,.0f} SAR.\n"
        f"The project documents also record {DOC_TOTAL}."
    )
    msgs = [{"role": "user", "content": REBAR_ASK}, tool]
    kept = _postprocess_answer(injected, _doc_rag(), msgs)
    _assert_calc_not_doc_total(kept, env)
    refused = _postprocess_answer(_CG_REFUSAL, _doc_rag(), msgs)
    _assert_calc_not_doc_total(refused, env)
