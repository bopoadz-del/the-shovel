"""Regression for leftover-hat UI fails L1 / L2 / L3 / L4 / L5 / L7.

Live UI battery (pinned hats on /v1/chat/stream):
  L1 named-file fetch of fenwick_waterproofing_spec (timestamped .docx)
  L2 bim-analyst hung in clash geom and never reported IfcWall=8
  L3 contracts-manager searched the project instead of computing 480,000
     from word-numbers ("ten percent of four million eight hundred thousand")
  L4 validation short-circuited at syntactic `value is None` on a prose claim
  L5 document-ingestion omitted the Next: <hat> handoff
  L7 heavy-reasoning got empty sympy metadata then force_synthesis blocked
     formula_executor_v2, so 2.4 / 6840 never appeared

These tests pin the machinery (no live LLM).
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _cg_english_and_percent_values,
    _cg_grounded_numbers,
    _cost_grounding_gate,
    _ensure_ingestion_handoff,
    _next_agent_from_turn,
    _should_force_synthesis,
    _should_short_circuit_rag_miss,
    _looks_like_self_contained_calculation,
    _CG_REFUSAL,
)
from app.blocks.bim_extractor import BIMExtractorBlock
from app.blocks.sympy_reasoning import SymPyReasoningBlock
from app.blocks.validation_pipeline import ValidationPipelineBlock
from tests.conftest import requires_construction_kit


# ── L3: word-numbers ground the LD cap ──────────────────────────────────────


def test_english_percent_of_grounds_ld_cap():
    msg = (
        "LD cap is ten percent of four million eight hundred thousand. "
        "Notice bar is twenty one days."
    )
    values = _cg_english_and_percent_values(msg)
    assert any(abs(v - 4_800_000) < 1 for v in values)
    assert any(abs(v - 480_000) < 1 for v in values)


def test_worded_ld_cap_answer_passes_cost_gate():
    messages = [{
        "role": "user",
        "content": (
            "Liquidated damages cap is ten percent of four million eight "
            "hundred thousand. Compute the cap."
        ),
    }]
    grounded = _cg_grounded_numbers("", messages)
    assert any(abs(g - 480_000) <= 2400 for g in grounded)
    answer = "The LD cap is 480,000 SAR (10% of 4,800,000)."
    assert _cost_grounding_gate(answer, None, messages) == answer


def test_fabricated_rate_still_refused_with_worded_ld_prompt():
    messages = [{
        "role": "user",
        "content": "ten percent of four million eight hundred thousand LD cap",
    }]
    answer = "Use 465 SAR/m3 for C40 concrete."
    assert _cost_grounding_gate(answer, None, messages) == _CG_REFUSAL


# ── L7: empty sympy is not a deliverable; expression path yields 6840 ───────


L7_PROMPT = (
    "Stay on heavy-reasoning. Planned quantity 16, actual 18.4, rate 2850. "
    "Compute (18.4-16)*2850 with sympy_reasoning {expression}. "
    "Report the 2.4 overrun and the 6840 cost impact."
)


def test_leftover_l7_prompt_is_self_contained_and_skips_rag_miss():
    assert _looks_like_self_contained_calculation(L7_PROMPT)
    audit = {
        "identifier_miss": True,
        "threshold_fired": True,
        "extracted_identifiers": ["18.4-16"],
    }
    assert _should_short_circuit_rag_miss(audit, None, L7_PROMPT) is False


def test_e4_volume_claim_does_not_force_synthesis_on_validation():
    """Leftover E4: validating the raft ask must not disarm construction_calc.

    L4 (beam claim, no calc verb) still forces synthesis — pinned below.
    """
    rec = {
        "name": "validation_pipeline",
        "ok": True,
        "result": {
            "status": "success",
            "overall": "pass",
            "stages": {"syntactic": {"pass": True, "reason": "ok"}},
            "claim": (
                "Concrete volume for a raft 30×20×1.5 m including your "
                "documented waste factor."
            ),
        },
    }
    assert _should_force_synthesis(rec) is False


def test_empty_sympy_does_not_force_synthesis():
    rec = {
        "name": "sympy_reasoning",
        "ok": True,
        "result": {
            "status": "success",
            "variances": [],
            "cost_impacts": [],
            "formulas": {"variance_pct": "(actual - avg) / avg * 100"},
            "items_analyzed": 0,
        },
    }
    assert _should_force_synthesis(rec) is False


def test_sympy_expression_result_does_force_synthesis():
    rec = {
        "name": "sympy_reasoning",
        "ok": True,
        "result": {
            "status": "success",
            "expression": "(18.4-16)*2850",
            "value": 6840.0,
            "variances": [],
            "cost_impacts": [],
        },
    }
    assert _should_force_synthesis(rec) is True


@pytest.mark.asyncio
async def test_sympy_free_expression_computes_l7_product():
    block = SymPyReasoningBlock()
    r = await block.process({"expression": "(18.4-16)*2850"})
    assert r["status"] == "success"
    assert r["value"] == 6840.0
    assert r["expression"]


@pytest.mark.asyncio
async def test_sympy_unicode_minus_and_times():
    block = SymPyReasoningBlock()
    r = await block.process({"expression": "(18.4−16)×2850"})
    assert r["status"] == "success"
    assert r["value"] == pytest.approx(6840.0)


@pytest.mark.asyncio
async def test_sympy_rejects_non_arithmetic_expression():
    block = SymPyReasoningBlock()
    r = await block.process({"expression": "__import__('os')"})
    assert r["status"] == "error"


# ── L4: prose claim runs Physical, not syntactic skip ───────────────────────


L4_PROMPT = (
    "Stay on validation. Validate this office floor beam forty metres long "
    "on a fifty millimetre steel I-section carrying eight hundred kilonewtons "
    "per metre. Report Physical and Tier."
)

L4_XML_LEAK = """I need to re-run the formula executor with valid JSON.

<function_calls>
<invoke name="formula_executor_v2">
<parameter name="input">{"task":"Calculate maximum bending moment for a 40 m beam"}</parameter>
</invoke>
</function_calls>

<function_results>
<result>
{"status": "success", "notes": "These absurd results confirm physical impossibility"}
</result>
</function_results>
"""


@pytest.mark.asyncio
async def test_prose_beam_claim_fails_physical_not_syntactic():
    block = ValidationPipelineBlock()
    claim = (
        "office floor beam forty metres long on a fifty millimetre steel "
        "I-section carrying eight hundred kilonewtons per metre"
    )
    r = await block.process({"value": None, "claim": claim})
    assert r["status"] == "success"
    assert r["stages"]["syntactic"]["pass"] is True
    assert r["stages"]["physical"]["pass"] is False
    assert r["first_failure"] == "physical"
    assert r["overall"] == "fail"
    assert r.get("tier") == 4
    assert "skipped" not in r["stages"]["physical"]["reason"]


def test_leftover_l4_xml_synthesis_grafts_physical_tier4():
    """Live leftover L4: pipeline was correct; synthesis dumped XML."""
    from app.agents.runtime import (
        _recover_tool_calls_from_content,
        _sanitize_final_text,
        _TOOL_FORMAT_FALLBACK,
    )

    claim = (
        "office floor beam forty metres long on a fifty millimetre steel "
        "I-section carrying eight hundred kilonewtons per metre"
    )
    payload = {
        "status": "success",
        "overall": "fail",
        "first_failure": "physical",
        "tier": 4,
        "stages": {
            "syntactic": {"pass": True, "reason": "prose claim"},
            "dimensional": {"pass": True, "reason": "units parsed"},
            "physical": {
                "pass": False,
                "reason": "span/depth 800:1 (40 m / 0.05 m) is physically implausible for a beam or floor member",
            },
            "empirical": {"pass": True, "reason": "skipped"},
            "operational": {"pass": True, "reason": "skipped"},
        },
        "claim": claim,
    }
    messages = [
        {"role": "user", "content": L4_PROMPT},
        {
            "role": "tool",
            "name": "validation_pipeline",
            "content": json.dumps({
                "block": "validation_pipeline",
                "status": "success",
                "result": payload,
            }),
        },
    ]
    text = _sanitize_final_text(L4_XML_LEAK, messages=messages)
    assert "function_calls" not in text
    assert "<invoke" not in text
    assert "Physical" in text
    assert "✗" in text
    assert "Tier: 4" in text
    assert "fail" in text.lower()
    assert "40 m" in text or "0.05 m" in text

    assert _sanitize_final_text(L4_XML_LEAK) == _TOOL_FORMAT_FALLBACK

    recovered = _recover_tool_calls_from_content(L4_XML_LEAK)
    assert recovered
    assert recovered[0]["function"]["name"] == "formula_executor_v2"


def test_leftover_l4_validation_still_forces_synthesis():
    """E4's checker carve-out must not disarm leftover L4's verdict."""
    rec = {
        "name": "validation_pipeline",
        "ok": True,
        "result": {
            "status": "success",
            "overall": "fail",
            "first_failure": "physical",
            "tier": 4,
            "stages": {
                "syntactic": {"pass": True, "reason": "prose claim"},
                "physical": {"pass": False, "reason": "implausible"},
            },
            "claim": (
                "office floor beam forty metres long on a fifty millimetre "
                "steel I-section carrying eight hundred kilonewtons per metre"
            ),
        },
    }
    assert _should_force_synthesis(rec) is True


# ── L2: clash detection off by default ──────────────────────────────────────


def test_bim_extractor_clash_off_by_default():
    assert BIMExtractorBlock.default_config["run_clash_detection"] is False


@pytest.mark.asyncio
async def test_construction_container_aliases_bim_extractor_action():
    """Leftover-hat L2 UI: the model called construction action bim_extractor."""
    from app.containers.construction import ConstructionContainer
    from pathlib import Path
    c = ConstructionContainer()
    c.wire("bim_extractor", BIMExtractorBlock())
    path = str(Path(__file__).resolve().parent / "fixtures" / "sample_office.ifc")
    r = await c.route("bim_extractor", {"file_path": path}, {"run_clash_detection": False})
    assert "Unknown action" not in str(r.get("error", ""))
    inner = r.get("result") if isinstance(r.get("result"), dict) else r
    qty = inner.get("quantities") or r.get("quantities") or {}
    walls = (qty.get("walls") or {}).get("count")
    assert walls == 8, inner
    assert inner.get("status") != "error"
    assert "IfcWall" in str(inner.get("ifc_schema") or "") or str(inner.get("ifc_schema") or "").startswith("IFC")


# ── Kernels ─────────────────────────────────────────────────────────────────


@requires_construction_kit
def test_contracts_kernel_treats_pasted_figures_as_source_of_truth():
    from app.agents.runtime import load_agents
    text = load_agents()["contracts-manager"].system_prompt.lower()
    assert "source of truth" in text
    assert "number-words" in text or "number words" in text
    assert "no rate on file" in text


@requires_construction_kit
def test_validation_kernel_never_sends_null_value():
    from app.agents.runtime import load_agents
    text = load_agents()["validation"].system_prompt.lower()
    assert "value: null" in text or "value:null" in text.replace(" ", "")
    assert "claim" in text
    assert "physical" in text


@requires_construction_kit
def test_heavy_kernel_user_rate_and_expression_path():
    from app.agents.runtime import load_agents
    text = load_agents()["heavy-reasoning"].system_prompt.lower()
    assert "user's own message" in text or "user typed" in text
    assert "{expression:" in text or "expression:" in text
    assert "empty sympy" in text or "no boq_data" in text


@requires_construction_kit
def test_bim_kernel_clash_off_unless_asked():
    from app.agents.runtime import load_agents
    text = load_agents()["bim-analyst"].system_prompt.lower()
    assert "clash" in text
    assert "pre-dispatch" in text or "pre-dispatched" in text
    assert "ifcwall" in text


# ── L1: named-file fetch is not a RAG-miss identifier ───────────────────────


L1_PROMPT = (
    "Open fenwick_waterproofing_spec. Quote the unique document token "
    "string from that file."
)


def test_leftover_l1_named_stem_does_not_rag_miss():
    """The leftover L1 prompt has no digit-bearing identifier, so the
    RAG-miss short-circuit must not fire; fetch_document predispatch
    (tests/test_ifc_predispatch.py) supplies the token from disk."""
    audit = {
        "identifier_miss": True,
        "threshold_fired": True,
        "extracted_identifiers": ["fenwick_waterproofing_spec"],
    }
    assert not _should_short_circuit_rag_miss(audit, None, L1_PROMPT)


# ── L5: document-ingestion must end with Next: <hat> ────────────────────────


def test_leftover_l5_grafts_next_bim_for_ifc():
    msgs = [{"role": "user", "content": "Ingest sample_office.ifc and classify it."}]
    out = _ensure_ingestion_handoff(
        "Looks like an IFC model with walls and slabs.",
        msgs,
        "document-ingestion",
    )
    assert out.rstrip().endswith("Next: bim-analyst")
    assert _next_agent_from_turn(msgs) == "bim-analyst"


def test_leftover_l5_keeps_existing_next_line():
    msgs = [{"role": "user", "content": "ingest foo.ifc"}]
    text = "Looks like IFC.\n\nNext: bim-analyst"
    assert _ensure_ingestion_handoff(text, msgs, "document-ingestion") == text


def test_leftover_l5_does_not_graft_other_hats():
    msgs = [{"role": "user", "content": "ingest foo.ifc"}]
    text = "Looks like IFC."
    assert _ensure_ingestion_handoff(text, msgs, "bim-analyst") == text


def test_leftover_l5_docx_handoff_is_contracts_manager():
    msgs = [{"role": "user", "content": (
        "Open fenwick_waterproofing_spec.docx and classify it."
    )}]
    assert _next_agent_from_turn(msgs) == "contracts-manager"
    out = _ensure_ingestion_handoff("Looks like a spec.", msgs, "document-ingestion")
    assert out.rstrip().endswith("Next: contracts-manager")
