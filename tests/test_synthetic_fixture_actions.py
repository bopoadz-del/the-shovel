"""Proof-of-life for the eval battery's 'dead' actions, on synthetic fixtures.

The 2026-07-08 sweep read a cluster of features as FAIL because the fixture
project had no input documents — the actions honestly refused ("upload the
register first") and the short refusals were scored as failures. These tests
run the real blocks/actions against the wholly synthetic fixture files from
``scripts/synthetic_fixtures.py`` and assert genuine output, proving the
capabilities themselves work when given their inputs. (BIM family excluded —
parked pending a real model fixture.)
"""
from __future__ import annotations

import pytest

from scripts.synthetic_fixtures import (
    build_boq_xlsx,
    build_ground_floor_dxf,
    build_rfi_register_xlsx,
    build_spec_section_txt,
    build_vo_log_xlsx,
    _BOQ_ROWS,
    _VO_ROWS,
)

EXPECTED_BOQ_TOTAL = round(sum(q * r for *_, q, r in _BOQ_ROWS), 2)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_fixtures")
    build_boq_xlsx(d / "synthetic_boq.xlsx")
    build_rfi_register_xlsx(d / "synthetic_rfi_register.xlsx")
    build_vo_log_xlsx(d / "synthetic_variation_order_log.xlsx")
    build_spec_section_txt(d / "synthetic_spec_section_03_concrete.txt")
    build_ground_floor_dxf(d / "ground_floor_plan.dxf")
    return d


# ── boq_process ("blocked" in the sweep — proven working here) ───────────────

@pytest.mark.asyncio
async def test_boq_processor_reads_synthetic_workbook(fixture_dir):
    from app.blocks.boq_processor import BOQProcessorBlock

    block = BOQProcessorBlock()
    result = await block.process({"file_path": str(fixture_dir / "synthetic_boq.xlsx")}, {})
    assert result.get("status") == "success", result
    assert result.get("item_count") == len(_BOQ_ROWS)
    assert len(result.get("line_items") or []) == len(_BOQ_ROWS)
    assert result.get("total_cost") == pytest.approx(EXPECTED_BOQ_TOTAL, rel=0.01)


@pytest.mark.asyncio
async def test_container_boq_process_action(fixture_dir):
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    # boq_processor registers via CEREBRUM_DOMAIN_KITS=construction; wire it
    # directly so this proof runs in the virgin CI profile too.
    from app.blocks.boq_processor import BOQProcessorBlock

    container.wire("boq_processor", BOQProcessorBlock())
    result = await container.route(
        "boq_process",
        {"file_path": str(fixture_dir / "synthetic_boq.xlsx")},
        {"action": "boq_process"},
    )
    assert result.get("status") == "success", result

    # Assert the PAYLOAD, not just the status string. Control-delete
    # 2026-08-12: replacing the whole boq_process body with
    # `return {"status": "success", "items": [], "total": 0}` left this test
    # GREEN, because a status check cannot tell a parsed workbook from an empty
    # one. The action could have stopped reading the file entirely and nothing
    # here would have noticed.
    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    line_items = (payload.get("line_items") or payload.get("items")
                  or payload.get("boq_items") or [])
    assert len(line_items) == len(_BOQ_ROWS), (
        f"expected {len(_BOQ_ROWS)} line items off the synthetic workbook, "
        f"got {len(line_items)} — keys present: {sorted(payload)}"
    )
    total = payload.get("total_cost", payload.get("total"))
    assert total == pytest.approx(EXPECTED_BOQ_TOTAL, rel=0.01), (
        f"expected total ~{EXPECTED_BOQ_TOTAL}, got {total}"
    )


# ── drawing_qto ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drawing_qto_measures_synthetic_dxf(fixture_dir):
    from app.blocks.drawing_qto import DrawingQTOBlock

    block = DrawingQTOBlock()
    result = await block.process({"file_path": str(fixture_dir / "ground_floor_plan.dxf")}, {})
    assert result.get("status") == "success", result
    blob = str(result)
    # Two slab bays: 30x20=600 and 18x12=216 m2 — both must be measured.
    assert "600" in blob and "216" in blob, f"slab areas missing from QTO result: {blob[:400]}"


# ── spec analysis (process_specification_full — previously untested) ─────────

@pytest.mark.asyncio
async def test_process_specification_full_on_synthetic_spec(fixture_dir):
    from app.containers.construction import ConstructionContainer

    text = (fixture_dir / "synthetic_spec_section_03_concrete.txt").read_text(encoding="utf-8")
    container = ConstructionContainer()
    # The spec_analyzer block registers via CEREBRUM_DOMAIN_KITS=construction;
    # wire it directly so this proof runs in the virgin CI profile too.
    from app.blocks.spec_analyzer import SpecAnalyzerBlock
    container.wire("spec_analyzer", SpecAnalyzerBlock())
    result = await container.process_specification_full({"extracted_text": text}, {})
    assert result.get("status") == "success", result
    blob = str(result)
    assert "C32/40" in blob or "32/40" in blob, f"concrete grade not extracted: {blob[:400]}"


@pytest.mark.asyncio
async def test_process_specification_full_honest_error_without_input():
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().process_specification_full({}, {})
    assert result.get("status") == "error"
    assert "No specification provided" in result.get("error", "")


# ── value_engineering (previously untested) ──────────────────────────────────

@pytest.mark.asyncio
async def test_value_engineering_on_boq_items():
    from app.containers.construction import ConstructionContainer

    boq = [
        {"description": d, "unit": u, "quantity": q, "rate": r, "total_cost": q * r}
        for _, _, d, u, q, r in _BOQ_ROWS
    ]
    result = await ConstructionContainer().value_engineering({"boq": boq}, {})
    assert result.get("status") == "success", result
    assert result["current_project_cost"] == pytest.approx(EXPECTED_BOQ_TOTAL, rel=0.01)
    assert "scenarios" in result and "recommended_scenario" in result


# ── variation_order_manager (previously untested) ────────────────────────────

@pytest.mark.asyncio
async def test_variation_order_manager_prices_a_vo():
    from app.containers.construction import ConstructionContainer

    existing = [
        {"vo_number": no, "description": desc, "value": val, "status": status,
         "total": val}
        for no, desc, val, status in _VO_ROWS[:2]
    ]
    result = await ConstructionContainer().variation_order_manager(
        {
            "variation_data": {
                "vo_number": "VO-12",
                "description": "Add 300m of storm drain diversion",
                "type": "addition",
                "direct_cost": 420000,
                "schedule_impact_days": 14,
            },
            "existing_vos": existing,
            "contract_value": 60_000_000,
        },
        {},
    )
    assert result.get("status") == "success", result
    assert result["vo_number"] == "VO-12"
    assert "pricing" in result


@pytest.mark.asyncio
async def test_variation_order_manager_honest_error_without_vo():
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().variation_order_manager({}, {})
    assert result.get("status") == "error"


# ── registers index end-to-end (rfi/vo docs are readable spreadsheets) ───────

def test_rfi_register_extractable(fixture_dir):
    """The register uploads as a normal document — its text must be
    extractable so RAG can answer 'how many RFIs are open'."""
    from app.core.doc_index import extract_document_text

    text = extract_document_text(
        str(fixture_dir / "synthetic_rfi_register.xlsx"), "synthetic_rfi_register.xlsx"
    )
    assert "RFI-004" in text and "overdue" in text.lower()


def test_vo_log_extractable(fixture_dir):
    from app.core.doc_index import extract_document_text

    text = extract_document_text(
        str(fixture_dir / "synthetic_variation_order_log.xlsx"),
        "synthetic_variation_order_log.xlsx",
    )
    assert "VO-04" in text and "Pending" in text
