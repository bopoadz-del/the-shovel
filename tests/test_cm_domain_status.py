"""domain_status derivation from auto_pipeline panels."""
from __future__ import annotations

import pytest

from app.core.cm_domain_status import panels_to_domain_status


def test_panels_to_domain_status_maps_real_panels():
    panels = [
        {"type": "document_info", "data": {"file": "x.pdf", "doc_type": "drawing"}},
        {"type": "quantities", "data": {"concrete_volume_m3": 12.0}},
        {
            "type": "cost_estimate",
            "data": {"subtotal": 1000, "total_estimate": 1200},
            "line_items": [{"name": "concrete", "total": 1000}],
        },
        {
            "type": "procurement",
            "data": {
                "procurement_list": [{"item": "rebar", "qty": 1}],
                "critical_long_lead_items": [{"item": "switchgear"}],
            },
        },
        {"type": "risks", "data": [{"description": "delay", "likelihood": "med"}]},
        {"type": "contract", "data": {"clauses": []}},
    ]
    status = panels_to_domain_status(panels)
    assert "document" in status
    assert "cost" in status
    assert status["cost"]["status"] == "healthy"
    assert "procurement" in status
    # Long-lead critical items → warning, not fabricated lines
    assert status["procurement"]["status"] == "warning"
    assert "risk" in status


def test_empty_procurement_omitted_or_planned():
    panels = [
        {"type": "procurement", "data": {"procurement_list": []}},
    ]
    status = panels_to_domain_status(panels)
    # No payload → stays planned (or absent healthy signal)
    assert status.get("procurement", {}).get("status") in ("planned", None) or (
        status["procurement"]["status"] == "planned"
    )


def test_pipeline_warning_marks_failed_or_warning():
    panels = [
        {"type": "cost_estimate", "data": {}, "error": "Cost estimate failed"},
    ]
    status = panels_to_domain_status(
        panels, pipeline_warnings=[{"panel": "cost_estimate", "error": "boom"}]
    )
    assert status["cost"]["status"] == "failed"


@pytest.mark.asyncio
async def test_health_check_accepts_derived_domain_status():
    """project_dashboard.health_check is non-empty when fed auto_pipeline output."""
    from app.blocks.project_dashboard import ProjectDashboardBlock

    panels = [
        {"type": "document_info", "data": {"file": "a.pdf"}},
        {"type": "risks", "data": [{"description": "x", "likelihood": "low", "impact": "med"}]},
        {"type": "schedule", "data": {"activity_count": 10}},
    ]
    domain_status = panels_to_domain_status(panels)
    assert domain_status

    block = ProjectDashboardBlock()
    result = await block.process({
        "action": "health_check",
        "domain_status": domain_status,
    })
    assert result["status"] == "success"
    assert result["overall_status"] != "unknown"
    assert result.get("domain_scores") or result.get("overall_score") is not None
