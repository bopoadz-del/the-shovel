"""Derive project_dashboard-ready domain_status from auto_pipeline panels.

Keeps ActivityGraph / ProjectHealthAggregator fed without inventing BOQ or
procurement line items — only reflects what panels already contain.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Panel type → dependency_graph.Domain value
_PANEL_DOMAIN: Dict[str, str] = {
    "document_info": "document",
    "quantities": "cost",
    "cost_estimate": "cost",
    "procurement": "procurement",
    "risks": "risk",
    "submittals": "document",
    "schedule": "schedule",
    "contract": "contract",
}


def _panel_has_payload(panel: Dict[str, Any]) -> bool:
    if panel.get("error"):
        return False
    data = panel.get("data")
    if data is None:
        return False
    if isinstance(data, dict):
        if data.get("error"):
            return False
        # Empty cost summary / empty procurement list → no healthy signal
        if panel.get("type") == "procurement":
            items = data.get("procurement_list") or []
            return bool(items)
        if panel.get("type") == "cost_estimate":
            return bool(data.get("total_estimate") or data.get("subtotal") or panel.get("line_items"))
        if panel.get("type") == "risks":
            return bool(data) if isinstance(data, list) else bool(data)
        if panel.get("type") == "submittals":
            return bool(data) if isinstance(data, list) else bool(data)
        return bool(data)
    if isinstance(data, list):
        return len(data) > 0
    return True


def panels_to_domain_status(
    panels: Optional[List[Dict[str, Any]]],
    *,
    pipeline_warnings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Map validated auto_pipeline panels → domain_status for health_check.

    Status vocabulary matches ProjectHealthAggregator / dependency_graph:
    healthy | warning | failed | planned (and related). Empty panels omit
    the domain rather than inventing synthetic content.
    """
    domain_status: Dict[str, Dict[str, Any]] = {}
    warned_panels = {
        (w.get("panel") or "").strip()
        for w in (pipeline_warnings or [])
        if isinstance(w, dict)
    }

    for panel in panels or []:
        if not isinstance(panel, dict):
            continue
        ptype = panel.get("type") or ""
        domain = _PANEL_DOMAIN.get(ptype)
        if not domain:
            continue

        entry = domain_status.setdefault(
            domain,
            {
                "status": "planned",
                "count": 0,
                "blocked": 0,
                "overdue": 0,
                "high_risk": 0,
                "panels": [],
            },
        )
        entry["count"] += 1
        entry["panels"].append(ptype)

        has_error = bool(panel.get("error")) or ptype in warned_panels
        has_payload = _panel_has_payload(panel)

        if has_error:
            # Prefer failed for cost/quality-ish; warning otherwise
            if domain in ("cost", "quality", "safety"):
                entry["status"] = "failed"
            else:
                if entry["status"] not in ("failed", "critical"):
                    entry["status"] = "warning"
            entry["blocked"] += 1
        elif has_payload:
            if entry["status"] in ("planned",):
                entry["status"] = "healthy"
            # Long-lead procurement → warning signal without fabricating items
            if ptype == "procurement":
                data = panel.get("data") or {}
                critical = data.get("critical_long_lead_items") or []
                if critical and entry["status"] == "healthy":
                    entry["status"] = "warning"
                    entry["high_risk"] = max(entry["high_risk"], len(critical))
            if ptype == "risks":
                risks = panel.get("data") or []
                if isinstance(risks, list) and len(risks) >= 3:
                    entry["high_risk"] = max(entry["high_risk"], min(len(risks), 5))
                    if entry["status"] == "healthy":
                        entry["status"] = "warning"

    return domain_status
