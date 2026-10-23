"""Panel contract — the shape every UI renderer relies on.

The React dashboard consumes objects shaped like `Panel`. Backend code that
emits panels must construct them through these models so a shape regression
fails fast at the source instead of silently rendering garbage in the UI.

History of why this matters:
- `panel.line_items` vs `panel.data.procurement_list` — silent UI breakage.
- `parse_primavera_schedule` returning `{status:"error","Unsupported format"}`
  rendered as raw JSON in the schedule panel.
- `panel.type == "schedule"` for both .xer and .xlsx with different inner
  shapes — frontend had to sniff `data.format`.

The models below standardize all of that. Each panel has a fixed `type` plus
a typed `data`. Errors get their own `error: True | False` flag so the
renderer can branch cleanly without sniffing nested keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _PanelBase(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate extra fields during migration

    title: str
    error: bool = False
    error_message: Optional[str] = None


# ── Document info ─────────────────────────────────────────────────────────
class DocumentInfoData(BaseModel):
    file: Optional[str] = None
    doc_type: Optional[str] = None
    status: Optional[str] = None
    pages: Optional[int] = None
    title: Optional[str] = None
    project: Optional[str] = None


class DocumentInfoPanel(_PanelBase):
    type: Literal["document_info"] = "document_info"
    data: DocumentInfoData


# ── Quantities ────────────────────────────────────────────────────────────
# The data is a free-form dict of `<material>: <quantity>` because the
# whitelist filter in _calculate_quantities determines what's emitted.
class QuantitiesPanel(_PanelBase):
    type: Literal["quantities"] = "quantities"
    data: Dict[str, Union[int, float, Dict[str, Any]]]


# ── Cost estimate ─────────────────────────────────────────────────────────
class CostEstimateData(BaseModel):
    subtotal: float = 0.0
    overhead: float = 0.0
    contingency: float = 0.0
    total_estimate: float = 0.0


class CostEstimatePanel(_PanelBase):
    type: Literal["cost_estimate"] = "cost_estimate"
    data: CostEstimateData
    line_items: List[Dict[str, Any]] = Field(default_factory=list)


# ── Procurement ───────────────────────────────────────────────────────────
class ProcurementItem(BaseModel):
    item: str
    quantity: float
    unit: str = "ea"
    unit_cost: float = 0.0
    total_cost: float = 0.0
    category: Optional[str] = None
    lead_time_weeks: int = 0
    priority: Literal["critical", "high", "normal"] = "normal"
    supplier_type: Optional[str] = None
    order_date: Optional[str] = None


class ProcurementData(BaseModel):
    procurement_list: List[ProcurementItem] = Field(default_factory=list)
    total_items: int = 0
    total_procurement_cost: float = 0.0
    critical_long_lead_items: int = 0
    action_required: List[str] = Field(default_factory=list)


class ProcurementPanel(_PanelBase):
    type: Literal["procurement"] = "procurement"
    data: ProcurementData


# ── Schedule (Primavera) and Schedule (Excel) ────────────────────────────
class XlsxSheetSummary(BaseModel):
    name: str
    row_count: int
    col_count: int
    preview: List[List[Any]] = Field(default_factory=list)


class XlsxScheduleData(BaseModel):
    format: Literal["xlsx"] = "xlsx"
    file: str
    sheets: List[XlsxSheetSummary] = Field(default_factory=list)
    schedule_targets: List[Any] = Field(default_factory=list)
    equipment_specs: List[Any] = Field(default_factory=list)
    constraints: List[Any] = Field(default_factory=list)
    requirements_count: int = 0


class PrimaveraScheduleData(BaseModel):
    format: Literal["xer"] = "xer"
    activity_count: Optional[int] = None
    critical_path_count: Optional[int] = None
    milestone_count: Optional[int] = None
    project_start: Optional[str] = None
    project_finish: Optional[str] = None


class SchedulePanel(_PanelBase):
    type: Literal["schedule"] = "schedule"
    # Discriminate by `format` field at the data level
    data: Union[XlsxScheduleData, PrimaveraScheduleData, Dict[str, Any]]


# ── Risks / Submittals / Contract ─────────────────────────────────────────
class RiskItem(BaseModel):
    description: str
    likelihood: Literal["Low", "Medium", "High"] = "Medium"
    impact: Literal["Low", "Medium", "High"] = "Medium"


class RisksPanel(_PanelBase):
    type: Literal["risks"] = "risks"
    data: List[RiskItem]
    total: Optional[int] = None


class SubmittalsPanel(_PanelBase):
    type: Literal["submittals"] = "submittals"
    data: List[Dict[str, Any]]
    total: Optional[int] = None


class ContractPanel(_PanelBase):
    type: Literal["contract"] = "contract"
    data: Dict[str, Any]


# ── Discriminated union (for typing; actual emission uses dict()) ────────
Panel = Union[
    DocumentInfoPanel,
    QuantitiesPanel,
    CostEstimatePanel,
    ProcurementPanel,
    SchedulePanel,
    RisksPanel,
    SubmittalsPanel,
    ContractPanel,
]


# ── Helpers used by callers (auto_pipeline etc.) ──────────────────────────
def make_error_panel(panel_type: str, title: str, message: str) -> Dict[str, Any]:
    """Construct an error-flavored panel of the given type without raising."""
    return {
        "type": panel_type,
        "title": title,
        "error": True,
        "error_message": message,
        "data": {},
    }


def validate_panel(panel: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort validation: returns the panel dict unchanged if valid;
    on validation failure, returns a typed error_panel of the same type with the message.

    We do not raise — backend regression should not break the UI; it should
    render an obvious red banner instead.
    """
    panel_type = panel.get("type")
    type_to_model = {
        "document_info": DocumentInfoPanel,
        "quantities": QuantitiesPanel,
        "cost_estimate": CostEstimatePanel,
        "procurement": ProcurementPanel,
        "schedule": SchedulePanel,
        "risks": RisksPanel,
        "submittals": SubmittalsPanel,
        "contract": ContractPanel,
    }
    Model = type_to_model.get(panel_type or "")
    if not Model:
        return panel  # unknown panel type — pass through, frontend has a default branch
    try:
        Model.model_validate(panel)
        return panel
    except Exception as e:
        return make_error_panel(panel_type or "unknown", panel.get("title", panel_type or "Panel"), f"Panel shape invalid: {e}")
