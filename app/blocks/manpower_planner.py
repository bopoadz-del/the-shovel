"""Manpower Planner block — roadmap-named shim over app.lib.pm_computations.

The L2 Schedule Engine roadmap expected ``app/blocks/manpower_planner.py``. The
trade-based resource histogram lives in
``app.lib.pm_computations.resource_histogram``. This block is a thin wrapper
that exposes the same capability under the roadmap name.
"""

from typing import Any, Dict, List, Optional

from app.core.universal_base import UniversalBlock


def _normalize_activity(a: Any) -> Any:
    """Coerce BOQ / generate_wbs activity shapes into CPM ``Activity`` input.

    The schema models crews as ``resources=[{trade, count}]`` and duration as
    ``duration``. Callers naturally write ``manpower`` / ``crew_size`` and
    ``duration_days``, and predecessors as bare id strings. Pydantic v2 then
    either drops the unknown keys (histogram all-zero, status success) or
    rejects the payload (422 on ``/v1/schedule/manpower``).
    """
    if not isinstance(a, dict):
        return a
    out = dict(a)
    if out.get("duration") is None and out.get("duration_days") is not None:
        try:
            out["duration"] = int(out["duration_days"])
        except (TypeError, ValueError):
            pass
    preds = out.get("predecessors") or []
    coerced: List[Any] = []
    for p in preds:
        if isinstance(p, str):
            coerced.append({"predecessor_id": p})
        elif isinstance(p, dict) and "predecessor_id" not in p and p.get("id"):
            coerced.append({**p, "predecessor_id": p["id"]})
        else:
            coerced.append(p)
    out["predecessors"] = coerced
    resources = out.get("resources")
    if isinstance(resources, list) and resources and isinstance(resources[0], str):
        mp = out.get("manpower") or out.get("crew") or out.get("crew_size") or 1
        out["resources"] = [{"trade": t, "count": float(mp)} for t in resources]
        return out
    if not resources:
        mp = (out.get("manpower") or out.get("crew") or out.get("crew_size")
              or out.get("headcount"))
        if mp:
            out["resources"] = [
                {"trade": out.get("trade") or "General", "count": float(mp)}
            ]
    return out


class ManpowerPlannerBlock(UniversalBlock):
    auto_validate = False
    name = "manpower_planner"
    version = "1.0.0"
    description = "Build a trade-based manpower histogram from CPM results"
    layer = 3
    tags = ["domain", "construction", "schedule", "manpower", "histogram"]
    requires = []

    default_config = {
        "default_period_unit": "week",
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Compute a trade-based manpower histogram.

        Accepts either:
          * ``activities`` + ``results`` (CPM output already computed)
          * ``cpm_input`` dict -> the block runs CPM first, then histograms
          * ``task_resources`` (TASKRSRC-shaped list) for real P6 resource loading
        """
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        activities = data.get("activities") or params.get("activities") or []
        results = data.get("results") or params.get("results") or []
        task_resources = data.get("task_resources") or params.get("task_resources") or []
        period_unit = (
            data.get("period_unit")
            or params.get("period_unit")
            or self.config.get("default_period_unit", "week")
        )

        if not activities:
            return {"status": "error", "error": "No activities provided"}

        try:
            from app.lib.pm_computations import compute_cpm, resource_histogram
            from app.schemas.cpm import Activity, CPMInput

            # CPM Activity wants duration + resources=[{trade,count}] +
            # predecessors=[{predecessor_id}]. BOQ-derived schedules emit
            # duration_days, crew_size/manpower, and predecessor id strings,
            # which used to 422 the /v1/schedule/manpower shim.
            activities = [_normalize_activity(a) for a in activities]
            act_objects = [
                Activity(**a) if isinstance(a, dict) else a for a in activities
            ]
            unresourced = not task_resources and not any(
                getattr(a, "resources", None) for a in act_objects
            )

            if not results and act_objects:
                cpm_output = compute_cpm(CPMInput(activities=act_objects))
                result_objects = list(cpm_output.results)
            else:
                from app.schemas.cpm import CPMResult

                result_objects = [
                    CPMResult(**r) if isinstance(r, dict) else r for r in results
                ]

            histogram = resource_histogram(
                results=result_objects,
                activities=act_objects,
                period_unit=period_unit,
                task_resources=task_resources or None,
            )
            out = {
                "status": "success",
                "period_unit": histogram.period_unit,
                "periods": [p.model_dump() for p in histogram.periods],
                "peak_total": histogram.peak_total,
                "peak_period": histogram.peak_period,
                "by_trade_totals": histogram.by_trade_totals,
                "total_manhours": histogram.total_manhours,
            }
            if unresourced:
                out["warning"] = (
                    "no resources on any activity -- histogram is all zeros; "
                    "provide activities[].resources=[{trade, count}] (or "
                    "manpower/crew/crew_size/headcount) or task_resources"
                )
            return out
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"Manpower planning failed: {exc}"}
