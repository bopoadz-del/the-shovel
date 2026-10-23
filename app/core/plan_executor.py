"""Plan executor — Reasoning Engine Plan 5.

Runs an ExecutionPlan's steps against a ProjectSession. No AI: each step type
maps to a handler that calls a tested app/lib function (Plans 1/1b) or the
Plan-4 code generator, then writes the result into session.data.
"""

import json

from app.schemas.cpm import CPMInput
from app.schemas.execution_plan import (
    ExecutionPlan, PlanRunResult, PlanStep, StepResult,
)
from app.schemas.project_session import ProjectSession
from app.lib.pm_computations import (
    compute_cpm, resource_histogram, gantt_data, compress_schedule,
)


class PlanExecutionError(Exception):
    """A step could not run (bad args, missing state)."""


# Keys generated code may pull from session.data. The LLM controls
# `from_session`, so anything outside this allowlist is silently skipped.
_FROM_SESSION_ALLOWLIST = {
    "cpm_results", "manpower", "gantt", "compressed", "activities",
}

# Bound for a single step's `output` snapshot so the DELIVER prompt stays sane.
_STEP_OUTPUT_LIMIT = 2000


def _summarize_output(value):
    """Return `value` for the StepResult, replacing it with a compact summary
    string when its JSON form is large."""
    if value is None:
        return None
    try:
        rendered = json.dumps(value, default=str)
    except Exception:                                     # noqa: BLE001
        return str(value)[:_STEP_OUTPUT_LIMIT]
    if len(rendered) <= _STEP_OUTPUT_LIMIT:
        return value
    return rendered[:_STEP_OUTPUT_LIMIT] + "… (truncated)"


def _require_cpm_input(session: ProjectSession) -> CPMInput:
    activities = session.data.get("activities")
    if not activities:
        raise PlanExecutionError(
            "No activities in the session — load a schedule first."
        )
    return CPMInput.model_validate({"activities": activities})


class PlanExecutor:
    """Dispatches ExecutionPlan steps. Stateless — safe to reuse."""

    def __init__(self, code_block=None):
        """`code_block` is a FormulaExecutorV2Block (Plan 4). Constructed
        lazily on first use when not injected, so library-only plans need no
        LLM block."""
        self._code_block = code_block

    def _get_code_block(self):
        if self._code_block is None:
            from app.blocks.formula_executor_v2 import FormulaExecutorV2Block
            self._code_block = FormulaExecutorV2Block()
        return self._code_block

    async def run(
        self, plan: ExecutionPlan, session: ProjectSession
    ) -> PlanRunResult:
        results = []
        any_error = False
        for step in plan.steps:
            try:
                value = await self._run_step(step, session)
                results.append(StepResult(
                    type=step.type,
                    output_key=step.output_key,
                    status="success",
                    output=_summarize_output(value),
                ))
            except Exception as e:                       # noqa: BLE001
                any_error = True
                results.append(StepResult(
                    type=step.type,
                    output_key=step.output_key,
                    status="error",
                    error=str(e),
                ))
        if not any_error:
            status = "success"
        elif any(r.status == "success" for r in results):
            status = "partial"
        else:
            status = "error"
        return PlanRunResult(status=status, step_results=results)

    async def _run_step(self, step: PlanStep, session: ProjectSession):
        """Run one step. Returns the value written into session.data so the
        caller can record it on the StepResult."""
        handler = getattr(self, f"_step_{step.type}", None)
        if handler is None:
            raise PlanExecutionError(f"Unknown step type: '{step.type}'")
        return await handler(step, session)

    # ── library step handlers ────────────────────────────────────────────
    async def _step_compute_cpm(self, step, session):
        out = compute_cpm(_require_cpm_input(session))
        value = out.model_dump(mode="json")
        session.data[step.output_key or "cpm_results"] = value
        return value

    async def _step_resource_histogram(self, step, session):
        try:
            cpm_input = _require_cpm_input(session)
        except Exception:
            # Session activities came from generate_wbs (histogram-from-brief
            # plan), whose shape predates the CPM schema. Bridge them the same
            # way cost_load does, then express crews as a single General
            # resource so the synthetic-distribution path time-phases them.
            from pydantic import ValidationError  # noqa: F401 — clarity
            from app.lib.schedule_bridge import bridge_wbs_to_cost_loaded
            acts = (session.data.get("wbs") or {}).get("activities") \
                or session.data.get("activities") or []
            if not acts:
                raise PlanExecutionError(
                    "No activities in the session — run build_wbs first.")
            bridged = bridge_wbs_to_cost_loaded(
                acts, crew_per_trade=step.args.get("crew_per_trade", 4))
            cpm_input = CPMInput.model_validate({"activities": [
                {
                    "id": b["id"], "name": b["name"], "duration": b["duration"],
                    "wbs_code": b.get("wbs", ""),
                    "predecessors": [{"predecessor_id": p} for p in b["predecessors"]],
                    "resources": [{"trade": "General", "count": b["manpower"]}],
                }
                for b in bridged if b.get("id")
            ]})
        out = compute_cpm(cpm_input)
        hist = resource_histogram(
            out.results, cpm_input.activities,
            period_unit=step.args.get("period_unit", "week"),
        )
        value = hist.model_dump(mode="json")
        session.data[step.output_key or "manpower"] = value
        return value

    async def _step_gantt(self, step, session):
        out = compute_cpm(_require_cpm_input(session))
        bars = gantt_data(out.results)
        value = [b.model_dump(mode="json") for b in bars]
        session.data[step.output_key or "gantt"] = value
        return value

    async def _step_compress(self, step, session):
        reductions = step.args.get("reductions")
        if not isinstance(reductions, dict):
            raise PlanExecutionError(
                "compress step needs an args.reductions dict"
            )
        reductions = {k: int(v) for k, v in reductions.items()}
        revised, days_saved = compress_schedule(
            _require_cpm_input(session), reductions
        )
        value = {
            "revised": revised.model_dump(mode="json"),
            "days_saved": days_saved,
        }
        session.data[step.output_key or "compressed"] = value
        return value

    # ── schedule-workflow step handlers (predefined reasoning) ───────────
    # These call the SAME blocks/libs the export endpoints call (one execution
    # surface), writing staged state into session.data / session.artifacts.
    async def _step_extract_document(self, step, session):
        """document_engine over RFP/BOD docs -> real lead times + milestones."""
        from app.lib.schedule_feed import extract_schedule_feed
        doc_ids = step.args.get("document_ids") or []
        lead_times, milestones = await extract_schedule_feed(doc_ids)
        session.data["procurement_lead_times"] = lead_times
        session.data["target_milestones"] = milestones
        value = {"lead_times": lead_times, "target_milestones": milestones}
        session.data[step.output_key or "schedule_feed"] = value
        return value

    async def _step_build_wbs(self, step, session):
        """generate_wbs, consuming any lead times/milestones the extract step
        staged. Writes the WBS + its activities into the session."""
        from app.containers.construction import ConstructionContainer
        params = {
            "brief": step.args.get("brief") or "",
            "target_count": step.args.get("target_count", 200),
            "project_type": step.args.get("project_type"),
            "start_date": step.args.get("start_date"),
            "user_message": step.args.get("user_message") or step.args.get("brief") or "",
            "duration_overrides": step.args.get("duration_overrides"),
            "procurement_lead_times": session.data.get("procurement_lead_times") or [],
            "target_milestones": session.data.get("target_milestones") or [],
            "project_id": step.args.get("project_id") or session.data.get("project_id"),
            "boq_items": step.args.get("boq_items"),
        }
        wbs = await ConstructionContainer().generate_wbs({}, params)
        session.data["wbs"] = wbs
        session.data["activities"] = wbs.get("activities") or []
        value = {
            "actual_count": wbs.get("actual_count"),
            "duration_days": (wbs.get("summary") or {}).get("total_duration_days"),
            "procurement_injected": wbs.get("procurement_injected"),
            "duration_overrides_applied": wbs.get("duration_overrides_applied") or [],
        }
        session.data[step.output_key or "wbs_summary"] = value
        return value

    async def _step_cost_load(self, step, session):
        """Bridge the WBS -> cost-loaded workbook. Builds + STAGES it (temp
        path + summary in session.data); does not expose it. render_artifact
        does the exposing — so a plan without render_artifact answers inline."""
        import os as _os, tempfile
        from app.lib.schedule_bridge import bridge_wbs_to_cost_loaded
        from app.lib.pm_excel import generate_cost_loaded_schedule
        wbs = session.data.get("wbs") or {}
        acts = wbs.get("activities") or session.data.get("activities") or []
        if not acts:
            raise PlanExecutionError("no WBS activities staged — run build_wbs first")
        day_rate = step.args.get("day_rate")
        bridged = bridge_wbs_to_cost_loaded(
            acts, crew_per_trade=step.args.get("crew_per_trade", 4), day_rate=day_rate)
        meta = {"project": step.args.get("project_name") or "Project",
                "currency": step.args.get("currency", "SAR")}
        sd = step.args.get("start_date") or wbs.get("start_date")
        if sd:
            meta["start_date"] = sd
        if day_rate:
            meta["cost_basis"] = "Indicative Labor"
        milestones = session.data.get("target_milestones") or []
        if milestones:
            meta["target_milestones"] = milestones
        wb = generate_cost_loaded_schedule(meta, bridged)
        fd, path = tempfile.mkstemp(prefix="sched_plan_", suffix=".xlsx"); _os.close(fd)
        wb.save(path)
        total_mandays = sum(int(a.get("manpower", 0) or 0) * int(a.get("duration", 0) or 0)
                            for a in bridged)
        summary = {
            "duration_days": (wbs.get("summary") or {}).get("total_duration_days"),
            "critical_count": (wbs.get("summary") or {}).get("critical_count"),
            "procurement_injected": wbs.get("procurement_injected", 0),
            "activities": wbs.get("actual_count") or len(acts),
            "total_man_days": total_mandays,
            "target_milestones": milestones,
            "duration_overrides_applied": wbs.get("duration_overrides_applied") or [],
        }
        session.data["schedule_workbook_path"] = path
        session.data["schedule_summary"] = summary
        session.data[step.output_key or "cost_load"] = summary
        return summary

    async def _step_render_artifact(self, step, session):
        """MATERIALIZE the staged workbook — expose it as a session artifact.
        Only included in a plan when the intent asks for a deliverable."""
        from app.schemas.project_session import Artifact
        path = session.data.get("schedule_workbook_path")
        if not path:
            raise PlanExecutionError("no staged workbook — run cost_load first")
        name = step.args.get("name") or "Schedule.xlsx"
        art = Artifact(name=name, path=path, type=step.args.get("type", "excel"))
        session.artifacts.append(art)
        value = {"name": name, "path": path, "type": art.type}
        session.data[step.output_key or "artifact"] = value
        return value

    async def _step_deliver(self, step, session):
        """Caller-handled delivery step (NO_AUTO_DISPATCH / needs_caller_render).

        Marks the plan as ready for the caller to materialize exports. Does
        not invent files — only records intent when an artifact was staged.
        """
        arts = list(getattr(session, "artifacts", None) or [])
        value = {
            "needs_caller_render": True,
            "artifact_count": len(arts),
            "artifacts": [
                {"name": getattr(a, "name", None), "path": getattr(a, "path", None)}
                for a in arts[:10]
            ],
        }
        session.data[step.output_key or "deliver"] = value
        return value

    async def run_cm_plan_steps(self, steps, session):
        """Execute MultiDomainPlanBuilder steps that PlanExecutor can handle.

        Construction-block steps that need files are skipped with
        ``status=skipped`` (not fabricated). ``needs_caller_render`` /
        ``dispatch=False`` steps map to ``render_artifact`` / ``deliver``
        handlers when the step_type matches.
        """
        from app.schemas.execution_plan import PlanStep

        results = []
        for raw in steps:
            step_type = raw.get("step_type") or (raw.get("params") or {}).get("action")
            if not step_type:
                results.append({"step_type": None, "status": "skipped", "reason": "no step_type"})
                continue
            if raw.get("needs_caller_render") or raw.get("dispatch") is False:
                handler = getattr(self, f"_step_{step_type}", None)
                if handler is None:
                    results.append({
                        "step_type": step_type,
                        "status": "needs_caller_render",
                        "reason": "NO_AUTO_DISPATCH — caller must materialize",
                    })
                    continue
                try:
                    pseudo = PlanStep(
                        type=step_type,
                        args=dict(raw.get("params") or {}),
                        description=raw.get("description") or "",
                    )
                    value = await handler(pseudo, session)
                    results.append({
                        "step_type": step_type,
                        "status": "success",
                        "output": value,
                    })
                except Exception as exc:  # noqa: BLE001
                    results.append({
                        "step_type": step_type,
                        "status": "error",
                        "error": str(exc),
                    })
                continue
            handler = getattr(self, f"_step_{step_type}", None)
            if handler is None:
                results.append({
                    "step_type": step_type,
                    "status": "skipped",
                    "reason": "no PlanExecutor handler — use ConstructionContainer.route",
                    "block": raw.get("block"),
                    "action": (raw.get("params") or {}).get("action"),
                })
                continue
            try:
                pseudo = PlanStep(
                    type=step_type,
                    args=dict(raw.get("params") or {}),
                )
                value = await handler(pseudo, session)
                results.append({
                    "step_type": step_type,
                    "status": "success",
                    "output": value,
                })
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "step_type": step_type,
                    "status": "error",
                    "error": str(exc),
                })
        any_error = any(r.get("status") == "error" for r in results)
        any_ok = any(r.get("status") == "success" for r in results)
        if not any_error:
            status = "success"
        elif any_ok:
            status = "partial"
        else:
            status = "error"
        return {"status": status, "step_results": results}

    # ── code-generation step handler (delegates to Plan 4) ───────────────
    async def _step_generate_code(self, step, session):
        task = step.args.get("task")
        if not task:
            raise PlanExecutionError("generate_code step needs an args.task")
        variables = dict(step.args.get("variables") or {})
        # let generated code read prior session state when asked — but only
        # from known computed keys, since the LLM controls `from_session`
        for key in step.args.get("from_session", []):
            if key in _FROM_SESSION_ALLOWLIST and key in session.data:
                variables[key] = session.data[key]
        out = await self._get_code_block().process({
            "task": task, "variables": variables, "session": session,
        })
        if out.get("status") != "success":
            raise PlanExecutionError(
                out.get("error", "code generation failed")
            )
        session.data[step.output_key or "generated"] = out
        return out