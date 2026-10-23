"""Primavera Parser Block - Parse Oracle Primavera P6 XER schedule files"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from app.core.universal_base import UniversalBlock

_logger = logging.getLogger(__name__)


class PrimaveraParserBlock(UniversalBlock):
    auto_validate = False
    name = "primavera_parser"
    version = "1.0.0"
    description = "Parse Primavera P6 .xer schedule files: low-float activities, milestones, resource definitions"
    layer = 3
    tags = ["domain", "construction", "schedule", "primavera", "xer", "cpm"]
    requires = []

    default_config = {
        "include_resources": True,
        "critical_float_days": 0,
    }

    ui_schema = {
        "input": {
            "type": "file",
            "accept": [".xer"],
            "placeholder": "Upload Primavera P6 .xer schedule file...",
        },
        "output": {
            "type": "table",
            "fields": [
                {"name": "activity_count", "type": "number", "label": "Activities"},
                {"name": "low_float_activities", "type": "list", "label": "Low-Float Activities"},
                {"name": "milestones", "type": "list", "label": "Milestones"},
                {"name": "schedule_data", "type": "json", "label": "Schedule Summary"},
            ],
        },
        "quick_actions": [
            {"icon": "️", "label": "Low-Float Activities", "prompt": "Show activities with total float at or below the threshold"},
            {"icon": "", "label": "Milestones", "prompt": "List all project milestones"},
            {"icon": "", "label": "Resource Definitions", "prompt": "Show resource definitions from the RSRC table"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        file_path = data.get("file_path") or params.get("file_path") or data.get("text") or data.get("input") or (input_data if isinstance(input_data, str) else "")
        if not file_path:
            return {"status": "error", "error": "No file_path provided — requires a Primavera P6 .xer file"}
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}
        if not file_path.lower().endswith(".xer"):
            return {"status": "error", "error": "File must be a .xer Primavera export"}

        try:
            # open_plaintext transparently decrypts when DATA_ENCRYPTION_KEY is
            # set; plaintext files short-circuit. Primavera P6 exports XER in
            # Windows-1252; UTF-8 mangles Arabic / European chars.
            from app.core.file_crypto import open_plaintext
            with open_plaintext(file_path) as plain_path:
                with open(plain_path, "r", encoding="cp1252", errors="replace") as f:
                    xer_text = f.read()
            tables = self._parse_xer_text(xer_text)
        except Exception as e:
            return {"status": "error", "error": f"XER parse error: {e}"}

        float_threshold = int(
            params.get("critical_float_days", self.config.get("critical_float_days", 0))
        )
        include_resources = params.get(
            "include_resources", self.config.get("include_resources", True)
        )

        activities = self._extract_activities(tables)
        wbs = self._extract_wbs(tables)
        resource_definitions = self._extract_resources(tables) if include_resources else []
        project_meta = self._extract_project(tables)

        # Low-float filter remains a useful display field, but it is not the
        # CPM critical path. The real driving-path analysis below runs the
        # forward+backward pass via app.lib.pm_computations.
        low_float_activities = [
            a for a in activities
            if a.get("total_float_days", 999) <= float_threshold
        ]
        milestones = [
            a for a in activities
            if a.get("type") in ("TT_Mile", "TT_FinMile", "TT_StartMile", "milestone")
        ]

        # Project date range
        start_dates = [a["start"] for a in activities if a.get("start")]
        end_dates = [a["finish"] for a in activities if a.get("finish")]
        project_start = min(start_dates) if start_dates else None
        project_finish = max(end_dates) if end_dates else None

        schedule_data = {
            "project": project_meta,
            "activity_count": len(activities),
            "wbs_count": len(wbs),
            "resource_definition_count": len(resource_definitions),
            "project_start": project_start,
            "project_finish": project_finish,
            "low_float_activity_count": len(low_float_activities),
            "milestone_count": len(milestones),
        }

        # Response payload caps. PR #3 flagged the original hardcoded
        # ``activities[:200]`` as a known follow-up — it silently
        # truncated real-world P6 schedules (often 1000+ activities) so
        # downstream CPM consumers saw ~80 % of the work disappear. The
        # caps are now generous defaults and operator-configurable per
        # call via ``params``; ``max_*=0`` returns the full list.
        #
        # ``activity_count`` already reports the true total so callers
        # can detect truncation. The defaults below cover real EPC-scale
        # schedules without forcing every caller to opt in.
        # Codex P2 fix on PR #28: the earlier implementation returned
        # ``len(activities)`` for the "unlimited" (0) case regardless of
        # which list the cap would be applied to. For an XER whose RSRC
        # table has more rows than TASK, ``resource_definitions[:max_resources]``
        # with ``max_resources == len(activities)`` STILL truncated.
        # Returning None means "no cap" — ``lst[:None]`` is the full list in
        # Python, so the slice always reflects the caller's actual list size.
        def _cap(name: str, default: int) -> Optional[int]:
            try:
                v = int(params.get(name, default))
            except (TypeError, ValueError):
                v = default
            return v if v > 0 else None  # 0 → unlimited (lst[:None] == lst)

        max_activities = _cap("max_activities", 5000)
        max_wbs = _cap("max_wbs", 2000)
        max_low_float = _cap("max_low_float_activities", 500)
        max_milestones = _cap("max_milestones", 500)
        max_resources = _cap("max_resources", 200)

        response: Dict[str, Any] = {
            "status": "success",
            "schedule_data": schedule_data,
            "low_float_activities": low_float_activities[:max_low_float],
            "_note": "computed via real CPM forward+backward pass",
            "milestones": milestones[:max_milestones],
            "activities": activities[:max_activities],
            "wbs": wbs[:max_wbs],
            "resource_definitions": resource_definitions[:max_resources],
            "activity_count": len(activities),
        }
        # Say TRUNCATED out loud. activity_count already carried the true
        # total, but detecting the cap required a caller to compare it against
        # len(activities) -- and a live run against an 11,149-task baseline
        # was nearly reported as "5,000 activities" because nothing stated the
        # list was cut. A cap that has to be inferred reads as completeness.
        response["activities_returned"] = len(response["activities"])
        response["activities_truncated"] = (
            len(response["activities"]) < len(activities)
        )
        if response["activities_truncated"]:
            response["truncation_note"] = (
                f"activities list capped at {len(response['activities'])} of "
                f"{len(activities)} (raise params.max_activities or pass 0 for "
                "all); CPM below is computed on the FULL network regardless."
            )

        # ── Real CPM via app.lib.pm_computations ─────────────────────────
        # The block's "activities" list keys by task_id (numeric P6 row id);
        # the CPM library keys by task_code (the human "A1010"-style id).
        # The CPM output below therefore references task_codes — callers can
        # join via the "code" field on each activity dict.
        try:
            from app.lib.pm_computations import compute_cpm, parse_xer_full
            from app.schemas.cpm import CPMInput

            parsed = parse_xer_full(xer_text)
            cpm_activities = parsed.get("activities") or []
            if cpm_activities:
                cpm_input = CPMInput(activities=cpm_activities)
                cpm_out = compute_cpm(cpm_input)
                per_activity_float = {
                    r.id: r.total_float for r in cpm_out.results
                }
                near_critical_ids = [
                    r.id for r in cpm_out.results
                    if not r.is_critical and 0 < r.total_float <= 5
                ]
                response["cpm"] = {
                    "total_duration_days": cpm_out.project_duration,
                    "critical_path_activity_ids": list(cpm_out.critical_path),
                    "near_critical_activity_ids": near_critical_ids,
                    "per_activity_float": per_activity_float,
                    "id_space": "task_code (matches activities[].code)",
                }
            else:
                response["cpm"] = {
                    "total_duration_days": 0,
                    "critical_path_activity_ids": [],
                    "near_critical_activity_ids": [],
                    "per_activity_float": {},
                }
            # Surface calendar + TASKRSRC parse results for downstream callers.
            response["calendars_parsed"] = parsed.get("calendars_parsed", {})
            response["task_resources_count"] = len(parsed.get("task_resources") or [])
        except Exception as cpm_exc:  # noqa: BLE001 — never crash the parser
            _logger.warning(
                "primavera_parser: CPM computation failed (%s); "
                "falling back to low-float filter only",
                cpm_exc,
            )
            response["cpm_error"] = f"{type(cpm_exc).__name__}: {cpm_exc}"
            # Preserve the original semantic of the legacy field for the fallback.
            response["_note"] = (
                "low_float_activities is a total_float filter, "
                "not a CPM driving-path analysis (CPM failed; see cpm_error)"
            )

        return response

    def _parse_xer(self, file_path: str) -> Dict[str, List[Dict]]:
        """Parse XER tab-delimited table format into dict of table_name → rows.

        Retained as a convenience for callers that pass a path directly; the
        new ``process()`` flow reads the text once and uses
        :meth:`_parse_xer_text` to avoid a second file open.
        """
        # Primavera P6 exports XER files in Windows-1252 by default; UTF-8 mangles Arabic / European chars.
        with open(file_path, "r", encoding="cp1252", errors="replace") as f:
            return self._parse_xer_text(f.read())

    def _parse_xer_text(self, text: str) -> Dict[str, List[Dict]]:
        """Parse XER tab-delimited table text into dict of table_name → rows."""
        tables: Dict[str, List[Dict]] = {}
        current_table: Optional[str] = None
        headers: List[str] = []

        for line in text.splitlines():
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("%T"):
                current_table = line[3:].strip()
                tables[current_table] = []
                headers = []
            elif line.startswith("%F"):
                headers = line[3:].split("\t")
            elif line.startswith("%R") and current_table and headers:
                values = line[3:].split("\t")
                row = dict(zip(headers, values + [""] * max(0, len(headers) - len(values))))
                tables[current_table].append(row)
        return tables

    def _extract_activities(self, tables: Dict) -> List[Dict]:
        rows = tables.get("TASK", [])
        activities = []
        for r in rows:
            tf = r.get("total_float_hr_cnt", "")
            try:
                float_days = float(tf) / 8.0 if tf else 999
            except ValueError:
                float_days = 999

            activities.append({
                "id": r.get("task_id", ""),
                "code": r.get("task_code", ""),
                "name": r.get("task_name", ""),
                "type": r.get("task_type", ""),
                "status": r.get("status_code", ""),
                "start": _parse_date(r.get("act_start_date") or r.get("early_start_date", "")),
                "finish": _parse_date(r.get("act_end_date") or r.get("early_end_date", "")),
                "original_duration_days": _orig_dur_days(r),
                "remaining_duration_days": _to_float(r.get("remain_drtn_hr_cnt", "0")) / 8,
                "total_float_days": round(float_days, 1),
                "percent_complete": _to_float(r.get("phys_complete_pct", "0")),
                "wbs_id": r.get("wbs_id", ""),
            })
        return sorted(activities, key=lambda x: x.get("start") or "9999")

    def _extract_wbs(self, tables: Dict) -> List[Dict]:
        rows = tables.get("PROJWBS", [])
        return [
            {
                "id": r.get("wbs_id", ""),
                "code": r.get("wbs_short_name", ""),
                "name": r.get("wbs_name", ""),
                "parent_id": r.get("parent_wbs_id", ""),
                "level": int(_to_float(r.get("seq_num", "0"))),
            }
            for r in rows
        ]

    def _extract_resources(self, tables: Dict) -> List[Dict]:
        rows = tables.get("RSRC", [])
        return [
            {
                "id": r.get("rsrc_id", ""),
                "name": r.get("rsrc_name", ""),
                "short_name": r.get("rsrc_short_name", ""),
                "type": r.get("rsrc_type", ""),
                "unit_type": r.get("unit_id", ""),
            }
            for r in rows
        ]

    def _extract_project(self, tables: Dict) -> Dict:
        rows = tables.get("PROJECT", [])
        if not rows:
            return {}
        r = rows[0]
        # P6 data date is last_recalc_date when present; plan_start is only a
        # fallback for look-ahead / as-of defaults.
        data_date = (
            _parse_date(r.get("last_recalc_date", ""))
            or _parse_date(r.get("plan_start_date", ""))
        )
        return {
            "id": r.get("proj_id", ""),
            "name": r.get("proj_short_name", ""),
            "planned_start": _parse_date(r.get("plan_start_date", "")),
            "data_date": data_date,
            "must_finish": _parse_date(r.get("scd_end_date", "")),
            "status": r.get("status_code", ""),
        }


def _parse_date(val: str) -> Optional[str]:
    if not val or val.strip() == "":
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d-%b-%y %H:%M", "%d-%b-%y"):
        try:
            return datetime.strptime(val.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return val.strip() or None


def _to_float(val: str) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _orig_dur_days(row: Dict) -> float:
    """Original duration in days from P6 row.

    Real P6 column is ``target_drtn_hr_cnt`` (hours). Fall back to legacy ``orig_dur``
    for backward compat.
    """
    raw = row.get("target_drtn_hr_cnt")
    if raw is None or str(raw).strip() == "":
        raw = row.get("orig_dur", "0")
    try:
        return float(str(raw).replace(",", "").strip()) / 8.0
    except (ValueError, TypeError):
        return 0.0
