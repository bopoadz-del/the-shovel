"""Fast-Track Analyzer block — roadmap-named shim over app.lib.pm_computations.

The L2 Schedule Engine roadmap expected ``app/blocks/fasttrack_analyzer.py``.
Schedule compression / crash analysis lives in
``app.lib.pm_computations.compress_schedule``. This block is a thin wrapper
that exposes the same capability under the roadmap name and adds the narrative
scenarios produced by the construction container's recovery generator.
"""

from typing import Any, Dict, List

from app.core.universal_base import UniversalBlock


class FastTrackAnalyzerBlock(UniversalBlock):
    auto_validate = False
    name = "fasttrack_analyzer"
    version = "1.0.0"
    description = "Fast-track / crash compression analysis for CPM networks"
    layer = 3
    tags = ["domain", "construction", "schedule", "fast-track", "crash"]
    requires = []

    default_config = {
        "crash_savings_ratio": 0.5,
        "fast_track_savings_ratio": 0.3,
        "scope_reduction_savings_ratio": 0.5,
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Analyze compression options for a CPM network.

        Input fields:
          * ``activities``: list of activity dicts
          * ``reductions``: dict mapping activity id -> working days to remove
          * ``total_delay_days``: optional delay used to build narrative scenarios
        """
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        activities = data.get("activities") or params.get("activities") or []
        reductions = data.get("reductions") or params.get("reductions") or {}
        total_delay = data.get("total_delay_days") or params.get("total_delay_days") or 0

        if not activities:
            return {"status": "error", "error": "No activities provided"}
        if not reductions:
            return {"status": "error", "error": "No reductions provided"}
        # A list (or any non-mapping) used to slip past the truthiness check
        # and crash deep in compress_schedule with "unhashable type: 'dict'"
        # -- state the contract instead of leaking an internals error.
        if not isinstance(reductions, dict):
            return {
                "status": "error",
                "error": (
                    "reductions must be a mapping of activity id -> working "
                    'days to remove, e.g. {"B": 4}'
                ),
            }
        bad = {k: v for k, v in reductions.items() if not isinstance(v, (int, float)) or v < 0}
        if bad:
            return {
                "status": "error",
                "error": f"reductions values must be non-negative day counts; got {bad}",
            }

        try:
            from app.lib.pm_computations import compress_schedule, compute_cpm
            from app.schemas.cpm import Activity, CPMInput

            act_objects = [
                Activity(**a) if isinstance(a, dict) else a for a in activities
            ]
            cpm_input = CPMInput(activities=act_objects)
            baseline = compute_cpm(cpm_input)
            revised, days_saved = compress_schedule(cpm_input, reductions)

            ratios = {
                "crash": self.config.get("crash_savings_ratio", 0.5),
                "fast_track": self.config.get("fast_track_savings_ratio", 0.3),
                "scope_reduction": self.config.get("scope_reduction_savings_ratio", 0.5),
            }
            scenarios = [
                {
                    "strategy": "Crash Critical Path",
                    "description": "Add resources to critical activities",
                    "potential_savings_days": round(total_delay * ratios["crash"], 1),
                    "cost_impact": "High",
                    "feasibility": "Medium",
                },
                {
                    "strategy": "Fast Track",
                    "description": "Overlap sequential activities",
                    "potential_savings_days": round(total_delay * ratios["fast_track"], 1),
                    "cost_impact": "Medium",
                    "feasibility": "Medium",
                },
                {
                    "strategy": "Scope Reduction",
                    "description": "Defer non-critical scope to later phase",
                    "potential_savings_days": round(total_delay * ratios["scope_reduction"], 1),
                    "cost_impact": "Low",
                    "feasibility": "High",
                },
            ]

            return {
                "status": "success",
                "baseline_duration": baseline.project_duration,
                "revised_duration": revised.project_duration,
                "days_saved": days_saved,
                "compressed_activity_ids": sorted(reductions.keys()),
                "scenarios": scenarios,
                "revised_results": [r.model_dump() for r in revised.results],
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"Fast-track analysis failed: {exc}"}
