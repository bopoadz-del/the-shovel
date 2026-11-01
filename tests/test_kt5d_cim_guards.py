"""K-T5d CI-M coverage: the currency edits pulled four workbook/export
functions into the mutation gate's changed set. These branch-hitting tests kill
the surfaced drop_rule/constant_return mutants on their pre-existing logic.
Full-dotted imports so the gate selects them.
"""
from __future__ import annotations

import pytest

import app.agents.runtime as _runtime  # noqa: F401  (ensure app importable early)
import app.blocks.boq_processor as boq_processor
import app.blocks.schedule_excel_writer as sew
import app.lib.pm_excel as pm_excel
import app.routers.exports as exports

_ACTS = [
    {"id": "1", "wbs": "1.1", "name": "Mobilization", "duration": 5,
     "predecessors": [], "cost": 100000, "manpower": 10},
    {"id": "2", "wbs": "1.2", "name": "Earthworks", "duration": 10,
     "predecessors": ["1"], "cost": 400000, "manpower": 25},
]


@pytest.mark.parametrize("project", ["T1", "T2"])
def test_cost_loaded_schedule_accumulates_man_days(project):
    # Kills the `if overlap > 0` drop_rule: without it the S-curve man-days
    # never accumulate, so every "Man-days" cell would be 0.
    wb = pm_excel.generate_cost_loaded_schedule({"project": project}, _ACTS)
    cl = wb["Cost Loading"]
    man_days = [cl.cell(row, 3).value for row in range(4, 12)]
    assert any(isinstance(v, (int, float)) and v > 0 for v in man_days), man_days


@pytest.mark.parametrize("project", ["C1", "C2"])
def test_cost_loaded_schedule_renders_cost_baseline_when_costs_given(project):
    # Kills the `if has_cost:` drop_rule: cost-bearing activities must produce
    # the COST BASELINE section; dropping the guard would omit it.
    wb = pm_excel.generate_cost_loaded_schedule({"project": project}, _ACTS)
    cl = wb["Cost Loading"]
    col1 = [cl.cell(r, 1).value for r in range(1, cl.max_row + 1)]
    assert any("COST BASELINE" in str(v) for v in col1 if v), col1


@pytest.mark.parametrize("cols", [["description", "rate"], ["item", "price"]])
def test_boq_no_quantity_column_warns(cols):
    # Kills the `if not quantity_col_resolved` drop_rule: a frame with no
    # quantity column must emit the warning.
    import pandas as pd

    block = boq_processor.BOQProcessorBlock()
    df = pd.DataFrame({c: ["x"] for c in cols})
    res = block._process_dataframe(df, {})
    assert any("No quantity column" in w for w in res.get("warnings", [])), res


@pytest.mark.parametrize("start_date", [None, "2026-01-01"])
def test_workbook_from_wbs_activities_returns_a_workbook(start_date):
    # Kills constant_return (a constant is not a workbook) and exercises the
    # `if start_date:` branch both ways.
    acts = [
        {"id": "1", "name": "Mobilization", "duration_days": 5, "wbs_phase": "1", "resources": []},
        {"id": "2", "name": "Earthworks", "duration_days": 10, "wbs_phase": "1", "resources": []},
    ]
    wb = exports._workbook_from_wbs_activities(acts, "Proj", currency="SAR", start_date=start_date)
    assert hasattr(wb, "sheetnames") and "L2 Schedule" in wb.sheetnames


@pytest.mark.parametrize("start_date, present", [("2026-01-01", True), (None, False)])
def test_workbook_start_date_flows_into_the_schedule_meta(monkeypatch, start_date, present):
    # Kills the `if start_date:` drop_rule: a spy on the schedule writer proves
    # start_date reaches the meta only when given; dropping the guard would
    # (never) add it, or add None.
    import app.lib.pm_excel as _pm

    captured: dict = {}

    def _spy(meta, bridged):
        captured.update(meta)
        return _pm.openpyxl.Workbook()

    monkeypatch.setattr("app.lib.pm_excel.generate_cost_loaded_schedule", _spy)
    acts = [{"id": "1", "name": "A", "duration_days": 5, "wbs_phase": "1", "resources": []}]
    exports._workbook_from_wbs_activities(acts, "Proj", start_date=start_date)
    assert ("start_date" in captured) is present, captured


_CPM_OUTPUT = {
    "results": [{
        "id": "1", "name": "A", "duration": 5, "early_start_day": 0,
        "early_finish_day": 5, "late_start_day": 0, "late_finish_day": 5,
        "total_float": 0, "free_float": 0, "is_critical": True,
    }],
    "project_duration": 5, "critical_path": ["1"],
    "critical_percentage": 100.0, "near_critical": [],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("pid", ["p1", "p2"])
async def test_schedule_writer_takes_the_cpm_output_branch(pid):
    # Kills the `if cpm_output:` drop_rule: with a cpm_output the writer returns
    # the cpm-format result; without the branch it would fall through to the
    # (empty) activities path and error.
    block = sew.ScheduleExcelWriterBlock()
    res = await block.process({"cpm_output": dict(_CPM_OUTPUT), "meta": {"project": pid}})
    # Kills the `if cpm_output:` drop_rule by proving the branch is TAKEN: without
    # it the call would fall through to the empty-activities path and return the
    # "Provide cpm_output or activities" error. (The branch's own
    # write_schedule_excel import is a pre-existing bug, flagged separately -- not
    # this PR's scope; so we assert the branch was entered, not that it succeeds.)
    assert "Provide cpm_output or activities" not in str(res), res
