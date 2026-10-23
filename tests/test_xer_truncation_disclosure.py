"""A capped activity list must SAY it is capped.

Live-fire phase 3: a real 25MB, 11,149-task baseline XER parsed with
status success and an activities list of exactly 5,000 -- the default cap.
The true total sat in activity_count, but detecting truncation required
comparing two fields, and the run was nearly reported as "5,000 activities".
A cap that has to be inferred reads as completeness.

Fixture is synthetic (200 tiny tasks, cap forced to 50) so the fence runs
everywhere without the client file.
"""
from __future__ import annotations

import pytest

from app.blocks.primavera_parser import PrimaveraParserBlock


def _mini_xer(tmp_path, n_tasks: int) -> str:
    lines = [
        "ERMHDR\t7.0\t2024-01-01\tProject\tadmin\tA\tdb\tPM\tUSD",
        "%T\tPROJECT",
        "%F\tproj_id\tproj_short_name",
        "%R\t1\tTEST",
        "%T\tCALENDAR",
        "%F\tclndr_id\tclndr_name",
        "%R\t1\tStandard",
        "%T\tTASK",
        "%F\ttask_id\tproj_id\ttask_code\ttask_name\ttask_type\tstatus_code"
        "\ttarget_start_date\ttarget_end_date\ttarget_drtn_hr_cnt\tclndr_id",
    ]
    for i in range(n_tasks):
        lines.append(
            f"%R\t{1000+i}\t1\tA{i:04}\tTask {i}\tTT_Task\tTK_NotStart"
            f"\t2024-01-01\t2024-01-05\t32\t1"
        )
    lines.append("%E")
    path = tmp_path / "mini.xer"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.mark.asyncio
async def test_a_capped_list_declares_itself(tmp_path):
    res = await PrimaveraParserBlock().process(
        {"file_path": _mini_xer(tmp_path, 200)}, {"max_activities": 50})
    assert res["status"] == "success"
    assert res["activity_count"] == 200, "true total must survive"
    assert res["activities_returned"] == 50
    assert res["activities_truncated"] is True
    assert "truncation_note" in res and "200" in res["truncation_note"]


@pytest.mark.asyncio
async def test_an_uncapped_list_does_not_cry_wolf(tmp_path):
    res = await PrimaveraParserBlock().process(
        {"file_path": _mini_xer(tmp_path, 30)}, {"max_activities": 50})
    assert res["activities_truncated"] is False
    assert "truncation_note" not in res
