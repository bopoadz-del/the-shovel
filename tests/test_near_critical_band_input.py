"""The near-critical float band is a caller input with no default (S-K2 slice 2).

Before this change three sites labelled an activity near-critical when its
total float sat inside a fixed band of days that nobody supplied (two used
``<= 5``, one ``< 5``). Which band counts as "near" is a project choice, so:

* with no band supplied, no activity is labelled near-critical and the output
  says so (``near_critical`` is ``None``, not an empty list);
* with a band supplied, positive total float ``<= band`` is near-critical, and
  a different band gives a different answer from the same network;
* a non-positive or non-numeric band is an error naming the input.

Every test builds its own network.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.blocks.cpm_engine import CPMEngineBlock
from app.blocks.primavera_parser import PrimaveraParserBlock
from app.lib.pm_computations import (
    compute_cpm,
    near_critical_ids,
    parse_near_critical_band,
)
from app.schemas.cpm import Activity, CPMInput, Dependency


def _net() -> list[Activity]:
    """A(4) -> D(1); B(3) -> D; C(1) -> D.  Floats: A 0, B 1, C 3, D 0."""
    return [
        Activity(id="A", duration=4),
        Activity(id="B", duration=3),
        Activity(id="C", duration=1),
        Activity(id="D", duration=1, predecessors=[
            Dependency(predecessor_id="A"),
            Dependency(predecessor_id="B"),
            Dependency(predecessor_id="C"),
        ]),
    ]


# -- kernel -----------------------------------------------------------------

def test_no_band_means_near_critical_is_not_computed():
    out = compute_cpm(CPMInput(activities=_net()))
    floats = {r.id: r.total_float for r in out.results}
    assert floats == {"A": 0, "B": 1, "C": 3, "D": 0}
    assert out.near_critical is None
    assert out.near_critical_band_days is None


def test_the_supplied_band_decides_membership():
    narrow = compute_cpm(CPMInput(activities=_net(), near_critical_band_days=2))
    wide = compute_cpm(CPMInput(activities=_net(), near_critical_band_days=3))
    assert narrow.near_critical == ["B"]
    assert narrow.near_critical_band_days == 2
    # the band is inclusive: float 3 is inside a 3-day band
    assert sorted(wide.near_critical) == ["B", "C"]
    assert wide.near_critical_band_days == 3


def test_a_band_below_every_float_gives_an_empty_list_not_none():
    out = compute_cpm(CPMInput(activities=_net(), near_critical_band_days=0.5))
    assert out.near_critical == []


def test_empty_network_keeps_the_same_contract():
    assert compute_cpm(CPMInput(activities=[])).near_critical is None
    assert compute_cpm(
        CPMInput(activities=[], near_critical_band_days=4)).near_critical == []


@pytest.mark.parametrize("bad", [0, -1])
def test_schema_rejects_a_non_positive_band(bad):
    with pytest.raises(ValidationError):
        CPMInput(activities=_net(), near_critical_band_days=bad)


def test_near_critical_ids_excludes_critical_and_respects_band():
    floats = [("x", 0), ("y", 2), ("z", 6)]
    assert near_critical_ids(floats, None) is None
    assert near_critical_ids(floats, 2) == ["y"]
    assert near_critical_ids(floats, 6) == ["y", "z"]


@pytest.mark.parametrize("given,expected", [
    (None, None), ("", None), ("  ", None), (4, 4.0), ("2.5", 2.5),
])
def test_parse_band_accepts_absent_or_positive(given, expected):
    assert parse_near_critical_band(given) == expected


@pytest.mark.parametrize("bad", [0, -3, "0", "abc", True, [5]])
def test_parse_band_rejects_bad_values_naming_the_input(bad):
    with pytest.raises(ValueError, match="near_critical_band_days"):
        parse_near_critical_band(bad)


# -- cpm_engine block ---------------------------------------------------------

def _dict_net() -> list[dict]:
    return [a.model_dump(mode="json") for a in _net()]


@pytest.mark.asyncio
async def test_cpm_engine_without_band_labels_nothing():
    res = await CPMEngineBlock().process({"activities": _dict_net()})
    assert res["status"] == "success"
    assert res["near_critical"] is None
    assert res["near_critical_band_days"] is None


@pytest.mark.asyncio
async def test_cpm_engine_band_from_data_or_params():
    a = await CPMEngineBlock().process(
        {"activities": _dict_net(), "near_critical_band_days": 2})
    b = await CPMEngineBlock().process(
        {"activities": _dict_net()}, {"near_critical_band_days": 3})
    assert a["near_critical"] == ["B"]
    assert sorted(b["near_critical"]) == ["B", "C"]
    assert b["near_critical_band_days"] == 3


@pytest.mark.asyncio
async def test_cpm_engine_bad_band_is_an_error_naming_the_input():
    res = await CPMEngineBlock().process(
        {"activities": _dict_net()}, {"near_critical_band_days": -2})
    assert res["status"] == "error"
    assert "near_critical_band_days" in res["error"]


# -- primavera_parser block ---------------------------------------------------

def _xer(tmp_path, durations_hr: dict[str, int]) -> str:
    """Unlinked tasks: each one's float is the longest duration minus its own."""
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
    for i, (code, hrs) in enumerate(durations_hr.items()):
        lines.append(
            f"%R\t{1000+i}\t1\t{code}\tTask {code}\tTT_Task\tTK_NotStart"
            f"\t2024-01-01\t2024-01-10\t{hrs}\t1"
        )
    lines.append("%E")
    path = tmp_path / "band.xer"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _float_by_code(res) -> dict:
    return res["cpm"]["per_activity_float"]


@pytest.mark.asyncio
async def test_primavera_parser_without_band_labels_nothing(tmp_path):
    path = _xer(tmp_path, {"L1": 40, "S1": 32, "S2": 16})
    res = await PrimaveraParserBlock().process({"file_path": path}, {})
    assert res["status"] == "success"
    floats = _float_by_code(res)
    assert sorted(v for v in floats.values()) == [0, 1, 3]
    assert res["cpm"]["near_critical_activity_ids"] is None
    assert res["cpm"]["near_critical_band_days"] is None


@pytest.mark.asyncio
async def test_primavera_parser_band_decides_membership(tmp_path):
    path = _xer(tmp_path, {"L1": 40, "S1": 32, "S2": 16})
    narrow = await PrimaveraParserBlock().process(
        {"file_path": path}, {"near_critical_band_days": 2})
    wide = await PrimaveraParserBlock().process(
        {"file_path": path}, {"near_critical_band_days": 3})
    floats = _float_by_code(narrow)
    by_float = {v: k for k, v in floats.items()}
    assert narrow["cpm"]["near_critical_activity_ids"] == [by_float[1]]
    assert sorted(wide["cpm"]["near_critical_activity_ids"]) == sorted(
        [by_float[1], by_float[3]])


@pytest.mark.asyncio
async def test_primavera_parser_bad_band_is_an_error(tmp_path):
    path = _xer(tmp_path, {"L1": 40})
    res = await PrimaveraParserBlock().process(
        {"file_path": path}, {"near_critical_band_days": "soon"})
    assert res["status"] == "error"
    assert "near_critical_band_days" in res["error"]


# -- construction container: P6 total_float already on the activities ---------

def _container():
    from app.containers.construction import ConstructionContainer
    return ConstructionContainer()


def _sched() -> dict:
    return {"activities": [
        {"id": "K", "total_float": 0},
        {"id": "M", "total_float": 2},
        {"id": "N", "total_float": 4},
        {"id": "P", "total_float": 9},
    ]}


def test_container_cpm_without_band_labels_nothing():
    out = _container()._calculate_cpm(_sched())
    assert out["near_critical_count"] is None
    assert out["near_critical_activities"] is None
    assert out["near_critical_band_days"] is None
    assert out["critical_path"] == ["K"]


def test_container_cpm_band_decides_membership():
    c = _container()
    narrow = c._calculate_cpm(_sched(), 2)
    wide = c._calculate_cpm(_sched(), 4)
    assert [a["id"] for a in narrow["near_critical_activities"]] == ["M"]
    assert narrow["near_critical_count"] == 1
    assert [a["id"] for a in wide["near_critical_activities"]] == ["M", "N"]
    assert wide["near_critical_band_days"] == 4


@pytest.mark.asyncio
async def test_parse_primavera_schedule_bad_band_is_an_error(tmp_path):
    path = _xer(tmp_path, {"L1": 40})
    res = await _container().parse_primavera_schedule(
        {"file_path": path}, {"near_critical_band_days": 0})
    assert res["status"] == "error"
    assert "near_critical_band_days" in res["error"]
