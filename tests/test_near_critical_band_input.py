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

@pytest.mark.parametrize("band,expected", [
    (None, None),          # no band: not computed, not an empty list
    (0.5, []),             # band below every positive float
    (2, ["B"]),
    (3, ["B", "C"]),       # inclusive: float 3 sits inside a 3-day band
])
def test_the_supplied_band_decides_membership(band, expected):
    out = compute_cpm(CPMInput(activities=_net(), near_critical_band_days=band))
    # the band never changes the floats themselves
    assert {r.id: r.total_float for r in out.results} == {
        "A": 0, "B": 1, "C": 3, "D": 0}
    got = None if out.near_critical is None else sorted(out.near_critical)
    assert got == expected
    assert out.near_critical_band_days == band


@pytest.mark.parametrize("band,expected", [(None, None), (4, [])])
def test_empty_network_keeps_the_same_contract(band, expected):
    out = compute_cpm(CPMInput(activities=[], near_critical_band_days=band))
    assert out.near_critical == expected


@pytest.mark.parametrize("bad", [0, -1])
def test_schema_rejects_a_non_positive_band(bad):
    with pytest.raises(ValidationError):
        CPMInput(activities=_net(), near_critical_band_days=bad)


@pytest.mark.parametrize("band,expected", [
    (None, None), (2, ["y"]), (6, ["y", "z"]),
])
def test_near_critical_ids_excludes_critical_and_respects_band(band, expected):
    assert near_critical_ids([("x", 0), ("y", 2), ("z", 6)], band) == expected


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
@pytest.mark.parametrize("where,band,expected", [
    (None, None, None),
    ("data", 2, ["B"]),
    ("params", 3, ["B", "C"]),
])
async def test_cpm_engine_band_from_data_or_params(where, band, expected):
    data = {"activities": _dict_net()}
    params = {}
    if where == "data":
        data["near_critical_band_days"] = band
    elif where == "params":
        params["near_critical_band_days"] = band
    res = await CPMEngineBlock().process(data, params)
    assert res["status"] == "success"
    got = None if res["near_critical"] is None else sorted(res["near_critical"])
    assert got == expected
    assert res["near_critical_band_days"] == band


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-2, "x"])
async def test_cpm_engine_bad_band_is_an_error_naming_the_input(bad):
    res = await CPMEngineBlock().process(
        {"activities": _dict_net()}, {"near_critical_band_days": bad})
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


_DURS = {"L1": 40, "S1": 32, "S2": 16}   # floats 0, 1, 3 (8 h days)


@pytest.mark.asyncio
@pytest.mark.parametrize("band,member_floats", [
    (None, None), (2, [1]), (3, [1, 3]),
])
async def test_primavera_parser_band_decides_membership(
        tmp_path, band, member_floats):
    params = {} if band is None else {"near_critical_band_days": band}
    res = await PrimaveraParserBlock().process(
        {"file_path": _xer(tmp_path, _DURS)}, params)
    assert res["status"] == "success"
    floats = res["cpm"]["per_activity_float"]
    assert sorted(floats.values()) == [0, 1, 3]
    ids = res["cpm"]["near_critical_activity_ids"]
    got = None if ids is None else sorted(floats[i] for i in ids)
    assert got == member_floats
    assert res["cpm"]["near_critical_band_days"] == band


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["soon", 0])
async def test_primavera_parser_bad_band_is_an_error(tmp_path, bad):
    res = await PrimaveraParserBlock().process(
        {"file_path": _xer(tmp_path, {"L1": 40})},
        {"near_critical_band_days": bad})
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


@pytest.mark.parametrize("band,expected", [
    (None, None), (2, ["M"]), (4, ["M", "N"]),
])
def test_container_cpm_band_decides_membership(band, expected):
    out = _container()._calculate_cpm(_sched(), band)
    acts = out["near_critical_activities"]
    assert (None if acts is None else [a["id"] for a in acts]) == expected
    assert out["near_critical_count"] == (
        None if expected is None else len(expected))
    assert out["near_critical_band_days"] == band
    assert out["critical_path"] == ["K"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, "later"])
async def test_parse_primavera_schedule_bad_band_is_an_error(tmp_path, bad):
    from app.containers.construction.schedule import ConstructionScheduleMixin
    res = await ConstructionScheduleMixin.parse_primavera_schedule(
        _container(), {"file_path": _xer(tmp_path, {"L1": 40})},
        {"near_critical_band_days": bad})
    assert res["status"] == "error"
    assert "near_critical_band_days" in res["error"]


# -- guards in the functions this slice touched -------------------------------

@pytest.mark.parametrize("band,near", [(None, None), (3, [])])
def test_container_cpm_empty_schedule_keeps_the_band_contract(band, near):
    out = _container()._calculate_cpm({"activities": []}, band)
    assert out == {
        "critical_path": [], "project_duration_days": 0, "average_float": 0,
        "near_critical_count": None if near is None else 0,
        "near_critical_activities": near,
        "near_critical_band_days": band,
    }


@pytest.mark.parametrize("ids", [["A", "A"], ["X", "Y", "X"]])
def test_compute_cpm_refuses_duplicate_activity_ids(ids):
    acts = [Activity(id=i, duration=1) for i in ids]
    with pytest.raises(ValueError, match="Duplicate activity ids"):
        compute_cpm(CPMInput(activities=acts, near_critical_band_days=2))


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{}, {"file_path": ""}])
async def test_primavera_parser_without_a_file_names_what_is_missing(data):
    res = await PrimaveraParserBlock().process(
        data, {"near_critical_band_days": 2})
    assert res["status"] == "error"
    assert "No file_path provided" in res["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{}, {"file_path": ""}])
async def test_parse_primavera_schedule_without_a_file_names_what_is_missing(data):
    from app.containers.construction.schedule import ConstructionScheduleMixin
    res = await ConstructionScheduleMixin.parse_primavera_schedule(
        _container(), data, {"near_critical_band_days": 2})
    assert res == {"status": "error", "error": "No schedule file provided"}
