"""ConstructionContainer._calculate_cpm schedules a network that states its
logic but not its float (exam 6, M05 engine 3).

Each decision runs on two invented networks; every expected value is a hand
forward/backward pass written beside its row, never copied from a product
answer. No project, person or document is named.
"""
from __future__ import annotations

import pytest


def _cpm(acts, band=None):
    from app.containers.construction import ConstructionContainer
    return ConstructionContainer()._calculate_cpm({"activities": acts}, band)


def _net(spec):
    """spec: id -> (duration, [predecessors])."""
    return [{"id": i, "duration": d, "predecessors": list(p)} for i, (d, p) in spec.items()]


# Hand: A 0-4, B 4-9, C 4-6, D 9-12 | duration 12 | C: LS 7, float 3.
NET_1 = {"A": (4, []), "B": (5, ["A"]), "C": (2, ["A"]), "D": (3, ["B", "C"])}
# Hand: X 0-6, Y 6-8, Z 0-3 (Z free, LF 8) | duration 8 | Z float 5.
NET_2 = {"X": (6, []), "Y": (2, ["X"]), "Z": (3, [])}


@pytest.mark.parametrize("spec, duration, critical, floats", [
    (NET_1, 12, ["A", "B", "D"], {"A": 0, "B": 0, "C": 3, "D": 0}),
    (NET_2, 8, ["X", "Y"], {"X": 0, "Y": 0, "Z": 5}),
])
def test_logic_without_float_is_scheduled(spec, duration, critical, floats):
    out = _cpm(_net(spec))
    assert out["project_duration_days"] == duration
    assert out["critical_path"] == critical
    got = {a["id"]: a["total_float"] for a in _cpm(_net(spec), 99)["near_critical_activities"]}
    assert got == {k: v for k, v in floats.items() if v > 0}


@pytest.mark.parametrize("spec, band, near", [
    (NET_1, 3, ["C"]),   # C float 3 sits inside a 3-day band
    (NET_2, 4, []),      # Z float 5 is outside a 4-day band
])
def test_near_critical_reads_the_derived_float(spec, band, near):
    out = _cpm(_net(spec), band)
    assert [a["id"] for a in out["near_critical_activities"]] == near


@pytest.mark.parametrize("floats, critical", [
    ({"A": 0, "B": 6, "C": 0, "D": 0}, ["A", "C", "D"]),  # the scheduler's float wins over the logic
    ({"A": 1, "B": 0, "C": 2, "D": 0}, ["B", "D"]),
])
def test_a_stated_float_is_kept(floats, critical):
    acts = [{**a, "total_float": floats[a["id"]]} for a in _net(NET_1)]
    out = _cpm(acts)
    assert out["critical_path"] == critical
    assert all("early_start" not in a for a in out["critical_path_activities"])


@pytest.mark.parametrize("acts", [
    [{"id": "A", "duration": 4}, {"id": "B", "duration": 5}],
    [{"id": "P", "duration": 2, "predecessors": []}, {"id": "Q", "duration": 7, "predecessors": []}],
])
def test_no_stated_logic_is_not_scheduled(acts):
    out = _cpm(acts)
    assert out["project_duration_days"] == 0
    assert all("early_start" not in a for a in out["critical_path_activities"])


@pytest.mark.parametrize("acts", [
    [{"id": "A", "duration": 3, "predecessors": ["B"]}, {"id": "B", "duration": 2, "predecessors": ["A"]}],  # cycle
    [{"id": "A", "duration": 3, "predecessors": ["GHOST"]}],                                                # unknown id
    [{"id": "A", "duration": 2.5, "predecessors": []}, {"id": "B", "duration": 1, "predecessors": ["A"]}],  # fractional
    [{"id": "A", "duration": 3, "predecessors": [7]}],                                                       # unreadable
    [{"id": "A", "duration": 3, "predecessors": [{}]}],                                                      # empty dict
])
def test_an_unreadable_network_keeps_the_stated_values(acts):
    out = _cpm(acts)
    assert out["project_duration_days"] == 0
    assert all("early_start" not in a for a in out["critical_path_activities"])


@pytest.mark.parametrize("acts, duration", [
    # A 0-5; B after A with lag 2: 7-10 -> 10
    ([{"id": "A", "duration": 5}, {"id": "B", "duration": 3,
                                   "predecessors": [{"predecessor_id": "A", "lag": 2}]}], 10),
    # P 0-4; Q after P (keyed "id"): 4-10 -> 10 ... and R 1 day after Q: 11
    ([{"id": "P", "duration": 4}, {"id": "Q", "duration": 6, "predecessors": [{"id": "P"}]},
      {"id": "R", "duration": 1, "predecessors": [" Q "]}], 11),
])
def test_predecessor_dicts_and_lags_are_read(acts, duration):
    assert _cpm(acts)["project_duration_days"] == duration


@pytest.mark.parametrize("spec", [NET_1, NET_2])
def test_the_callers_activities_are_not_changed(spec):
    acts = _net(spec)
    before = [dict(a) for a in acts]
    _cpm(acts)
    assert acts == before
