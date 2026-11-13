"""cpm_network: the registry's CPM entry over a whole activity network.

Every test builds its own network. The networks differ in shape so a
hard-coded answer cannot pass more than one: in the merge network every
non-critical activity's free float equals its total float; in the shared-slack
network one activity has total float 1 and free float 0, because its
successor shares the slack; the lag network starts a successor before its
predecessor finishes.
"""
from __future__ import annotations

import pytest

from app.agents.hats.planning.formulas.construction_formulas_planning import cpm_network
from app.lib import formula_registry
from app.lib.construction_formulas import CALCULATORS


def _run(acts):
    return cpm_network(acts)


@pytest.mark.parametrize("name, owner", [
    ("cpm_network", "planning"),
    ("critical_path_float", "planning"),
])
def test_float_formulas_registered_on_the_planning_hat_with_no_schema_gap(name, owner):
    specs = {s.name: s for s in formula_registry.all_specs()}
    assert specs[name].owner == owner
    assert name in CALCULATORS
    assert not [g for g in formula_registry.schema_gaps() if f":{name}." in g or f":{name}:" in g]


# (network, project duration, critical path, {id: (ES, EF, LS, LF, TF, FF)})
_NETWORKS = [
    pytest.param(
        [{"id": "A", "duration": 3},
         {"id": "B", "duration": 4, "predecessors": ["A"]},
         {"id": "C", "duration": 2, "predecessors": "A"},
         {"id": "D", "duration": 1, "predecessors": "B, C"}],
        8, ["A", "B", "D"],
        {"A": (0, 3, 0, 3, 0, 0), "C": (3, 5, 5, 7, 2, 2), "D": (7, 8, 7, 8, 0, 0)},
        id="merge"),
    pytest.param(
        [{"id": "A", "duration": 2},
         {"id": "B", "duration": 3, "predecessors": ["A"]},
         {"id": "C", "duration": 1, "predecessors": ["A"]},
         {"id": "E", "duration": 1, "predecessors": ["C"]},
         {"id": "D", "duration": 4, "predecessors": ["B", "E"]}],
        9, ["A", "B", "D"],
        # C can slip 1 day without moving the finish, but any slip moves E.
        {"C": (2, 3, 3, 4, 1, 0), "E": (3, 4, 4, 5, 1, 1)},
        id="shared-slack"),
    pytest.param(
        [{"id": "P", "duration": 4},
         {"id": "Q", "duration": 3, "predecessors": [{"id": "P", "type": "SS", "lag": 2}]}],
        5, ["P", "Q"],
        {"Q": (2, 5, 2, 5, 0, 0)},
        id="start-to-start-lag"),
    pytest.param(
        '[{"id": "X", "duration": 5}, {"id": "Y", "duration": 2, "predecessors": ["X"]}]',
        7, ["X", "Y"],
        {"Y": (5, 7, 5, 7, 0, 0)},
        id="json-text"),
]


@pytest.mark.parametrize("network, duration, critical, expected", _NETWORKS)
def test_network_dates_floats_and_critical_path(network, duration, critical, expected):
    r = _run(network)
    rows = {row["id"]: row for row in r["activities"]}
    assert r["project_duration"] == duration
    assert r["critical_path"] == critical
    for aid, (es, ef, ls, lf, tf, ff) in expected.items():
        row = rows[aid]
        got = (row["early_start"], row["early_finish"], row["late_start"],
               row["late_finish"], row["total_float"], row["free_float"])
        assert got == (es, ef, ls, lf, tf, ff), aid
        line = next(w for w in r["working"] if w.startswith(f"{aid}: "))
        assert f"TF = LS - ES = {ls} - {es} = {tf}; FF = {ff}" in line


@pytest.mark.parametrize("network, words", [
    ([{"id": "A", "duration": 1, "predecessors": ["B"]},
      {"id": "B", "duration": 1, "predecessors": ["A"]}], ["loop"]),
    ([{"id": "A", "duration": 1, "predecessors": ["Z"]}], ["Z"]),
    ([{"id": "A", "duration": 2.5}], ["A", "whole"]),
    ([{"id": "K"}], ["K", "duration"]),
    ([], ["at least one activity"]),
    ("not json", ["list"]),
])
def test_unschedulable_network_is_refused_by_name_not_raised(network, words):
    r = _run(network)
    assert set(r) == {"error"}
    for word in words:
        assert word in r["error"]
