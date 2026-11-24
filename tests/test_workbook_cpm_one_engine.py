"""KS-4: the cost-loaded workbook's CPM runs through the envelope, one copy.

The workbook used to carry its own forward/backward pass (finish-to-start
only, a loop silently scheduled in listed order, a fractional duration
truncated). It now asks the registered ``cpm_network`` kernel through
``run_calculation``; these tests pin that the numbers it writes are the
kernel's, and that a network the kernel refuses is never written.

Every network is built here; expected dates come from a hand pass in this
file, not from the code under test.
"""
from __future__ import annotations

import inspect

import openpyxl
import pytest

from app.lib import pm_excel
from app.lib.pm_computations import compute_cpm
from app.schemas.cpm import CPMInput


def _hand_pass(acts):
    """Finish-to-start forward/backward pass written out for the test."""
    dur = {a["id"]: a["duration"] for a in acts}
    preds = {a["id"]: list(a["predecessors"]) for a in acts}
    es, ef = {}, {}
    pending = [a["id"] for a in acts]
    while pending:
        for i in list(pending):
            if all(p in ef for p in preds[i]):
                es[i] = max((ef[p] for p in preds[i]), default=0)
                ef[i] = es[i] + dur[i]
                pending.remove(i)
    proj = max(ef.values())
    succ = {i: [j for j in preds if i in preds[j]] for i in preds}
    ls, lf = {}, {}
    pending = [a["id"] for a in acts]
    while pending:
        for i in list(pending):
            if all(s in ls for s in succ[i]):
                lf[i] = min((ls[s] for s in succ[i]), default=proj)
                ls[i] = lf[i] - dur[i]
                pending.remove(i)
    return {i: {"es": es[i], "ef": ef[i], "ls": ls[i], "lf": lf[i],
                "tf": ls[i] - es[i], "critical": ls[i] - es[i] <= 0} for i in dur}, proj


NETWORK_A = [
    {"id": "A", "name": "a", "duration": 3, "predecessors": []},
    {"id": "B", "name": "b", "duration": 4, "predecessors": ["A"]},
    {"id": "C", "name": "c", "duration": 2, "predecessors": ["A"]},
    {"id": "D", "name": "d", "duration": 1, "predecessors": ["B", "C"]},
]
NETWORK_B = [
    {"id": "S1", "name": "p", "duration": 5, "predecessors": []},
    {"id": "S2", "name": "q", "duration": 2, "predecessors": []},
    {"id": "S3", "name": "r", "duration": 6, "predecessors": ["S2"]},
    {"id": "S4", "name": "s", "duration": 3, "predecessors": ["S1", "S3"]},
    {"id": "S5", "name": "t", "duration": 1, "predecessors": ["S1"]},
]


@pytest.mark.parametrize("network", [NETWORK_A, NETWORK_B])
def test_workbook_dates_are_the_hand_pass(network):
    cpm, proj, order = pm_excel._cpm(network)
    expected, expected_proj = _hand_pass(network)
    assert proj == expected_proj
    assert cpm == expected
    assert order == [a["id"] for a in network]


@pytest.mark.parametrize("network", [NETWORK_A, NETWORK_B])
def test_workbook_dates_are_the_shared_engine(network):
    cpm, proj, _ = pm_excel._cpm(network)
    out = compute_cpm(CPMInput(activities=[
        {"id": a["id"], "duration": a["duration"],
         "predecessors": [{"predecessor_id": p} for p in a["predecessors"]]}
        for a in network]))
    assert proj == out.project_duration
    for r in out.results:
        assert (cpm[r.id]["es"], cpm[r.id]["lf"], cpm[r.id]["tf"]) == (
            r.early_start_day, r.late_finish_day, r.total_float)


@pytest.mark.parametrize("loop", [
    [{"id": "A", "duration": 2, "predecessors": ["B"]},
     {"id": "B", "duration": 1, "predecessors": ["A"]}],
    [{"id": "X", "duration": 1, "predecessors": []},
     {"id": "Y", "duration": 2, "predecessors": ["Z"]},
     {"id": "Z", "duration": 3, "predecessors": ["Y"]}],
])
def test_a_loop_is_refused_not_scheduled_in_listed_order(loop):
    with pytest.raises(ValueError, match="loop"):
        pm_excel._cpm(loop)
    with pytest.raises(ValueError):
        pm_excel.generate_cost_loaded_schedule({"project": "p"}, loop)


@pytest.mark.parametrize("duration", [2.5, 0.4])
def test_a_fractional_duration_is_refused_not_truncated(duration):
    net = [{"id": "A", "duration": 1, "predecessors": []},
           {"id": "B", "duration": duration, "predecessors": ["A"]}]
    with pytest.raises(ValueError, match="whole number"):
        pm_excel._cpm(net)


@pytest.mark.parametrize("network", [NETWORK_A, NETWORK_B])
def test_the_written_sheet_carries_the_kernel_numbers(network):
    wb = pm_excel.generate_cost_loaded_schedule({"project": "p"}, network)
    ws = wb["L2 Schedule"]
    expected, _ = _hand_pass(network)
    seen = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        if row[0] in expected:
            seen[row[0]] = {"es": row[5], "ef": row[6], "ls": row[7], "lf": row[8],
                            "tf": row[9], "critical": row[10] == "YES"}
    assert seen == expected
    assert isinstance(wb, openpyxl.Workbook)


def test_no_private_pass_remains():
    src = inspect.getsource(pm_excel)
    assert "run_calculation" in inspect.getsource(pm_excel._cpm)
    assert "deque" not in src
    assert "popleft" not in src


# ── the binder keeps a one-item list for an input declared as a list ────────
#
# ``run_calculation`` used to read EVERY one-item list as its item, before
# the type-aware coercion ran, so a one-activity network reached the kernel
# as a bare mapping and was refused as "no activity". A scalar input given
# as a one-item list is still read as its item.

@pytest.mark.parametrize("act", [
    {"id": "1.1", "name": "first", "duration": 7, "predecessors": []},
    {"id": "Q", "name": "", "duration": 12, "predecessors": []},
])
def test_a_one_activity_network_is_still_a_network(act):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("cpm_network", {"activities": [dict(act)]})
    assert env["status"] == "success", env
    assert env["result"]["project_duration"] == act["duration"]
    assert env["result"]["critical_path"] == [act["id"]]
    cpm, proj, _ = pm_excel._cpm([dict(act)])
    assert proj == act["duration"]
    assert cpm[act["id"]] == {"es": 0, "ef": act["duration"], "ls": 0,
                              "lf": act["duration"], "tf": 0, "critical": True}


@pytest.mark.parametrize("es,ef,ls,lf", [(2, 6, 5, 9), (0, 4, 0, 4)])
def test_a_one_item_list_for_a_scalar_input_is_its_item(es, ef, ls, lf):
    from app.lib.construction_formulas import run_calculation

    listed = run_calculation("critical_path_float", {
        "early_start": [es], "early_finish": [ef],
        "late_start": [ls], "late_finish": [lf]})
    assert listed["status"] == "success", listed
    assert listed["result"]["total_float"] == ls - es
    assert listed["result"]["is_critical"] is (ls - es <= 0)


@pytest.mark.parametrize("empty", [[], ()])
def test_an_empty_activity_list_is_an_empty_workbook_not_a_refusal(empty):
    # No activities is not a network the kernel must refuse: the workbook has
    # no rows, no project duration and no listed order. Without the guard the
    # empty network reaches the kernel and the writer raises.
    assert pm_excel._cpm(empty) == ({}, 0, [])
