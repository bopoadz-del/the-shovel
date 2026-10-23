"""W2 schedule — planted-network CPM truth + metamorphic float laws.

New test shape (2026-07-26): a hand-built activity network with known
critical path, then perturbations whose effect on project duration is
provable: consuming float changes nothing; stretching the critical path
moves the end date by exactly the stretch.
"""
from app.lib.pm_excel import _cpm, generate_cost_loaded_schedule

# A(5) -> B(3) -> C(2) is the critical chain (duration 10).
# D(1) hangs off A and has 4 days of float (EF 6 vs project end 10).
NETWORK = [
    {"id": "A", "name": "Mobilise", "duration": 5, "predecessors": []},
    {"id": "B", "name": "Excavate", "duration": 3, "predecessors": ["A"]},
    {"id": "C", "name": "Blind",    "duration": 2, "predecessors": ["B"]},
    {"id": "D", "name": "Survey",   "duration": 1, "predecessors": ["A"]},
]


def test_planted_critical_path():
    cpm, proj, order = _cpm(NETWORK)
    assert proj == 10
    assert [cpm[i]["critical"] for i in ("A", "B", "C")] == [True, True, True]
    assert cpm["D"]["critical"] is False
    assert cpm["D"]["tf"] == 4  # 10 - (5+1)
    # ES/EF chain arithmetic on the critical path.
    assert (cpm["A"]["es"], cpm["A"]["ef"]) == (0, 5)
    assert (cpm["B"]["es"], cpm["B"]["ef"]) == (5, 8)
    assert (cpm["C"]["es"], cpm["C"]["ef"]) == (8, 10)


def test_consuming_float_does_not_move_the_end_date():
    # Stretch D by exactly its float: duration must stay 10, D turns critical.
    net = [dict(a) for a in NETWORK]
    net[3]["duration"] = 5  # 1 + 4 float
    cpm, proj, _ = _cpm(net)
    assert proj == 10
    assert cpm["D"]["critical"] is True
    assert cpm["D"]["tf"] == 0


def test_stretching_the_critical_path_moves_the_end_exactly():
    net = [dict(a) for a in NETWORK]
    net[2]["duration"] = 2 + 3  # C stretched by 3
    _, proj, _ = _cpm(net)
    assert proj == 13


def test_exceeding_float_moves_the_end_by_the_excess_only():
    net = [dict(a) for a in NETWORK]
    net[3]["duration"] = 1 + 4 + 2  # float fully consumed + 2 days over
    cpm, proj, _ = _cpm(net)
    assert proj == 12
    # The old critical chain now floats by the same 2 days.
    assert cpm["C"]["tf"] == 2


def test_parallel_branch_addition_is_inert_below_critical_length():
    # Adding an independent 9-day branch (< 10) must not move the end date.
    net = [dict(a) for a in NETWORK] + [
        {"id": "E", "name": "Procure", "duration": 9, "predecessors": []},
    ]
    cpm, proj, _ = _cpm(net)
    assert proj == 10
    assert cpm["E"]["critical"] is False


def test_cost_loaded_schedule_never_fabricates_zero_costs():
    # No activity carries a cost -> the workbook must leave cost cells EMPTY
    # (None), not render a fabricated 0 column.
    wb = generate_cost_loaded_schedule({"project": "Planted"}, NETWORK)
    s = wb["L2 Schedule"]
    cost_cells = [s.cell(r, 12).value for r in range(4, 4 + len(NETWORK))]
    assert set(cost_cells) == {None}, cost_cells


def test_cost_loaded_schedule_orders_by_topology_and_carries_cpm():
    acts = [dict(a, cost=1000 * (i + 1)) for i, a in enumerate(NETWORK)]
    wb = generate_cost_loaded_schedule({"project": "Planted"}, acts)
    s = wb["L2 Schedule"]
    ids = [s.cell(r, 1).value for r in range(4, 4 + len(acts))]
    assert ids[0] == "A"                      # topological start
    assert set(ids) == {"A", "B", "C", "D"}
    # The critical column mirrors the CPM truth.
    crit = {s.cell(r, 1).value: s.cell(r, 11).value for r in range(4, 4 + len(acts))}
    assert crit["A"] == "YES" and crit["D"] in ("", None)
