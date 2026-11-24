"""S-K2 item 2: CPM total float and free float, verified by slipping.

The audit of 13 Nov (S-K2, second entry) asks for "CPM forward and backward
pass, total AND free float". ``cpm_network`` already prints both; this file
checks them against what the floats mean rather than against figures written
down by hand:

* total float of an activity is the largest number of working days its start
  can slip without moving the project finish;
* free float is the largest slip that moves no successor's early start
  (an activity with no successor takes its total float, as the formula's
  docstring states, because its late finish is the project finish).

The test schedules every network twice over with its own forward pass, which
reads the four relation types from their definitions (FS: start_k >= finish_j
+ lag; SS: start_k >= start_j + lag; FF: finish_k >= finish_j + lag; SF:
finish_k >= start_j + lag; no activity before day 0). It shares no code with
``app.lib.pm_computations``. The networks are generated from seeds, mixing all
four relation types and lags, so no answer is written into the test.
"""
from __future__ import annotations

import random

import pytest

from app.agents.hats.planning.formulas.construction_formulas_planning import cpm_network

_TYPES = ("FS", "SS", "FF", "SF")


def _network(seed: int, size: int) -> list[dict]:
    rng = random.Random(seed)
    acts = []
    for i in range(size):
        preds = []
        for j in rng.sample(range(i), k=min(i, rng.randint(0, 3))):
            preds.append({"id": f"A{j}", "type": rng.choice(_TYPES), "lag": rng.randint(0, 3)})
        acts.append({"id": f"A{i}", "duration": rng.randint(0, 6), "predecessors": preds})
    return acts


def _early_starts(acts: list[dict], slip: dict[str, int]) -> dict[str, int]:
    """Earliest starts, ``slip`` days later than earliest for the named ones.

    ``acts`` is listed so every predecessor comes before its successor."""
    dur = {a["id"]: a["duration"] for a in acts}
    start: dict[str, int] = {}
    for a in acts:
        k, d_k, s = a["id"], a["duration"], 0
        for p in a["predecessors"]:
            j, lag = p["id"], p["lag"]
            s_j, f_j = start[j], start[j] + dur[j]
            s = max(s, {"FS": f_j + lag, "SS": s_j + lag,
                        "FF": f_j + lag - d_k, "SF": s_j + lag - d_k}[p["type"]])
        start[k] = s + slip.get(k, 0)
    return start


def _finish(acts: list[dict], start: dict[str, int]) -> int:
    return max(start[a["id"]] + a["duration"] for a in acts)


def _largest_slip(acts, aid, holds, limit) -> int:
    slip = 0
    while slip < limit and holds(_early_starts(acts, {aid: slip + 1})):
        slip += 1
    return slip


_SEEDS = [(11, 8), (29, 12), (47, 15), (83, 10), (1, 10), (18, 10)]


@pytest.mark.parametrize("seed,size", _SEEDS)
def test_floats_are_the_slips_the_network_absorbs(seed, size):
    acts = _network(seed, size)
    out = cpm_network(acts)
    assert "error" not in out, out
    rows = {r["id"]: r for r in out["activities"]}

    early = _early_starts(acts, {})
    duration = _finish(acts, early)
    assert out["project_duration"] == duration
    limit = duration + 10 * size
    successors = {a["id"]: [] for a in acts}
    for a in acts:
        for p in a["predecessors"]:
            successors[p["id"]].append(a["id"])

    for a in acts:
        aid = a["id"]
        row = rows[aid]
        assert row["early_start"] == early[aid], aid
        tf = _largest_slip(acts, aid, lambda st: _finish(acts, st) == duration, limit)
        assert row["total_float"] == tf, (aid, "total float")
        assert row["is_critical"] == (tf == 0), aid
        if successors[aid]:
            ff = _largest_slip(
                acts, aid,
                lambda st: all(st[s] == early[s] for s in successors[aid]), limit)
        else:
            ff = tf
        assert row["free_float"] == ff, (aid, "free float")
        assert 0 <= row["free_float"] <= row["total_float"], aid


@pytest.mark.parametrize("kind", _TYPES)
def test_every_relation_type_is_exercised(kind):
    # The seeded networks above must reach every relation type, or the slip
    # check would not verify the pass for the types it never meets.
    assert any(p["type"] == kind for seed, size in _SEEDS
               for a in _network(seed, size) for p in a["predecessors"])


def _types_binding_positive_free_float(seed: int, size: int) -> set[str]:
    """Relation types that set a positive free float in one seeded network.

    A link from j to k sets j's free float when slipping j one day past it
    moves k. With j's free float above zero and both j and k lasting at least
    a day, j's start and finish differ and so do k's, so a pass that reads
    the wrong end of either activity gives a different free float."""
    acts = _network(seed, size)
    dur = {a["id"]: a["duration"] for a in acts}
    early = _early_starts(acts, {})
    succ: dict[str, list[tuple[str, str]]] = {a["id"]: [] for a in acts}
    for a in acts:
        for p in a["predecessors"]:
            succ[p["id"]].append((a["id"], p["type"]))
    limit = _finish(acts, early) + 10 * size
    found: set[str] = set()
    for a in acts:
        j = a["id"]
        if not succ[j] or dur[j] == 0:
            continue
        ff = _largest_slip(
            acts, j, lambda st: all(st[s] == early[s] for s, _ in succ[j]), limit)
        if ff == 0 or ff >= limit:
            continue
        moved = _early_starts(acts, {j: ff + 1})
        found |= {t for s, t in succ[j] if moved[s] != early[s] and dur[s] > 0}
    return found


@pytest.mark.parametrize("kind", _TYPES)
def test_every_relation_type_sets_a_positive_free_float(kind):
    # Appearing in a network is not enough: a link that never sets a positive
    # free float cannot show which end of an activity the pass reads.
    assert any(kind in _types_binding_positive_free_float(seed, size)
               for seed, size in _SEEDS)
