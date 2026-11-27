"""The design hat's uplift calculator takes its default factor of safety from
the knowledge entry it declares, not from a literal in its signature.

Expected floor counts are worked by hand from the factor the entry states;
the inputs are invented.
"""

from __future__ import annotations

import math

import pytest

from app.agents.hats.design.formulas import construction_formulas as design
from app.agents.hats.design.formulas.construction_formulas import dewatering_uplift_check
from app.blocks import _knowledge


def _stated_factor():
    declared = design._UPLIFT_K.declared()["required_fos"]
    return float(_knowledge.shipped_entry(declared.entry_id)["thresholds"]["min_fos"])


@pytest.mark.parametrize(
    "head, raft, floors_built, floor_t",
    [
        (23.0, 4.0, 0, 0.6),  # invented
        (12.0, 1.5, 3, 0.25),  # invented
    ],
)
def test_unstated_factor_is_the_entrys(head, raft, floors_built, floor_t):
    fos = _stated_factor()
    r = dewatering_uplift_check(
        water_depth=head, raft_thickness=raft, floor_count=floors_built,
        floor_thickness=floor_t, concrete_unit_weight=2.5, water_unit_weight=1.0,
    )
    # hand: floors = ceil((fos x head - raft x 2.5) / (floor_t x 2.5))
    want = max(0, math.ceil((fos * head - raft * 2.5) / (floor_t * 2.5)))
    assert r.min_floors_for_stop == want
    assert r.can_stop is (r.fos >= fos)
    assert f"(required: {fos})" in r.notes[2]


@pytest.mark.parametrize(
    "stated, floors",
    [
        (2.0, 24),  # hand: head 10, raft 1 m x 2.5 = 2.5, floor 0.3 x 2.5 = 0.75; ceil((20 - 2.5) / 0.75) = ceil(23.33) = 24
        (1.0, 10),  # hand: ceil((10 - 2.5) / 0.75) = 10
    ],
)
def test_a_stated_factor_still_wins(stated, floors):
    r = dewatering_uplift_check(
        water_depth=10.0, raft_thickness=1.0, floor_count=0, floor_thickness=0.3,
        required_fos=stated,
    )
    assert r.min_floors_for_stop == floors


@pytest.mark.parametrize(
    "kwargs",
    [
        {"water_depth": -1.0, "raft_thickness": 1.0, "floor_count": 2},
        {"water_depth": -0.5, "raft_thickness": 2.0, "floor_count": 0},
    ],
)
def test_negative_head_is_refused(kwargs):
    with pytest.raises(ValueError, match="water_depth"):
        dewatering_uplift_check(**kwargs)


def _entry_id():
    return design._UPLIFT_K.declared()["required_fos"].entry_id


@pytest.mark.parametrize("head, raft", [(23.0, 4.0), (12.0, 1.5)])  # invented
def test_the_public_door_records_the_entrys_factor(head, raft):
    # Through the calculator door with no factor given, the run's constants
    # record names the factor applied and the knowledge entry that states it;
    # it is not disclosed as a signature default.
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(
        "dewatering_uplift_check", {"water_depth": head, "raft_thickness": raft, "floor_count": 0}
    )
    assert env["status"] == "success", env
    rows = env["result"]["constants_used"]
    assert [r["name"] for r in rows] == ["required_fos"]
    assert rows[0]["value"] == _stated_factor()
    assert rows[0]["source"].startswith("knowledge entry " + _entry_id())
    assert all(d["parameter"] != "required_fos" for d in env["result"].get("defaults_used", []))


@pytest.mark.parametrize("given", [1.5, 2.0])
def test_a_given_factor_is_not_recorded_as_a_constant(given):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(
        "dewatering_uplift_check",
        {"water_depth": 10.0, "raft_thickness": 1.0, "floor_count": 0, "required_fos": given},
    )
    assert env["status"] == "success", env
    assert "constants_used" not in env["result"]
    assert f"(required: {given})" in env["result"]["notes"][2]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_entry_breaks_the_run_and_says_so(tmp_path, monkeypatch, bad):
    # The shipped entry turned non-finite: the door returns a raised error
    # envelope, never a figure computed on NaN or infinity.
    import json

    from app.lib.construction_formulas import run_calculation

    with open(_knowledge._KB_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for e in data["entries"]:
        if e.get("id") == _entry_id():
            e["thresholds"]["min_fos"] = bad
    kb = tmp_path / "kb.json"
    kb.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(_knowledge, "_KB_PATH", str(kb))
    env = run_calculation(
        "dewatering_uplift_check", {"water_depth": 10.0, "raft_thickness": 1.0, "floor_count": 0}
    )
    assert env["status"] == "error" and env.get("raised") is True, env
    assert "not a finite number" in env["error"]


@pytest.mark.parametrize(
    "head, raft, floors_built, floor_t",
    [
        (10.0, 1.8, 10, 0.3),  # hand: uplift 10; counter 1.8 x 2.5 + 10 x 0.3 x 2.5 = 4.5 + 7.5 = 12.0; FOS 1.2
        (20.0, 4.0, 7, 0.8),  # hand: uplift 20; counter 4 x 2.5 + 7 x 0.8 x 2.5 = 10 + 14 = 24.0; FOS 1.2
    ],
)
def test_fos_exactly_at_the_stated_factor_can_stop(head, raft, floors_built, floor_t):
    # The line itself passes: FOS equal to the required factor is enough.
    assert _stated_factor() == 1.2  # the hand values above assume the entry's 1.2
    r = dewatering_uplift_check(
        water_depth=head, raft_thickness=raft, floor_count=floors_built, floor_thickness=floor_t,
    )
    assert r.fos == 1.2
    assert r.can_stop is True and r.needs_tension_piles is False
    assert r.min_floors_for_stop == floors_built


def _kb_stating(tmp_path, monkeypatch, stated):
    """The shipped knowledge base with the uplift entry's factor set to ``stated``."""
    import json

    with open(_knowledge._KB_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for e in data["entries"]:
        if e.get("id") == _entry_id():
            e["thresholds"]["min_fos"] = stated
    kb = tmp_path / "kb.json"
    kb.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(_knowledge, "_KB_PATH", str(kb))


@pytest.mark.parametrize("stated", [0.9, 0.0, -2.5, 6.0])
def test_an_entry_outside_the_declared_range_is_refused_through_the_door(tmp_path, monkeypatch, stated):
    # The calculator accepts a required factor of 1 to 5 from a caller; the
    # entry's figure passes the same range. Outside it the run is refused,
    # naming the knowledge entry, never "OK - dewatering CAN stop" and
    # never an error that blames an input the user did not give.
    from app.lib.construction_formulas import run_calculation

    _kb_stating(tmp_path, monkeypatch, stated)
    env = run_calculation(
        "dewatering_uplift_check",
        {"water_depth": 10.0, "raft_thickness": 1.0, "floor_count": 10, "floor_thickness": 0.28},
    )
    assert env["status"] == "error" and env.get("raised") is True, env
    assert f"knowledge entry '{_entry_id()}' field 'thresholds.min_fos' states {stated:g}, outside" in env["error"]
    assert "(1 to 5 ratio)" in env["error"]
    assert "must be > 0" not in env["error"]


@pytest.mark.parametrize("stated, can_stop", [(1.0, True), (5.0, False)])
def test_an_entry_at_the_declared_bounds_runs(tmp_path, monkeypatch, stated, can_stop):
    # hand: uplift 10; counter 1.0 x 2.5 + 10 x 0.28 x 2.5 = 2.5 + 7.0 = 9.5; FOS 0.95.
    # 0.95 < 1.0 and < 5.0, so neither can stop; head 5 gives FOS 1.9: >= 1.0, < 5.0.
    from app.lib.construction_formulas import run_calculation

    _kb_stating(tmp_path, monkeypatch, stated)
    env = run_calculation(
        "dewatering_uplift_check",
        {"water_depth": 5.0, "raft_thickness": 1.0, "floor_count": 10, "floor_thickness": 0.28},
    )
    assert env["status"] == "success", env
    assert env["result"]["fos"] == pytest.approx(1.9, abs=1e-3)
    assert env["result"]["can_stop"] is can_stop
    assert env["result"]["constants_used"][0]["value"] == stated


@pytest.mark.parametrize("stated", [1.2, 1.4])
def test_the_description_before_the_run_names_the_entry_and_its_value(tmp_path, monkeypatch, stated):
    # A needs-input reply describes the factor as the entry's figure, in the
    # words, the expected-params row and the signature; never "None".
    from app.lib.construction_formulas import run_calculation

    _kb_stating(tmp_path, monkeypatch, stated)
    env = run_calculation("dewatering_uplift_check", {"water_depth": 10.0, "raft_thickness": 1.8})
    assert env["status"] == "needs_input", env
    words = f"from knowledge entry {_entry_id()} ({stated:g})"
    assert f"required_fos (ratio) (default {words})" in env["error"]
    row = next(r for r in env["expected_params"] if r["name"] == "required_fos")
    assert row["default_from"] == words and "default" not in row
    assert f"required_fos: 'Optional[float]' = <{words}>" in env["signature"]
    assert "None" not in env["error"] and "= None" not in env["signature"]
    # the other optional inputs keep their signature defaults
    assert "floor_thickness (m) (default 0.3)" in env["error"]


def test_the_description_says_why_when_the_entry_cannot_be_applied(tmp_path, monkeypatch):
    from app.lib.construction_formulas import run_calculation

    _kb_stating(tmp_path, monkeypatch, 7.0)
    env = run_calculation("dewatering_uplift_check", {"water_depth": 10.0, "raft_thickness": 1.8})
    row = next(r for r in env["expected_params"] if r["name"] == "required_fos")
    assert row["default_from"].startswith(f"from knowledge entry {_entry_id()}, which cannot be applied: ")
    assert "states 7, outside" in row["default_from"]


@pytest.mark.parametrize("words", ["from knowledge entry a.one (1.2)", "from knowledge entry b.two (3)"])
def test_a_stated_default_reads_as_its_words(words):
    from app.lib.construction_formulas import _StatedDefault

    assert repr(_StatedDefault(words)) == f"<{words}>"
    # the two methods by name, so the mutation gate selects this test for them
    d = _StatedDefault.__new__(_StatedDefault)
    d.__init__(words)
    assert d.__repr__() == f"<{words}>"


@pytest.mark.parametrize("stated", [1.1, 2.5])
def test_signature_text_shows_the_entry_in_place_of_the_placeholder(tmp_path, monkeypatch, stated):
    from app.lib.construction_formulas import _signature_text

    _kb_stating(tmp_path, monkeypatch, stated)
    text = _signature_text("dewatering_uplift_check", dewatering_uplift_check)
    assert text.startswith("dewatering_uplift_check(water_depth")
    assert f"required_fos: 'Optional[float]' = <from knowledge entry {_entry_id()} ({stated:g})>" in text
    assert "= None" not in text and "floor_thickness: 'float' = 0.3" in text


@pytest.mark.parametrize("fn_name", ["concrete_curing_time", "concrete_cylinders"])
def test_signature_text_leaves_a_calculator_with_no_knowledge_default_as_is(fn_name):
    import inspect

    from app.lib import formula_registry
    from app.lib.construction_formulas import _signature_text

    spec = next(s for s in formula_registry.all_specs() if s.name == fn_name)
    assert _signature_text(fn_name, spec.fn) == f"{fn_name}{inspect.signature(spec.fn)}"
