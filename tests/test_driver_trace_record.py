"""The trace record (trace/1): written by code, stamped verified only by a
tester or a user, exportable only when verified, path right, every figure
sourced and the question not held back. Inputs invented for the test."""
import json

import pytest

from app.agents.driver import trace_record as tr


def _record(figures=None, question_set="teaching", ended="answer", rounds=None):
    return tr.build(
        turn="ws-p-1", question="q", final_answer="a",
        rounds=rounds if rounds is not None else [
            {"n": 1, "calls": [{"tool": "construction_calc", "entry": "formula:f1",
                                "args": {"x": 4471}, "result": {"result": 4471}, "ms": 5}]},
            {"n": 2, "calls": [{"tool": "construction_calc", "entry": "formula:f2",
                                "args": {}, "result": {}, "ms": 3}]}],
        figures=figures if figures is not None else [{"figure": "2", "source": "calculator"}],
        ended=ended, question_set=question_set, seconds=1.23456,
        provider="p", model="m", temperature=0, seed=7)


def test_record_shape_is_plain_json():
    rec = _record()
    assert json.loads(json.dumps(rec)) == rec
    assert rec["schema"] == "trace/1"
    assert rec["rounds_used"] == 2 and rec["seconds"] == 1.235
    assert rec["verified"] is None and rec["path_right"] is None


@pytest.mark.parametrize("figures,expect", [
    ([{"figure": "2", "source": "calculator"}, {"figure": "20", "source": "user"}], True),
    ([{"figure": "7", "source": "project"}], True),
    ([], True),  # no figure: nothing could be invented
    ([{"figure": "2", "source": "calculator"}, {"figure": "880", "source": None}], False),
    ([{"figure": "880", "source": "unresolved"}], False),
    ([{"figure": "880", "source": ""}], False),
    (["880"], False),
])
def test_figures_from_tools(figures, expect):
    assert tr.figures_from_tools(figures) is expect
    assert _record(figures=figures)["figures_from_tools"] is expect


@pytest.mark.parametrize("kw", [{"ended": "finished"}, {"question_set": "test"}, {"question_set": ""}])
def test_unknown_ending_or_set_raises(kw):
    with pytest.raises(ValueError):
        _record(**kw)


@pytest.mark.parametrize("by", ["tester:run-1", "user:42"])
def test_stamp_by_tester_or_user(by):
    rec = tr.stamp_verified(_record(), by, "2026-11-26T12:00:00Z", True)
    assert rec["verified"]["by"] == by and rec["path_right"] is True


@pytest.mark.parametrize("by", ["model:deepseek", "tester:", "user", "", "owner:1", None])
def test_stamp_refuses_other_verifiers(by):
    with pytest.raises(ValueError):
        tr.stamp_verified(_record(), by, "t", True)


def test_stamp_needs_boolean_path_right():
    with pytest.raises(ValueError):
        tr.stamp_verified(_record(), "user:1", "t", "yes")


def test_stamp_does_not_mutate_original():
    rec = _record()
    tr.stamp_verified(rec, "user:1", "t", True)
    assert rec["verified"] is None


def test_exportable_needs_all_four():
    good = tr.stamp_verified(_record(), "tester:r1", "t", True)
    assert tr.exportable(good) is True
    assert tr.exportable(_record()) is False                                    # not verified
    assert tr.exportable(tr.stamp_verified(_record(), "user:1", "t", False)) is False  # wrong path
    unsourced = tr.stamp_verified(_record(figures=[{"figure": "9", "source": None}]), "user:1", "t", True)
    assert tr.exportable(unsourced) is False                                    # invented figure
    held = tr.stamp_verified(_record(question_set="held_back"), "tester:r1", "t", True)
    assert tr.exportable(held) is False                                         # held back
    frozen = tr.stamp_verified(_record(question_set="frozen"), "tester:r1", "t", True)
    assert tr.exportable(frozen) is True
    unset = tr.stamp_verified(_record(question_set="unset"), "tester:r1", "t", True)
    assert unset["question_set"] == "unset" and tr.exportable(unset) is False  # no set named
    assert tr.example_from(unset) is None
    forged = dict(good, verified={"by": "model:x", "at": "t"})
    assert tr.exportable(forged) is False                                       # stamp written by hand


def test_example_only_from_teaching_and_keeps_no_figure():
    rec = tr.stamp_verified(_record(), "tester:r1", "t", True)
    ex = tr.example_from(rec)
    assert ex["entries_in_order"] == ["formula:f1", "formula:f2"]
    dumped = json.dumps(ex)
    assert "4471" not in dumped and '"args"' not in dumped and '"result"' not in dumped
    assert "final_answer" not in ex and "figures" not in ex
    frozen = tr.stamp_verified(_record(question_set="frozen"), "tester:r1", "t", True)
    assert tr.example_from(frozen) is None
    assert tr.example_from(_record()) is None


def test_example_order_follows_rounds():
    rounds = [{"n": 1, "calls": [{"tool": "t", "entry": "formula:b"}]},
              {"n": 2, "calls": [{"tool": "t", "entry": "formula:a"}, {"tool": "lookup"}]}]
    rec = tr.stamp_verified(_record(rounds=rounds), "user:9", "t", True)
    assert tr.example_from(rec)["entries_in_order"] == ["formula:b", "formula:a", "lookup"]


@pytest.mark.parametrize("n", [1, 3])
def test_rounds_used_counts_rounds(n):
    rounds = [{"n": i + 1, "calls": []} for i in range(n)]
    assert _record(rounds=rounds)["rounds_used"] == n
