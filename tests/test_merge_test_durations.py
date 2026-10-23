"""The shard durations merge keeps each shard's fresh measurements."""
from scripts.merge_test_durations import merge


def test_fresh_values_win_and_untouched_tests_keep_their_committed_time():
    baseline = {"t::a": 1.0, "t::b": 2.0, "t::c": 3.0}
    shard1 = {"t::a": 1.5, "t::b": 2.0, "t::c": 3.0}          # ran a
    shard2 = {"t::a": 1.0, "t::b": 2.5, "t::c": 3.0, "t::d": 0.4}  # ran b, new d
    out = merge(baseline, [shard1, shard2])
    assert out == {"t::a": 1.5, "t::b": 2.5, "t::c": 3.0, "t::d": 0.4}


def test_no_baseline_takes_everything_measured():
    assert merge({}, [{"x": 1.0}, {"y": 2.0}]) == {"x": 1.0, "y": 2.0}
