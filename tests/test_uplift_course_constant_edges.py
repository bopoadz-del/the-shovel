"""Edges of the course-read uplift factor and of the knowledge-base cache key.

Each test pins one decision the earlier tests left open (found by mutating the
new logic one line at a time): a dotted path that runs into a value that is
not a mapping, the exact ``fos == required`` boundary, and a knowledge-base
path spelled two ways for one file. Every knowledge base and figure here is
invented in the test; no served entry is read.
"""
from __future__ import annotations

import json
import os

import pytest

from app.blocks import _knowledge as kb


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.setattr(kb, "_KB_CACHE", None)
    monkeypatch.setattr(kb, "_KB_KEY", None)
    yield
    monkeypatch.setattr(kb, "_KB_CACHE", None)
    monkeypatch.setattr(kb, "_KB_KEY", None)


def _write_kb(path, entries):
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "t", "entries": entries}),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# A dotted path that meets a number or a list half-way is a missing field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "thresholds",
    [
        1.3,              # the parent is the number itself, not a table of fields
        ["min_fos"],      # a list that holds the key's name, but no value under it
        "min_fos",        # a word that contains the key's name
    ],
    ids=["number", "list", "word"],
)
def test_a_path_through_a_non_mapping_is_a_missing_field(tmp_path, monkeypatch, thresholds):
    from app.agents.hats.design.formulas.construction_formulas import _course_constant

    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(_write_kb(
        tmp_path / "kb.json", [{"id": "x.fos", "thresholds": thresholds, "provenance": {}}])))
    with pytest.raises(LookupError, match="has no 'thresholds.min_fos'"):
        _course_constant("x.fos", "thresholds.min_fos", "ratio", "minimum FOS")


@pytest.mark.parametrize("value", [1.3, 2.0])
def test_a_path_through_mappings_reads_the_value(tmp_path, monkeypatch, value):
    from app.agents.hats.design.formulas.construction_formulas import _course_constant

    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(_write_kb(
        tmp_path / "kb.json", [{"id": "x.fos", "thresholds": {"min_fos": value}, "provenance": {}}])))
    assert _course_constant("x.fos", "thresholds.min_fos", "ratio", "minimum FOS").value == value


# ---------------------------------------------------------------------------
# A factor of safety exactly equal to the required one passes ("FOS >= required")
# ---------------------------------------------------------------------------
# Hand working (concrete 2.5 T/m3, water 1.0 T/m3, floors 0.5 m):
#   raft 1.0 m -> 2.5; 2 floors -> 2 x 0.5 x 2.5 = 2.5; counter = 5.0 T/m2
#   water 5 m -> uplift 5.0, FOS = 1.0;  water 4 m -> uplift 4.0, FOS = 1.25


@pytest.mark.parametrize(
    "water_depth, required_fos, can_stop",
    [
        (5.0, 1.0, True),    # exactly at the required factor: dewatering can stop
        (4.0, 1.25, True),   # the same boundary with another pair of figures
        (5.0, 1.01, False),  # just above the computed factor: it cannot
        (4.0, 1.26, False),
    ],
)
def test_a_factor_exactly_at_the_required_value_passes(water_depth, required_fos, can_stop):
    from app.agents.hats.design.formulas.construction_formulas import dewatering_uplift_check

    out = dewatering_uplift_check(water_depth=water_depth, raft_thickness=1.0, floor_count=2,
                                  floor_thickness=0.5, required_fos=required_fos)
    assert out.can_stop is can_stop


# ---------------------------------------------------------------------------
# One file named two ways is one file to the cache
# ---------------------------------------------------------------------------


def _other_spellings(tmp_path):
    (tmp_path / "sub").mkdir()
    return [
        os.path.join(str(tmp_path), "sub", "..", "kb.json"),
        os.path.join(str(tmp_path), ".", "kb.json"),
    ]


@pytest.mark.parametrize("spelling", [0, 1], ids=["parent-step", "dot-step"])
def test_the_same_file_under_another_spelling_is_served_from_the_cache(tmp_path, monkeypatch, spelling):
    path = _write_kb(tmp_path / "kb.json", [{"id": "x.k", "thresholds": {"v": 4.0}}])
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    first = kb._load_kb()
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", _other_spellings(tmp_path)[spelling])
    assert kb._load_kb() is first


@pytest.mark.parametrize("spelling, value", [(0, 3.0), (1, 0.5)], ids=["parent-step", "dot-step"])
def test_the_same_file_turned_unreadable_under_another_spelling_keeps_its_last_copy(
        tmp_path, monkeypatch, spelling, value):
    path = _write_kb(tmp_path / "kb.json", [{"id": "x.k", "thresholds": {"v": value}}])
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    assert kb.get_rule("x.k")["thresholds"]["v"] == value
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", _other_spellings(tmp_path)[spelling])
    assert kb.get_rule("x.k")["thresholds"]["v"] == value
