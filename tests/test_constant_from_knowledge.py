"""A calculator constant the knowledge base states is read from that entry.

Every test writes its own invented knowledge base and points the shipped
path at it; no entry, figure or source here is taken from the real one.
"""

from __future__ import annotations

import json

import pytest

from app.blocks import _knowledge
from app.lib.formula_registry import Constant, Constants


def _write_kb(path, entries):
    path.write_text(
        json.dumps({"schema_version": "1", "kb_version": "t", "entries": entries}),
        encoding="utf-8",
    )
    return str(path)


def _entry(eid, thresholds, provenance=None):
    return {"id": eid, "thresholds": thresholds, "provenance": provenance or {}}


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """Point the shipped knowledge base at an invented file."""

    def _use(entries):
        monkeypatch.setattr(_knowledge, "_KB_PATH", _write_kb(tmp_path / "shipped.json", entries))

    return _use


@pytest.mark.parametrize(
    "entries, entry_id, field_path, value, source",
    [
        # the written reference is the course reference when the entry has one
        (
            [_entry("walls.min_lap", {"min_ratio": 1.4},
                    {"course_reference": "invented course notes, p3", "source": "site notes"})],
            "walls.min_lap", "thresholds.min_ratio", 1.4,
            "knowledge entry walls.min_lap (invented course notes, p3)",
        ),
        # no course reference: the entry's own source is named instead; an
        # integer value is read as a number too
        (
            [_entry("pits.max_depth", {"limit": {"metres": 6}}, {"source": "invented handbook"})],
            "pits.max_depth", "thresholds.limit.metres", 6.0,
            "knowledge entry pits.max_depth (invented handbook)",
        ),
        # no provenance at all: the source still names the entry
        (
            [_entry("roofs.fall", {"min": 0.02})],
            "roofs.fall", "thresholds.min", 0.02,
            "knowledge entry roofs.fall",
        ),
    ],
)
def test_value_and_source_come_from_the_entry(shipped, entries, entry_id, field_path, value, source):
    shipped(entries)
    c = Constant.from_knowledge(entry_id, field_path, "ratio", "an invented limit")
    assert c.value == pytest.approx(value)
    assert c.source == source
    assert (c.unit, c.meaning) == ("ratio", "an invented limit")


@pytest.mark.parametrize(
    "entries, entry_id, field_path, says",
    [
        ([_entry("a.one", {"x": 1.5})], "a.two", "thresholds.x", "not in the shipped"),  # no such entry
        ([_entry("a.one", {"x": 1.5})], "a.one", "thresholds.y", "has no field"),  # no such field
        ([_entry("a.one", {"x": 1.5})], "a.one", "thresholds.x.deeper", "has no field"),  # path past a number
        ([_entry("a.one", {"x": ">= 1.5"})], "a.one", "thresholds.x", "is not a number"),  # words
        ([_entry("a.one", {"x": True})], "a.one", "thresholds.x", "is not a number"),  # a flag
        ([_entry("a.one", {"x": None})], "a.one", "thresholds.x", "is not a number"),  # stated as nothing
    ],
)
def test_no_number_in_the_entry_raises_never_falls_back(shipped, entries, entry_id, field_path, says):
    # The error names what is missing: the entry, the field, or a number.
    shipped(entries)
    with pytest.raises(LookupError, match=says):
        Constant.from_knowledge(entry_id, field_path, "ratio", "an invented limit")


@pytest.mark.parametrize(
    "entries, wanted, found",
    [
        ([_entry("a.one", {"x": 1}), _entry("b.two", {"x": 2})], "b.two", {"x": 2}),
        ([_entry("a.one", {"x": 1}), _entry("b.two", {"x": 2})], "a.one", {"x": 1}),
        ([_entry("a.one", {"x": 1})], "c.three", None),
    ],
)
def test_shipped_entry_finds_the_entry_by_id(shipped, entries, wanted, found):
    shipped(entries)
    entry = _knowledge.shipped_entry(wanted)
    assert (entry["thresholds"] if entry is not None else None) == found


@pytest.mark.parametrize("text", ["{ not json", ""])
def test_shipped_entry_unreadable_file_raises(tmp_path, monkeypatch, text):
    bad = tmp_path / "broken.json"
    bad.write_text(text, encoding="utf-8")
    monkeypatch.setattr(_knowledge, "_KB_PATH", str(bad))
    with pytest.raises(LookupError, match="unreadable"):
        _knowledge.shipped_entry("a.one")


def test_unreadable_shipped_file_raises(tmp_path, monkeypatch):
    bad = tmp_path / "broken.json"
    bad.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(_knowledge, "_KB_PATH", str(bad))
    with pytest.raises(LookupError, match="unreadable"):
        Constant.from_knowledge("a.one", "thresholds.x", "ratio", "an invented limit")


@pytest.mark.parametrize("shipped_x, served_x", [(1.5, 9.0), (3.0, 0.5)])
def test_served_override_does_not_change_the_constant(shipped, tmp_path, monkeypatch, shipped_x, served_x):
    # The served file (CONSTRUCTION_KB_FILE) may be swapped at run time; what
    # a calculator computes follows the shipped knowledge base only.
    shipped([_entry("a.one", {"x": shipped_x})])
    served = _write_kb(tmp_path / "served.json", [_entry("a.one", {"x": served_x})])
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", served)
    # The served reader keeps one cached copy across calls. Start from an
    # empty cache so this case never reads the copy an earlier case left
    # (two invented files can share an mtime; the loader's cache identity is
    # its own fix, not this test's subject).
    monkeypatch.setattr(_knowledge, "_KB_CACHE", None)
    assert _knowledge.get_rule("a.one")["thresholds"]["x"] == served_x  # served copy
    assert Constant.from_knowledge("a.one", "thresholds.x", "ratio", "m").value == shipped_x


@pytest.mark.parametrize("stated, expected", [(1.2, 12), (1.5, 15)])
def test_calculator_default_follows_whatever_the_entry_states(shipped, stated, expected):
    # Two knowledge bases that state different factors give two different
    # answers from the same calculator source; nothing in it holds the figure.
    shipped([_entry("x.fos", {"min_fos": stated})])
    k = Constants(required=Constant.from_knowledge("x.fos", "thresholds.min_fos", "ratio", "least factor"))

    def needed(load: float, required: float = k.required) -> int:
        return round(load * required)

    assert needed(10) == expected
    assert needed(10, required=2.0) == 20  # a stated factor still wins
