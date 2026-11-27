"""The knowledge-base cache serves the file it names, not a file with the same mtime.

Every knowledge base here is invented in the test; no served entry is read.
"""
from __future__ import annotations

import json
import os

import pytest

from app.blocks import _knowledge as kb

_STAMP_NS = 1_700_000_000_000_000_000  # one fixed mtime for every file below


def _write(path, value, stamp_ns=_STAMP_NS):
    path.write_text(json.dumps({
        "schema_version": "1", "kb_version": "t",
        "entries": [{"id": "x.k", "thresholds": {"v": value}}],
    }), encoding="utf-8")
    os.utime(path, ns=(stamp_ns, stamp_ns))
    return path


def _value():
    rule = kb.get_rule("x.k")
    return None if rule is None else rule["thresholds"]["v"]


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.setattr(kb, "_KB_CACHE", None)
    monkeypatch.setattr(kb, "_KB_KEY", None)
    yield
    monkeypatch.setattr(kb, "_KB_CACHE", None)
    monkeypatch.setattr(kb, "_KB_KEY", None)


@pytest.mark.parametrize("first, second", [(9.0, 0.5), (1.25, 1.2)])
def test_second_file_with_the_same_mtime_is_read(tmp_path, monkeypatch, first, second):
    a = _write(tmp_path / "a.json", first)
    b = _write(tmp_path / "b.json", second)
    assert os.stat(a).st_mtime_ns == os.stat(b).st_mtime_ns
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(a))
    assert _value() == first
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(b))
    assert _value() == second


@pytest.mark.parametrize(
    "first, second",
    [
        (1.5, 22.75),  # the rewrite changes the size; the mtime is put back
        (1.25, 1.2),   # also a size change (one digit fewer)
    ],
)
def test_rewrite_that_keeps_the_mtime_is_read(tmp_path, monkeypatch, first, second):
    path = _write(tmp_path / "kb.json", first)
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    assert _value() == first
    _write(path, second)
    assert _value() == second


@pytest.mark.parametrize(
    "broken_is_same_file, expected",
    [
        (True, 3.0),    # same file turned unreadable: its last good copy is kept
        (False, None),  # another file is unreadable: the first file's entries are not served
    ],
)
def test_failed_read_never_serves_another_files_entries(
        tmp_path, monkeypatch, broken_is_same_file, expected):
    good = _write(tmp_path / "good.json", 3.0)
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(good))
    assert _value() == 3.0
    broken = good if broken_is_same_file else tmp_path / "other.json"
    broken.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(broken))
    assert _value() == expected


@pytest.mark.parametrize("value", [4.0, 0.25])
def test_unchanged_file_is_served_from_the_cache(tmp_path, monkeypatch, value):
    path = _write(tmp_path / "kb.json", value)
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    first = kb._load_kb()
    assert kb._load_kb() is first
