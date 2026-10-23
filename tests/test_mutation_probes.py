"""The mutation runner must not be able to leave a mutant on disk.

The runner sabotages a guard, runs one test, and restores the file. The
dangerous state is the middle: if the process dies there, the working tree
holds a disabled guard and looks completely normal. That has happened in a
sibling repo — a monitor was left blinded on disk and two commits nearly
shipped it — which is why the marker exists and why it is tested.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.mutation_probes import MARKER, PROBES, apply_probe, restore_probe

ROOT = Path(__file__).resolve().parents[1]


def test_every_probe_targets_text_that_exists_exactly_once():
    """A probe whose target moved must fail loudly, not silently pass."""
    problems = []
    for probe in PROBES:
        src = (ROOT / probe.path).read_text(encoding="utf-8")
        newline = "\r\n" if "\r\n" in src else "\n"
        needle = probe.old.replace("\n", newline)
        n = src.count(needle)
        if n != 1:
            problems.append(f"{probe.name}: {n} match(es) in {probe.path}")
    assert not problems, "; ".join(problems)


def test_every_probe_names_a_test_that_exists():
    missing = []
    for probe in PROBES:
        rel = probe.test.split("::")[0]
        if not (ROOT / rel).is_file():
            missing.append(f"{probe.name} -> {rel}")
    assert not missing, "; ".join(missing)


def test_every_probe_changes_something():
    for probe in PROBES:
        assert probe.old != probe.new, probe.name


def test_apply_then_restore_is_byte_exact(tmp_path):
    f = tmp_path / "m.py"
    original = b"x = 1\nkeep = True\n"
    f.write_bytes(original)

    apply_probe(str(f), "x = 1", "x = 2", marker_dir=tmp_path)
    assert f.read_bytes() == b"x = 2\nkeep = True\n"
    assert (tmp_path / MARKER).exists()

    assert restore_probe(marker_dir=tmp_path) is True
    assert f.read_bytes() == original, "restore must be byte-exact"
    assert not (tmp_path / MARKER).exists()


def test_restore_is_a_no_op_without_a_marker(tmp_path):
    assert restore_probe(marker_dir=tmp_path) is False


def test_a_crashed_run_is_recovered_by_the_next_one(tmp_path):
    """The whole point of the marker.

    Simulates a killed process: the mutation is applied and nothing
    restores it. The next start must put the file back before doing
    anything else.
    """
    f = tmp_path / "guard.py"
    original = b"if user.is_admin:\n    allow()\n"
    f.write_bytes(original)

    apply_probe(str(f), "if user.is_admin:", "if True:", marker_dir=tmp_path)
    assert b"if True:" in f.read_bytes(), "precondition: the guard is sabotaged"
    # <- process dies here

    assert restore_probe(marker_dir=tmp_path) is True
    assert f.read_bytes() == original


def test_a_target_that_moved_refuses_rather_than_guessing(tmp_path):
    f = tmp_path / "m.py"
    f.write_bytes(b"y = 1\n")
    with pytest.raises(SystemExit) as exc:
        apply_probe(str(f), "x = 1", "x = 2", marker_dir=tmp_path)
    assert "not unique" in str(exc.value)
    assert f.read_bytes() == b"y = 1\n", "a refused probe must not write"


def test_a_crlf_file_is_matched_and_restored(tmp_path):
    """CI is Linux, dev boxes are Windows. Both must run the same probes."""
    f = tmp_path / "crlf.py"
    original = b"if guard:\r\n    return True\r\n"
    f.write_bytes(original)

    apply_probe(str(f), "if guard:\n    return True", "if guard:\n    return False",
                marker_dir=tmp_path)
    assert f.read_bytes() == b"if guard:\r\n    return False\r\n"
    restore_probe(marker_dir=tmp_path)
    assert f.read_bytes() == original
