"""Exam 7 (course knowledge): the course governs.

Two checks, both read the knowledge base itself; no course value is copied
into product code.

1. Coverage. Every knowledge entry whose provenance names the owner-supplied
   course material is exercised by at least one course test file
   (``tests/test_kb_course_*.py``). An entry encoded from the course with no
   test against the course's worked example or table fails here.

2. The course governs. Where a product calculator computes the same item as
   a course entry, the calculator's built-in value must equal the course
   value read from that entry. The pairing below is test data: it names the
   course entry, the field holding the course value, and the calculator
   parameter that carries the same value.

Expected values come from the course entries and from hand calculations in
the comments, never from the calculator's own output.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

from app.blocks import _knowledge as kb
from app.lib.construction_formulas import dewatering_uplift_check

COURSE_PREFIX = "owner-supplied course material"
TESTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers under test (each gets a passing and a failing input below)
# ---------------------------------------------------------------------------


def course_entries(entries):
    """Entries whose provenance names the owner-supplied course material."""
    return [
        e
        for e in entries
        if str((e.get("provenance") or {}).get("course_reference") or "")
        .lower()
        .startswith(COURSE_PREFIX)
    ]


def untested_ids(entries, test_texts):
    """Course entry ids that no course test text mentions."""
    return sorted(
        e["id"] for e in course_entries(entries) if not any(e["id"] in t for t in test_texts)
    )


def course_value(entry, field_path):
    """Read a value from an entry by a dotted path (e.g. thresholds.min_fos)."""
    node = entry
    for key in field_path.split("."):
        node = node[key]
    return node


def calculator_default(func, param):
    """The value a calculator uses when the caller does not pass ``param``."""
    default = inspect.signature(inspect.unwrap(func)).parameters[param].default
    assert default is not inspect.Parameter.empty, f"{param} has no built-in value"
    return default


def disagreements(pairs):
    """(entry id, course value, calculator value) for each pair that differs."""
    out = []
    for entry, field_path, func, param in pairs:
        cv = course_value(entry, field_path)
        pv = calculator_default(func, param)
        if not math.isclose(float(cv), float(pv), rel_tol=1e-9):
            out.append((entry["id"], cv, pv))
    return out


def _course_test_texts():
    # This file names entries only to compare values; it does not count as
    # a worked-example test for them.
    return [
        p.read_text(encoding="utf-8")
        for p in sorted(TESTS_DIR.glob("test_kb_course_*.py"))
        if p.name != Path(__file__).name
    ]


# ---------------------------------------------------------------------------
# 1. coverage: every course entry has a course test
# ---------------------------------------------------------------------------


def _domain(entry):
    return entry["id"].split(".", 1)[0]


def _course_domains():
    return sorted({_domain(e) for e in course_entries(kb.load_knowledge())})


def test_course_entries_exist():
    # The knowledge base carries course-sourced entries in more than one
    # domain; an empty or single-domain selection would make the per-domain
    # coverage check below pass vacuously or run on one input only.
    entries = course_entries(kb.load_knowledge())
    assert len(entries) >= 30
    assert len(_course_domains()) >= 2


@pytest.mark.parametrize("domain", _course_domains())
def test_every_course_entry_is_tested_against_the_course(domain):
    # One case per course domain, so a failure names the domain whose
    # course entries lack a course test.
    in_domain = [e for e in kb.load_knowledge() if _domain(e) == domain]
    assert untested_ids(in_domain, _course_test_texts()) == []


@pytest.mark.parametrize(
    "entries, texts, expected",
    [
        # pass: both course entries are named by a test text
        (
            [
                {"id": "a.one", "provenance": {"course_reference": "Owner-supplied course material, p1"}},
                {"id": "b.two", "provenance": {"course_reference": "owner-supplied course material, p2"}},
            ],
            ["get_rule('a.one')", "evaluate('b.two', x=1)"],
            [],
        ),
        # fail: b.two is from the course and no test names it; c.three is not
        # from the course, so it is not required
        (
            [
                {"id": "a.one", "provenance": {"course_reference": "owner-supplied course material, p1"}},
                {"id": "b.two", "provenance": {"course_reference": "owner-supplied course material, p2"}},
                {"id": "c.three", "provenance": {"course_reference": None}},
            ],
            ["get_rule('a.one')"],
            ["b.two"],
        ),
    ],
)
def test_untested_ids_pass_and_fail(entries, texts, expected):
    assert untested_ids(entries, texts) == expected


# ---------------------------------------------------------------------------
# 2. the course governs: calculator value == course value for the same item
# ---------------------------------------------------------------------------


def _fake_calc(required_fos: float = 1.2, other: int = 3):
    return required_fos, other


def _fake_calc_off(required_fos: float = 1.25):
    return required_fos


_COURSE_FOS = {"id": "x.fos", "thresholds": {"min_fos": 1.2}}


@pytest.mark.parametrize(
    "pairs, expected",
    [
        # pass: the calculator's default equals the course value (1.2 == 1.2)
        ([(_COURSE_FOS, "thresholds.min_fos", _fake_calc, "required_fos")], []),
        # fail: the calculator's default (1.25) differs from the course (1.2)
        (
            [(_COURSE_FOS, "thresholds.min_fos", _fake_calc_off, "required_fos")],
            [("x.fos", 1.2, 1.25)],
        ),
    ],
)
def test_disagreements_pass_and_fail(pairs, expected):
    assert disagreements(pairs) == expected


@pytest.mark.parametrize(
    "entry_id, field_path, course_figure",
    [
        # Course: FOS >= 1.2 (roads and earthworks section 1, p10; owner ruling #333).
        ("foundation.uplift_fos", "thresholds.min_fos", 1.2),
        # Course: temporary truss fillet welds 10 mm minimum (high-rise
        # structure part 3, p27 and p30 general notes).
        ("steelwork.temporary_truss_min_fillet_weld", "thresholds.value", 10),
    ],
)
def test_course_value_is_read_from_the_entry(entry_id, field_path, course_figure):
    entry = kb.get_rule(entry_id)
    assert course_value(entry, field_path) == pytest.approx(course_figure)
    assert entry["provenance"]["course_reference"].lower().startswith(COURSE_PREFIX)


# Owner ruling (15 Nov): the course governs. The calculator reads its
# required factor of safety from the course entry, so these hold.
@pytest.mark.parametrize(
    "entry_id, field_path",
    [
        # the uplift criterion itself: FOS >= 1.2 (section 1, p10)
        ("foundation.uplift_fos", "thresholds.min_fos"),
        # the same page's remedy trigger: tension piles below FOS 1.2; the
        # calculator's required factor decides the same pass/fail line
        ("foundation.uplift_remediation_tension_piles", "thresholds.trigger_below"),
    ],
)
def test_uplift_calculator_default_follows_the_course(entry_id, field_path):
    # The calculator reads its factor when it runs, so the value it applies
    # with no factor given is what the run records, not a signature default.
    from app.lib.formula_registry import recording_constants

    with recording_constants() as rec:
        dewatering_uplift_check(water_depth=10.0, raft_thickness=2.0, floor_count=3)
    applied = {r["name"]: r for r in rec.rows}["required_fos"]
    assert applied["value"] == pytest.approx(course_value(kb.get_rule(entry_id), field_path))
    assert applied.get("source")


def _floors(**kw):
    out = dewatering_uplift_check(**kw)
    return out.min_floors_for_stop if hasattr(out, "min_floors_for_stop") else out["min_floors_for_stop"]


@pytest.mark.parametrize(
    "head_m, raft_m, floor_equiv_m, floors",
    [
        # Course worked example (part 1 p9-10, as recorded in
        # foundation.dewatering_stop_floors) at the required 1.2:
        # 1.2 x 23 = 27.6 t/m2; raft 4 x 2.5 = 10; floor 0.6 x 2.5 = 1.5;
        # (27.6 - 10) / 1.5 = 11.73 -> 12 floors
        (23, 4.0, 0.6, 12),
        # invented: head 20 m, raft 2 m, floor 0.4 m equivalent:
        # 1.2 x 20 = 24; raft 2 x 2.5 = 5; floor 0.4 x 2.5 = 1.0;
        # (24 - 5) / 1.0 = 19 floors (at 1.25 it would be 20)
        (20, 2.0, 0.4, 19),
    ],
)
def test_uplift_calculator_matches_course_when_given_the_course_fos(head_m, raft_m, floor_equiv_m, floors):
    # The calculation itself follows the course once the course's 1.2 is passed.
    fos = course_value(kb.get_rule("foundation.uplift_fos"), "thresholds.min_fos")
    assert _floors(
        water_depth=head_m,
        raft_thickness=raft_m,
        floor_count=0,
        floor_thickness=floor_equiv_m,
        concrete_unit_weight=2.5,
        water_unit_weight=1.0,
        required_fos=fos,
    ) == floors


# With no factor of safety given, the calculator applies the course's 1.2.
@pytest.mark.parametrize(
    "head_m, raft_m, floor_equiv_m, floors",
    [
        (23, 4.0, 0.6, 12),  # hand value above (1.25 would give 12.5 -> 13)
        (20, 2.0, 0.4, 19),  # hand value above (1.25 would give 20)
    ],
)
def test_uplift_calculator_unstated_fos_gives_the_course_answer(head_m, raft_m, floor_equiv_m, floors):
    assert _floors(
        water_depth=head_m,
        raft_thickness=raft_m,
        floor_count=0,
        floor_thickness=floor_equiv_m,
        concrete_unit_weight=2.5,
        water_unit_weight=1.0,
    ) == floors


# ---------------------------------------------------------------------------
# 3. the calculator's course value is read from the entry, with its source
# ---------------------------------------------------------------------------


def _write_kb(tmp_path, entries):
    import json

    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "t", "entries": entries}),
                    encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    "entries, value, source",
    [
        # course entry: value read from the dotted path, the course reference is the source
        (
            [{"id": "x.fos", "thresholds": {"min_fos": 1.3},
              "provenance": {"course_reference": "owner-supplied course material, part 9 p4",
                             "source": "site notes"}}],
            1.3,
            "owner-supplied course material, part 9 p4",
        ),
        # no course reference: the entry's own source is named instead
        (
            [{"id": "x.fos", "thresholds": {"min_fos": 1.5},
              "provenance": {"course_reference": None, "source": "site notes"}}],
            1.5,
            "site notes",
        ),
    ],
)
def test_course_constant_reads_value_and_source(tmp_path, monkeypatch, entries, value, source):
    from app.agents.hats.design.formulas.construction_formulas import _course_constant

    monkeypatch.setenv("CONSTRUCTION_KB_FILE", _write_kb(tmp_path, entries))
    c = _course_constant("x.fos", "thresholds.min_fos", "ratio", "minimum FOS")
    assert (c.value, c.source) == (pytest.approx(value), source)


@pytest.mark.parametrize(
    "entry_id, field_path, says",
    [
        # no such entry: the error says the entry is missing, not a field
        ("x.absent", "thresholds.min_fos", "'x.absent' not found"),
        # entry without that field: the error names the missing field
        ("x.fos", "thresholds.max_fos", "has no 'thresholds.max_fos'"),
    ],
)
def test_course_constant_missing_raises_never_guesses(tmp_path, monkeypatch, entry_id, field_path, says):
    from app.agents.hats.design.formulas.construction_formulas import _course_constant

    monkeypatch.setenv("CONSTRUCTION_KB_FILE", _write_kb(
        tmp_path, [{"id": "x.fos", "thresholds": {"min_fos": 1.3}, "provenance": {}}]))
    with pytest.raises(LookupError) as err:
        _course_constant(entry_id, field_path, "ratio", "minimum FOS")
    assert says in str(err.value)


@pytest.mark.parametrize(
    "field_value",
    [
        True,          # a flag: float(True) would read as 1.0
        "one point two",  # a word: float() would raise ValueError
        "1.2",         # a number written as text is still not a number field
    ],
)
def test_course_constant_non_number_is_a_missing_value(tmp_path, monkeypatch, field_value):
    from app.agents.hats.design.formulas.construction_formulas import _course_constant

    monkeypatch.setenv("CONSTRUCTION_KB_FILE", _write_kb(
        tmp_path, [{"id": "x.fos", "thresholds": {"min_fos": field_value}, "provenance": {}}]))
    with pytest.raises(LookupError, match="is not a number"):
        _course_constant("x.fos", "thresholds.min_fos", "ratio", "minimum FOS")


@pytest.mark.parametrize(
    "entries, fos_given, outcome",
    [
        # no uplift entry, factor given by the caller: the check runs
        ([], 1.5, "runs"),
        # no uplift entry, no factor given: only this check fails, and says why
        ([], None, "not found"),
        # entry present, no factor given: the course's factor applies
        ([{"id": "foundation.uplift_fos", "thresholds": {"min_fos": 1.4},
           "provenance": {"course_reference": "owner-supplied course material, p1"}}], None, "runs"),
    ],
)
def test_module_loads_without_the_entry_and_only_the_check_needs_it(
        tmp_path, monkeypatch, entries, fos_given, outcome):
    import app.agents.hats.design.formulas.construction_formulas as mod

    monkeypatch.setenv("CONSTRUCTION_KB_FILE", _write_kb(tmp_path, entries))
    kw = {} if fos_given is None else {"required_fos": fos_given}
    if outcome == "runs":
        r = mod.dewatering_uplift_check(water_depth=10.0, raft_thickness=2.0, floor_count=3, **kw)
        assert r.fos == pytest.approx(5.0 / 10.0 + 3 * 0.3 * 2.5 / 10.0)
    else:
        with pytest.raises(LookupError, match=outcome):
            mod.dewatering_uplift_check(water_depth=10.0, raft_thickness=2.0, floor_count=3, **kw)


@pytest.mark.parametrize(
    "entries",
    [
        [],  # an empty knowledge base
        [{"id": "foundation.uplift_fos", "thresholds": {}, "provenance": {}}],  # entry, no field
    ],
)
def test_calculator_module_imports_without_the_course_value(tmp_path, entries):
    # A fresh interpreter: the import itself must not read the course value.
    import os
    import subprocess
    import sys

    env = dict(os.environ, CONSTRUCTION_KB_FILE=_write_kb(tmp_path, entries))
    done = subprocess.run(
        [sys.executable, "-c",
         "import app.agents.hats.design.formulas.construction_formulas"],
        cwd=str(Path(__file__).resolve().parents[1]), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("water_depth, refused", [(-0.5, True), (-3.0, True), (0.0, False), (4.0, False)])
def test_uplift_check_refuses_a_negative_water_depth(water_depth, refused):
    from app.agents.hats.design.formulas.construction_formulas import (
        dewatering_uplift_check as check,
    )

    if refused:
        with pytest.raises(ValueError, match="water_depth"):
            check(water_depth=water_depth, raft_thickness=1.0, floor_count=2)
    else:
        assert check(water_depth=water_depth, raft_thickness=1.0, floor_count=2) is not None


def _course_note(**kw):
    out = dewatering_uplift_check(water_depth=10, raft_thickness=1.0, floor_count=2, **kw)
    notes = out.notes if hasattr(out, "notes") else out["notes"]
    reference = kb.get_rule("foundation.uplift_fos")["provenance"]["course_reference"]
    return [n for n in notes if reference in n]


@pytest.mark.parametrize(
    "kwargs, named",
    [
        # no factor given: the course's value applies and the course is named
        ({}, True),
        # the caller's own factor (1.5, not the course's): no course is named
        ({"required_fos": 1.5}, False),
    ],
)
def test_uplift_result_names_the_course_only_when_its_value_applies(kwargs, named):
    assert bool(_course_note(**kwargs)) is named
