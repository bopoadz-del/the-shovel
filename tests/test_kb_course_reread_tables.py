"""Table entries in the construction knowledge base and their lookup.

Two parts:

* The mechanism (``_knowledge.lookup_table``) on INVENTED tables written to a
  temporary KB file, two inputs per decision: unknown id, wrong entry type,
  undeclared column, row filtering, held cells returned with the rows.
* The course entries added from the high-zoom re-read of the course pages:
  the mass-concrete mock-up maxima (the side-line depth-160 time is now
  read), the post-tensioned floor cycle chart in quarter-day cells (two rows
  print no bar), and the temporary truss drawings' minimum fillet weld.
"""

from __future__ import annotations

import json

import pytest

from app.blocks import _knowledge


# ---------------------------------------------------------------------------
# Mechanism, on invented tables
# ---------------------------------------------------------------------------


def _invented_kb():
    return {
        "schema_version": "1.1.0",
        "kb_version": "test",
        "entries": [
            {
                "id": "invented.pump_trials",
                "domain": "construction.concrete",
                "type": "table",
                "title": "Invented pump trials",
                "statement": "Invented rows for the lookup mechanism.",
                "table": {
                    "columns": [
                        {"name": "rig", "unit": None},
                        {"name": "stage", "unit": None},
                        {"name": "pressure", "unit": "bar"},
                    ],
                    "rows": [
                        {"rig": "north", "stage": 1, "pressure": 41.5},
                        {"rig": "north", "stage": 2, "pressure": None},
                        {"rig": "south", "stage": 1, "pressure": 37.0},
                    ],
                    "held": [
                        {"rig": "north", "stage": 2, "column": "pressure", "reason": "invented hold"}
                    ],
                },
                "applicability": {"applies_to": ["construction.concrete"], "project_specific": True},
                "provenance": {"source": "invented"},
                "credibility_tier": 2,
            },
            {
                "id": "invented.crew_sizes",
                "domain": "construction.buildings",
                "type": "table",
                "title": "Invented crew sizes",
                "statement": "Second invented table, no held cells.",
                "table": {
                    "columns": [{"name": "trade", "unit": None}, {"name": "crew", "unit": None}],
                    "rows": [{"trade": "fixer", "crew": 6}, {"trade": "carpenter", "crew": 9}],
                },
                "provenance": {"source": "invented"},
                "credibility_tier": 5,
            },
            {
                "id": "invented.a_formula",
                "type": "formula",
                "expression": "a*b",
                "variables": {"a": {}, "b": {}},
            },
            {
                "id": "invented.a_checklist",
                "type": "checklist",
                "checklist": ["one", "two"],
            },
        ],
    }


@pytest.fixture
def invented_kb(tmp_path, monkeypatch):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(_invented_kb()), encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    monkeypatch.setattr(_knowledge, "_KB_CACHE", None)
    monkeypatch.setattr(_knowledge, "_KB_MTIME", 0.0)
    yield path
    monkeypatch.setattr(_knowledge, "_KB_CACHE", None)
    monkeypatch.setattr(_knowledge, "_KB_MTIME", 0.0)


@pytest.mark.parametrize("rule_id", ["invented.no_such_table", "invented.also_missing"])
def test_unknown_table_id_raises_key_error(invented_kb, rule_id):
    with pytest.raises(KeyError):
        _knowledge.lookup_table(rule_id)


@pytest.mark.parametrize("rule_id,kind", [("invented.a_formula", "formula"), ("invented.a_checklist", "checklist")])
def test_non_table_entry_is_refused(invented_kb, rule_id, kind):
    with pytest.raises(ValueError, match=kind):
        _knowledge.lookup_table(rule_id)


@pytest.mark.parametrize(
    "rule_id,match",
    [
        ("invented.pump_trials", {"valve": "open"}),
        ("invented.crew_sizes", {"trade": "fixer", "shift": 2}),
    ],
)
def test_filter_on_undeclared_column_is_refused(invented_kb, rule_id, match):
    with pytest.raises(ValueError, match="has no column"):
        _knowledge.lookup_table(rule_id, **match)


@pytest.mark.parametrize(
    "rule_id,match,expected",
    [
        ("invented.pump_trials", {"rig": "north"}, [41.5, None]),
        ("invented.pump_trials", {"rig": "south", "stage": 1}, [37.0]),
    ],
)
def test_rows_filtered_on_every_given_column(invented_kb, rule_id, match, expected):
    out = _knowledge.lookup_table(rule_id, **match)
    assert [r["pressure"] for r in out["rows"]] == expected


@pytest.mark.parametrize(
    "rule_id,match,expected_crews",
    [
        ("invented.crew_sizes", {}, [6, 9]),
        ("invented.crew_sizes", {"trade": "carpenter"}, [9]),
    ],
)
def test_no_filter_returns_all_rows_and_one_filter_narrows(invented_kb, rule_id, match, expected_crews):
    out = _knowledge.lookup_table(rule_id, **match)
    assert [r["crew"] for r in out["rows"]] == expected_crews


@pytest.mark.parametrize(
    "rule_id,held_count,warn_count",
    [
        ("invented.pump_trials", 1, 2),  # tier 2 + project_specific
        ("invented.crew_sizes", 0, 0),  # tier 5, no applicability flags
    ],
)
def test_held_cells_and_warnings_come_back_with_the_rows(invented_kb, rule_id, held_count, warn_count):
    out = _knowledge.lookup_table(rule_id)
    assert out["rule_id"] == rule_id
    assert len(out["held"]) == held_count
    assert len(out["warnings"]) == warn_count
    assert out["provenance"] == {"source": "invented"}
    assert [c["name"] for c in out["columns"]]


# ---------------------------------------------------------------------------
# Course entries (re-read at high zoom)
# ---------------------------------------------------------------------------

MOCKUP = "thermal.mass_pour_mockup_peak_temperatures"
CYCLE = "highrise.pt_floor_five_day_cycle_chart"
WELD = "steelwork.temporary_truss_min_fillet_weld"


@pytest.mark.parametrize(
    "line,depth,temp,hours",
    [
        ("side", "160", 59, 32.68),  # newly read on the re-read (was ILLEGIBLE)
        ("side", "340", 57, 27.05),  # verify-file correction of the transcribed 27.55
        ("intermediate", "70", 64, 38.55),  # verify-file correction of the transcribed 38.58
        ("centre", "160", 66, 107.92),
    ],
)
def test_mockup_cells_as_read(line, depth, temp, hours):
    rows = _knowledge.lookup_table(MOCKUP, line=line, depth_label=depth)["rows"]
    assert len(rows) == 1
    assert rows[0]["max_temperature"] == temp
    assert rows[0]["time_of_max"] == pytest.approx(hours)


@pytest.mark.parametrize(
    "line,depth,column",
    [("centre", "bottom", "time_of_max"), ("corner", "160", "max_temperature")],
)
def test_mockup_unreadable_cells_are_null_and_listed_as_held(line, depth, column):
    out = _knowledge.lookup_table(MOCKUP, line=line, depth_label=depth)
    assert out["rows"][0][column] is None
    assert {"line": line, "depth_label": depth, "column": column} in [
        {k: h[k] for k in ("line", "depth_label", "column")} for h in out["held"]
    ]


@pytest.mark.parametrize(
    "line,expected_peak,expected_hours",
    [("centre", 66, [107.92, 94.33]), ("side", 60, [42.1, 40.28])],
)
def test_mockup_line_peak(line, expected_peak, expected_hours):
    rows = _knowledge.lookup_table(MOCKUP, line=line)["rows"]
    peak = max(r["max_temperature"] for r in rows if r["max_temperature"] is not None)
    assert peak == expected_peak
    assert [r["time_of_max"] for r in rows if r["max_temperature"] == peak] == expected_hours


@pytest.mark.parametrize("activity", ["Strip Edge", "Stress Tendon"])
def test_cycle_rows_without_a_bar_carry_no_position(activity):
    rows = _knowledge.lookup_table(CYCLE, activity=activity)["rows"]
    assert len(rows) == 1
    assert rows[0]["bar_printed"] is False
    assert (rows[0]["day"], rows[0]["first_quarter"], rows[0]["last_quarter"]) == (None, None, None)


@pytest.mark.parametrize(
    "activity,segments",
    [
        ("Column Reinforcement", [(1, 2, 3), (2, 1, 2)]),  # day 1 cell 4 empty between the two bars
        ("Slab Formwork", [(3, 1, 4), (4, 1, 1)]),
        ("Concrete", [(5, 1, 3)]),
        ("Grout and Finish", [(9, 1, 3)]),
    ],
)
def test_cycle_bars_in_quarter_day_cells(activity, segments):
    rows = _knowledge.lookup_table(CYCLE, activity=activity)["rows"]
    assert [(r["day"], r["first_quarter"], r["last_quarter"]) for r in rows] == segments


@pytest.mark.parametrize("chart_row,segments", [(1, [(1, 1, 2)]), (12, [(6, 1, 3), (9, 1, 3)])])
def test_cycle_two_column_formwork_rows_kept_apart_by_chart_row(chart_row, segments):
    rows = _knowledge.lookup_table(CYCLE, activity="Column Formwork", chart_row=chart_row)["rows"]
    assert [(r["day"], r["first_quarter"], r["last_quarter"]) for r in rows] == segments


@pytest.mark.parametrize("rule_id", [MOCKUP, CYCLE])
def test_course_tables_cite_course_pages_and_cells_stay_in_range(rule_id):
    out = _knowledge.lookup_table(rule_id)
    assert out["provenance"]["course_reference"].startswith("owner-supplied course material")
    assert out["warnings"]  # tier 3 always warns
    for row in out["rows"]:
        for q in ("first_quarter", "last_quarter"):
            if row.get(q) is not None:
                assert 1 <= row[q] <= 4
        if row.get("first_quarter") is not None:
            assert row["first_quarter"] <= row["last_quarter"]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("type", "threshold"),
        ("thresholds.value", 10),
        ("thresholds.unit", "millimeter"),
        ("applicability.project_specific", True),
    ],
)
def test_truss_min_fillet_weld_note(path, expected):
    node = _knowledge.get_rule(WELD)
    for key in path.split("."):
        node = node[key]
    assert node == expected


@pytest.mark.parametrize("rule_id,page", [(WELD, "p27"), (CYCLE, "p10")])
def test_reread_entries_cite_their_course_page(rule_id, page):
    assert page in _knowledge.get_rule(rule_id)["provenance"]["course_reference"]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("minimum fillet weld size truss", WELD),
        ("floor cycle chart tendon slab formwork", CYCLE),
    ],
)
def test_new_entries_are_found_by_search(query, expected):
    assert expected in [e["id"] for e in _knowledge.search_knowledge(query, top_k=5)]
