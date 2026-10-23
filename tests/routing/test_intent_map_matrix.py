"""Intent map is data — every yaml row routes both phrasings.

Adding a row must never require an app/agents/runtime.py edit. The map
lives in app/routing/intent_map.yaml (beside this matrix); runtime loads
it. This file is the golden matrix for those rows.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.agents.runtime import (
    _INTENT_MAP_PATH,
    _INTENT_TOOL_MAP,
    _forced_specific_tool,
    _load_intent_tool_map,
)

RUNTIME_PY = Path(__file__).resolve().parents[2] / "app" / "agents" / "runtime.py"
INTENT_MAP_YAML = Path(__file__).resolve().parents[2] / "app" / "routing" / "intent_map.yaml"

# Tools the matrix rows name — must be in the forced-tool available set.
AVAILABLE = {
    # Vendored from the Cerebrum-Blocks mep_coordination kit. Listed here for
    # the same reason as every other entry: the matrix asserts routing against
    # the tools a live turn would actually have.
    "mep_coordination",
    "commissioning_checklist",
    "primavera_parser",
    "construction_calc",
    "search_project_documents",
    "resource_histogram",
    "generate_wbs",
    "procurement_list_generator",
    "rfi_generator",
    "drawing_qto",
}


def _rows() -> list[dict]:
    data = yaml.safe_load(INTENT_MAP_YAML.read_text(encoding="utf-8")) or {}
    return list(data.get("rows") or [])


def _tail(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_intent_map_path_is_the_yaml_beside_this_matrix():
    assert _INTENT_MAP_PATH == INTENT_MAP_YAML
    assert _INTENT_MAP_PATH.name == "intent_map.yaml"
    assert _INTENT_MAP_PATH.parent.name == "routing"


def test_runtime_loads_every_yaml_row_in_order():
    yaml_pairs = [
        (tuple(r["phrases"]), r["tool"]) for r in _rows()
    ]
    assert list(_INTENT_TOOL_MAP) == yaml_pairs


def test_runtime_py_does_not_hardcode_the_phrase_tuples():
    """Adding a row is a yaml edit. Calc-map phrases must not live here.

    Deliverable-require phrases (``_DELIVERABLE_PHRASES``) still name
    commissioning in runtime.py — those are a different gate (tool_choice
    required vs named). The intent MAP rows are yaml-only.
    """
    src = RUNTIME_PY.read_text(encoding="utf-8")
    assert "_load_intent_tool_map" in src
    assert "_INTENT_MAP_PATH" in src
    assert "well-point spacing" not in src
    assert "score the tenders" not in src
    assert "modulus of rupture" not in src


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["tool"])
def test_every_row_has_two_distinct_phrasings(row):
    forms = [p.strip() for p in (row.get("phrasings") or []) if p and str(p).strip()]
    assert len(forms) >= 2, f"{row['tool']} needs at least two phrasings"
    assert forms[0].lower() != forms[1].lower(), f"{row['tool']} phrasings are identical"


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["tool"])
def test_both_phrasings_of_every_row_route_to_the_row_tool(row):
    """The owner's contract: BOTH forms, every row, same tool."""
    tool = row["tool"]
    forms = [p.strip() for p in (row.get("phrasings") or []) if p and str(p).strip()]
    assert len(forms) >= 2, tool
    a = _forced_specific_tool(_tail(forms[0]), AVAILABLE)
    b = _forced_specific_tool(_tail(forms[1]), AVAILABLE)
    assert a == tool, f"{tool} phrasing[0] routed to {a!r}: {forms[0]}"
    assert b == tool, f"{tool} phrasing[1] routed to {b!r}: {forms[1]}"


def test_adding_a_row_never_requires_a_runtime_py_edit(tmp_path):
    """A fourth tool appears after a yaml-only edit + reload."""
    extra = {
        "rows": [
            {
                "tool": "fixture_only_intent_tool",
                "phrases": ["extension of time claim", "eot claim letter"],
                "phrasings": [
                    "draft an extension of time claim for the wet weather",
                    "write the eot claim letter from the delay register",
                ],
            }
        ]
    }
    # Keep existing rows and append — proves the loader is row-generic.
    data = yaml.safe_load(INTENT_MAP_YAML.read_text(encoding="utf-8"))
    data["rows"] = list(data["rows"]) + extra["rows"]
    new_path = tmp_path / "intent_map.yaml"
    new_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    loaded = _load_intent_tool_map(new_path)
    tools = [tool for _phrases, tool in loaded]
    assert "fixture_only_intent_tool" in tools
    assert tools[-1] == "fixture_only_intent_tool"
    assert loaded[-1][0] == ("extension of time claim", "eot claim letter")
    # Process-level _INTENT_TOOL_MAP is unchanged — a yaml-only add does
    # not require touching runtime.py (and did not, in this test).
    src = RUNTIME_PY.read_text(encoding="utf-8")
    assert "fixture_only_intent_tool" not in src
    assert "extension of time claim" not in src
