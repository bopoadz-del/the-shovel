"""The construction knowledge base is a chat tool (BUILD 2).

Each test writes its own invented knowledge base (CONSTRUCTION_KB_FILE) and
registers the block itself, so the result does not depend on the shipped
entries or on the boot profile. Two inputs per decision.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents.base import construction_advisor as advisor_tool
from app.agents.core import tool_registry
from app.agents.core import tools as core_tools
from app.blocks import _knowledge as kb
from app.blocks.construction_advisor import ConstructionAdvisorBlock

TOOL = "construction_advisor"


def _formula(rule_id, title, expression, variables, unit, tier, source):
    return {
        "id": rule_id, "domain": "invented.domain", "type": "formula",
        "title": title, "statement": title + ".", "expression": expression,
        "variables": {v: {"unit": "-", "desc": v} for v in variables},
        "thresholds": {"unit": unit},
        "applicability": {"applies_to": ["invented.domain"]},
        "provenance": {"source": source}, "credibility_tier": tier,
    }


def _rule(rule_id, title, tier, source):
    return {
        "id": rule_id, "domain": "invented.domain", "type": "rule",
        "title": title, "statement": title + ".",
        "applicability": {"applies_to": ["invented.domain"]},
        "provenance": {"source": source}, "credibility_tier": tier,
    }


ENTRIES = [
    _formula("quarry.gravel_yield", "Gravel yield of a quarry bench", "L * W * H * k",
             ["L", "W", "H", "k"], "m3", 2, "invented quarry handbook"),
    _formula("lagoon.fill_rate", "Lagoon fill rate from inflow and seepage", "(Q - S) / A",
             ["Q", "S", "A"], "m/day", 4, "invented hydrology note"),
    _formula("kiln.energy_index", "Kiln energy index", "E * I / N",
             ["E", "I", "N"], "-", 3, "invented kiln note"),
    _rule("scaffold.tie_spacing", "Scaffold tie spacing on a windy facade", 1,
          "invented scaffold standard"),
    _rule("formwork.strip_time", "Formwork strip time for a cold slab", 3,
          "invented formwork guide"),
]


@pytest.fixture
def invented_kb(tmp_path, monkeypatch):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "t", "entries": ENTRIES}),
                    encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    monkeypatch.setattr(kb, "_KB_CACHE", None)
    import app.blocks as blocks
    monkeypatch.setitem(blocks.BLOCK_REGISTRY, TOOL, ConstructionAdvisorBlock)
    return path


def _agent(allowed):
    from app.agents.runtime import Agent
    return Agent(name="advisor-probe", description="probe", system_prompt="",
                 allowed_blocks=list(allowed))


def _call(args):
    agent = _agent([TOOL])
    tool_call = {"id": "c1", "type": "function",
                 "function": {"name": TOOL, "arguments": json.dumps(args)}}
    return asyncio.run(agent._run_tool_call(tool_call))


def _offered_names(agent, project_id=None):
    return [(t.get("function") or {}).get("name") for t in agent.tool_definitions(project_id=project_id)]


# -- the tool is registered in the package registry, with its schema -------

@pytest.mark.parametrize("path,want", [
    (("function", "name"), TOOL),
    (("function", "parameters", "required"), ["query"]),
])
def test_the_tool_is_a_base_registry_tool_with_a_declared_schema(path, want):
    spec = tool_registry.get(TOOL)
    assert spec is not None and spec.owner == "base"
    assert TOOL in tool_registry.tools_of("base")
    node = spec.schema
    for key in path:
        node = node[key]
    assert node == want


# -- search: each match carries its source and tier ------------------------

@pytest.mark.parametrize("query,want_id,want_source,want_tier", [
    ("scaffold tie spacing windy facade", "scaffold.tie_spacing", "invented scaffold standard", 1),
    ("formwork strip time cold slab", "formwork.strip_time", "invented formwork guide", 3),
])
def test_a_search_returns_the_rule_with_its_source_and_tier(invented_kb, query, want_id,
                                                            want_source, want_tier):
    out = _call({"query": query})
    assert out["ok"] is True
    top = out["result"]["matches"][0]
    assert top["id"] == want_id
    assert top["provenance"]["source"] == want_source
    assert top["credibility_tier"] == want_tier


# -- evaluate with stated values, the named formula ------------------------

@pytest.mark.parametrize("rule_id,values,want,unit,tier", [
    ("quarry.gravel_yield", {"L": 12, "W": 5, "H": 2.5, "k": 0.8}, 12 * 5 * 2.5 * 0.8, "m3", 2),
    ("lagoon.fill_rate", {"Q": 340, "S": 40, "A": 1500}, (340 - 40) / 1500, "m/day", 4),
    # variable names sympy reserves for its own objects still substitute
    ("kiln.energy_index", {"E": 6, "I": 7, "N": 3}, 6 * 7 / 3, "-", 3),
])
def test_stated_values_evaluate_the_named_formula(invented_kb, rule_id, values, want, unit, tier):
    # The query matches a different entry first: the named rule_id decides.
    out = _call({"query": "scaffold tie spacing", "rule_id": rule_id, "values": values})
    ev = out["result"]["evaluation"]
    assert ev["rule_id"] == rule_id
    assert ev["result"] == pytest.approx(want)
    assert ev["unit"] == unit
    assert ev["credibility_tier"] == tier
    assert ev["provenance"]["source"]


@pytest.mark.parametrize("values", [
    json.dumps({"L": 3, "W": 4, "H": 2, "k": 1.0}),   # a model sending the object as a string
    {"L": 3, "W": 4, "H": 2, "k": 1.0},
])
def test_values_as_object_or_json_string_both_evaluate(invented_kb, values):
    out = _call({"query": "gravel yield quarry bench", "values": values})
    assert out["result"]["evaluation"]["result"] == pytest.approx(24.0)


@pytest.mark.parametrize("values", [{"L": 3}, {"Q": 9}])
def test_missing_variables_are_an_evaluation_error_not_a_figure(invented_kb, values):
    out = _call({"query": "x", "rule_id": "quarry.gravel_yield", "values": values})
    assert "evaluation" not in out["result"]
    assert out["result"]["evaluation_error"]


@pytest.mark.parametrize("values", [["L", 3], "not json"])
def test_values_that_are_not_an_object_are_refused(invented_kb, values):
    out = _call({"query": "gravel yield", "values": values})
    assert "evaluation" not in out["result"]
    assert "values must be an object" in out["result"]["evaluation_error"]


@pytest.mark.parametrize("query", ["gravel yield quarry bench", "lagoon fill rate seepage"])
def test_without_values_nothing_is_evaluated(invented_kb, query):
    out = _call({"query": query})
    assert out["ok"] is True
    assert "evaluation" not in out["result"] and "evaluation_error" not in out["result"]


# -- offered in the chat tool list --------------------------------------------

@pytest.mark.parametrize("project_id", [None, "p-advisor"])
def test_an_agent_that_allows_the_block_is_offered_the_declared_schema(invented_kb, project_id):
    tools = _agent([TOOL]).tool_definitions(project_id=project_id)
    mine = [t for t in tools if (t.get("function") or {}).get("name") == TOOL]
    assert mine == [tool_registry.get(TOOL).schema]


@pytest.mark.parametrize("allowed", [[], ["sympy_reasoning"]])
def test_an_agent_without_the_block_is_not_offered_it(invented_kb, allowed):
    assert TOOL not in _offered_names(_agent(allowed))


@pytest.mark.parametrize("config", ["project-assistant", "construction-pm"])
def test_the_construction_chat_agents_allow_the_block(config):
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "app" / "agents" / "configs"
            / f"{config}.md").read_text(encoding="utf-8")
    head = text.split("---")[1]
    assert f"  - {TOOL}\n" in head.replace("\r\n", "\n")


# -- G-9: the rule over every config on disk, none named ----------------------
# The rule is the configs' own: a config whose loaded allowed blocks include
# the advisor block carries the allow-list line, and a config that carries
# the line has the block in its loaded allowed blocks. The configs are found
# by globbing the directory, so a future config is checked the day it lands.

def _carries_the_line(md) -> bool:
    head = md.read_text(encoding="utf-8").replace("\r\n", "\n").split("---")[1]
    return f"  - {TOOL}\n" in head


def _allows_the_block(md) -> bool:
    from app.agents.runtime import _parse_agent_file
    return TOOL in _parse_agent_file(md).allowed_blocks


def _configs_breaking_the_rule(configs_dir) -> list[str]:
    return [md.name for md in sorted(configs_dir.glob("*.md"))
            if _allows_the_block(md) != _carries_the_line(md)]


def test_every_config_on_disk_allows_the_block_exactly_when_it_carries_the_line():
    from app.agents.runtime import CONFIGS_DIR
    configs = sorted(CONFIGS_DIR.glob("*.md"))
    assert configs, "no agent config found on disk"
    assert _configs_breaking_the_rule(CONFIGS_DIR) == []
    # Both branches of the implication are exercised by the configs on disk.
    with_block = [md for md in configs if _allows_the_block(md)]
    assert 0 < len(with_block) < len(configs)


def _write_config(directory, stem, frontmatter_tail):
    (directory / f"{stem}.md").write_text(
        f"---\nname: {stem}\ndescription: invented agent\n{frontmatter_tail}---\n\n"
        "You are an invented agent used only by this test.\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("pair,want_breaks", [
    # Both branches held: one config with block and line, one with neither.
    ({"alpha-holds": f"allowed_blocks:\n  - sympy_reasoning\n  - {TOOL}\n",
      "beta-holds": "allowed_blocks:\n  - sympy_reasoning\n"},
     []),
    # Both branches broken: the block allowed without the line (a tab list
    # the loader accepts), and the line present without the block (under a
    # key the loader does not read as allowed blocks).
    ({"gamma-block-no-line": f"allowed_blocks:\n\t- {TOOL}\n",
      "delta-line-no-block": f"allowed_blocks:\n  - sympy_reasoning\nretired_blocks:\n  - {TOOL}\n"},
     ["delta-line-no-block.md", "gamma-block-no-line.md"]),
])
def test_the_rule_flags_a_synthetic_config_on_either_branch(tmp_path, pair, want_breaks):
    for stem, tail in pair.items():
        _write_config(tmp_path, stem, tail)
    assert _configs_breaking_the_rule(tmp_path) == want_breaks


# -- not installed: an honest refusal -----------------------------------------

@pytest.mark.parametrize("query", ["scaffold tie spacing", "gravel yield"])
def test_without_the_block_installed_the_tool_refuses(invented_kb, monkeypatch, query):
    import app.blocks as blocks
    monkeypatch.delitem(blocks.BLOCK_REGISTRY, TOOL)
    out = _call({"query": query})
    assert out["ok"] is False
    assert "not installed" in out["result"]["error"]


# -- the handler and the offer, called directly on their own modules ---------

@pytest.mark.parametrize("raw,want", [
    (json.dumps({"Q": 340, "S": 40}), {"Q": 340, "S": 40}),   # JSON text is decoded
    ({"L": 3, "W": 4}, {"L": 3, "W": 4}),                     # an object passes as is
])
def test_values_decodes_json_text_and_keeps_an_object(raw, want):
    assert advisor_tool._values(raw) == want


@pytest.mark.parametrize("args,want_id", [
    ({"query": "scaffold tie spacing windy facade"}, "scaffold.tie_spacing"),
    ({"query": "formwork strip time cold slab"}, "formwork.strip_time"),
])
def test_handle_construction_advisor_answers_a_call_directly(invented_kb, args, want_id):
    call = tool_registry.ToolCall(agent=None, name=TOOL, args=args)
    out = asyncio.run(advisor_tool.handle_construction_advisor(call))
    assert out["ok"] is True and out["name"] == TOOL
    assert out["result"]["matches"][0]["id"] == want_id


@pytest.mark.parametrize("allowed,offered", [([TOOL], True), (["sympy_reasoning"], False)])
def test_core_tool_definitions_offers_the_declared_schema_only_when_allowed(invented_kb,
                                                                            allowed, offered):
    defs = core_tools.tool_definitions(_agent(allowed), project_id=None)
    mine = [t for t in defs if (t.get("function") or {}).get("name") == TOOL]
    assert mine == ([advisor_tool.SCHEMA] if offered else [])


@pytest.mark.parametrize("name", [TOOL, "invented_block_never_installed"])
def test_core_tool_definitions_never_offers_a_block_that_is_not_installed(invented_kb,
                                                                          monkeypatch, name):
    import app.blocks as blocks
    monkeypatch.delitem(blocks.BLOCK_REGISTRY, TOOL)
    defs = core_tools.tool_definitions(_agent([name]), project_id=None)
    assert name not in [(t.get("function") or {}).get("name") for t in defs]


@pytest.mark.parametrize("project_id,offered", [(None, False), ("p-advisor", True)])
def test_core_tool_definitions_offers_project_search_only_with_a_project(invented_kb,
                                                                         project_id, offered):
    defs = core_tools.tool_definitions(_agent([TOOL]), project_id=project_id)
    names = [(t.get("function") or {}).get("name") for t in defs]
    assert ("search_project_documents" in names) is offered
