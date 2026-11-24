"""An ask the encoded knowledge covers reaches the knowledge tool first (D5a).

Every knowledge base and ask below is built in this file. Each decision has
an input that takes it and one that does not.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import runtime
from app.blocks import _knowledge as kb


def _entries():
    # Three made-up entries. "lintel" and "bearing" name one entry only;
    # "steel" and "length" are shared, so they weigh less.
    return [
        {"id": "masonry.lintel_bearing", "type": "rule",
         "title": "Lintel end bearing length",
         "statement": "A steel lintel bears at least 150 mm each end."},
        {"id": "steel.bolt_edge", "type": "rule",
         "title": "Bolt edge distance in steel plate",
         "statement": "Minimum edge distance is 1.5 times the hole length."},
        {"id": "timber.joist_notch", "type": "rule",
         "title": "Joist notch depth",
         "statement": "A notch is no deeper than one eighth of the joist depth."},
    ]


@pytest.fixture
def made_kb(monkeypatch):
    monkeypatch.setattr(kb, "load_knowledge", lambda domain=None: _entries())


# --- entry_covering_ask: the routing decision ------------------------------

def test_ask_holding_an_entrys_naming_words_is_covered(made_kb):
    entry = kb.entry_covering_ask("What end bearing does a lintel need?")
    assert entry is not None and entry["id"] == "masonry.lintel_bearing"


def test_second_entry_is_picked_by_its_own_words(made_kb):
    entry = kb.entry_covering_ask("how deep can I notch a joist")
    assert entry is not None and entry["id"] == "timber.joist_notch"


def test_ask_sharing_only_common_words_is_not_covered(made_kb):
    # "steel" and "length" appear in two entries; the naming words do not.
    assert kb.entry_covering_ask("What length of steel is on site?") is None


def test_unrelated_ask_is_not_covered(made_kb):
    assert kb.entry_covering_ask("Who signed the variation order?") is None


def test_empty_ask_is_not_covered(made_kb):
    assert kb.entry_covering_ask("") is None
    assert kb.entry_covering_ask("   ") is None


@pytest.mark.parametrize(
    "ask, expected_loads",
    [
        ("?! ...", 0),
        ("", 0),
        ("lintel bearing", 1),
        ("bearing of a lintel", 1),
    ],
)
def test_wordless_ask_never_reads_the_knowledge(monkeypatch, ask, expected_loads):
    # An ask with no words is decided before the entries are loaded; an ask
    # with words reads them. Counting the loads separates the two.
    loads = []

    def counting_load(domain=None):
        loads.append(domain)
        return _entries()

    monkeypatch.setattr(kb, "load_knowledge", counting_load)
    kb.entry_covering_ask(ask)
    assert len(loads) == expected_loads


def test_share_below_the_majority_is_refused(made_kb, monkeypatch):
    ask = "lintel end bearing"
    assert kb.entry_covering_ask(ask) is not None
    # The same ask fails once the bar asks for more than it holds.
    monkeypatch.setattr(kb, "ASK_COVERAGE_SHARE", 1.01)
    assert kb.entry_covering_ask(ask) is None


def test_weights_follow_the_entries_not_a_word_list():
    one = kb._token_weights([{"id": "a.x", "title": "alpha", "statement": "common"},
                             {"id": "b.y", "title": "beta", "statement": "common"}])
    assert one["alpha"] > one["common"]
    two = kb._token_weights([{"id": "a.x", "title": "alpha", "statement": "alpha"},
                             {"id": "b.y", "title": "alpha", "statement": "beta"}])
    assert two["beta"] > two["alpha"]


def test_entry_with_no_naming_words_is_skipped(monkeypatch):
    monkeypatch.setattr(kb, "load_knowledge", lambda domain=None: [
        {"id": "", "title": "", "statement": "lintel bearing"}])
    assert kb.entry_covering_ask("lintel bearing") is None


# --- _predispatch_kb_knowledge: the turn records the search -----------------

class _Agent:
    def __init__(self, allowed, matches):
        self.allowed_blocks = allowed
        self._matches = matches
        self.calls = []

    async def _run_tool_call(self, tc):
        self.calls.append(tc)
        return {"name": tc["function"]["name"], "ok": True,
                "result": {"status": "success", "matches": self._matches,
                           "count": len(self._matches)}}


_MATCH = [{"id": "masonry.lintel_bearing", "title": "Lintel end bearing length",
           "provenance": {"source": "made-up course notes, p3"}}]


@pytest.fixture
def tool_present(monkeypatch):
    from app.agents.core import tool_registry
    real = tool_registry.get
    monkeypatch.setattr(tool_registry, "get",
                        lambda name: object() if name == runtime._KB_TOOL else real(name))


def _run(agent, ask):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": ask}]
    out = asyncio.run(runtime._predispatch_kb_knowledge(agent, messages, operator_text=ask))
    return out, messages


def test_covered_ask_records_the_search_with_provenance(made_kb, tool_present):
    agent = _Agent([runtime._KB_TOOL], _MATCH)
    out, messages = _run(agent, "What end bearing does a lintel need?")
    assert out and out["name"] == runtime._KB_TOOL and out["predispatched"]
    assert json.loads(agent.calls[0]["function"]["arguments"])["query"].startswith("What end")
    tool_msg = [m for m in messages if m.get("role") == "tool"]
    assert tool_msg and "made-up course notes" in tool_msg[0]["content"]
    assert "not a project document" in messages[-1]["content"]


def test_an_unreached_model_does_not_print_the_search_as_the_answer(made_kb, tool_present):
    # The recorded search is evidence for the model, never the reply (G-8):
    # with the provider down, nothing is recovered from it, so the caller
    # keeps its honest failure message. A real pre-dispatched draft in the
    # same position is still recovered, so the rule tells the two apart.
    agent = _Agent([runtime._KB_TOOL], _MATCH)
    out, messages = _run(agent, "What end bearing does a lintel need?")
    assert out is not None
    assert runtime._recover_after_provider_failure(messages) == ""

    drafted = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    runtime._inject_predispatch(drafted, "made_tool", "Lintel bearing: 150 mm each end.",
                                "Present this draft.")
    assert runtime._recover_after_provider_failure(drafted) == "Lintel bearing: 150 mm each end."


def test_uncovered_ask_runs_no_search(made_kb, tool_present):
    agent = _Agent([runtime._KB_TOOL], _MATCH)
    out, messages = _run(agent, "Who signed the variation order?")
    assert out is None and agent.calls == [] and len(messages) == 2


def test_ask_about_a_project_record_stays_with_the_documents(made_kb, tool_present):
    agent = _Agent([runtime._KB_TOOL], _MATCH)
    out, _ = _run(agent, "What lintel end bearing does the specification say?")
    assert out is None and agent.calls == []
    out, _ = _run(agent, "What lintel end bearing is needed?")
    assert out is not None


def test_agent_without_the_tool_is_left_alone(made_kb, tool_present):
    agent = _Agent(["construction"], _MATCH)
    out, _ = _run(agent, "What end bearing does a lintel need?")
    assert out is None and agent.calls == []


def test_search_with_no_matches_adds_nothing(made_kb, tool_present):
    agent = _Agent([runtime._KB_TOOL], [])
    out, messages = _run(agent, "What end bearing does a lintel need?")
    assert out is None and len(agent.calls) == 1 and len(messages) == 2


# --- the two chat entries call it (Grok S671, Rule 4) ----------------------
# Each entry is driven end to end with a model that calls no tool. The
# calculator probe and the knowledge probe are replaced by recorders, so the
# only thing under test is the wiring at the call site: the knowledge probe
# runs when no calculator was pre-dispatched, and not when one was.

_ASK = "What end bearing does a lintel need?"


def _kb_record():
    return {"name": runtime._KB_TOOL, "ok": True, "predispatched": True,
            "result": {"status": "success", "matches": _MATCH, "count": 1},
            "falls_to_retrieval": False}


def _calc_record():
    return {"name": "construction_calc", "ok": True, "predispatched": True,
            "result": {"status": "success", "result": {"value": 1.0}}}


def _wire_entry(monkeypatch, calc_fires):
    seen = {"kb": [], "calc": []}

    async def calc(agent, messages, project_id, *, operator_text=None, **kw):
        seen["calc"].append(operator_text)
        return _calc_record() if calc_fires else None

    async def kb_probe(agent, messages, *, operator_text=None, **kw):
        seen["kb"].append(operator_text)
        return _kb_record()

    async def call_llm(_self, messages, api_key, **kwargs):
        return {"status": "success",
                "choice": {"message": {"role": "assistant",
                                       "content": "A lintel bears 150 mm each end."}}}

    monkeypatch.setattr(runtime, "_predispatch_formula_calc", calc)
    monkeypatch.setattr(runtime, "_predispatch_kb_knowledge", kb_probe)
    monkeypatch.setattr(runtime.Agent, "_call_llm", call_llm)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "0")
    return seen


def _entry_agent():
    return runtime.Agent(name="project-assistant", description="t", system_prompt="t",
                         allowed_blocks=[runtime._KB_TOOL])


@pytest.mark.parametrize("calc_fires, kb_runs", [(False, True), (True, False)])
def test_chat_entry_consults_knowledge_only_when_no_calculator_ran(
        monkeypatch, calc_fires, kb_runs):
    from tests._dev_key import DEV_KEY
    seen = _wire_entry(monkeypatch, calc_fires)
    out = asyncio.run(_entry_agent().chat(_ASK, api_key=DEV_KEY, project_id=None))
    assert seen["calc"] == [_ASK], seen
    assert seen["kb"] == ([_ASK] if kb_runs else []), seen
    names = [t.get("name") for t in (out.get("tool_calls") or [])]
    assert (runtime._KB_TOOL in names) is kb_runs, names


@pytest.mark.parametrize("calc_fires, kb_runs", [(False, True), (True, False)])
def test_stream_entry_consults_knowledge_only_when_no_calculator_ran(
        monkeypatch, calc_fires, kb_runs):
    seen = _wire_entry(monkeypatch, calc_fires)

    async def drain():
        return [ev async for ev in _entry_agent().chat_stream(
            user_message=_ASK, history=[], project_id=None,
            conversation_id=None, user_id=None)]

    events = asyncio.run(drain())
    assert seen["calc"] == [_ASK], seen
    assert seen["kb"] == ([_ASK] if kb_runs else []), seen
    results = [e.get("name") for e in events
               if e.get("type") == "tool_result" and e.get("predispatched")]
    assert (runtime._KB_TOOL in results) is kb_runs, results
