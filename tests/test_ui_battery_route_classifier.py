"""The ONE-USER UI battery runner reads the route the product TOOK from the
SSE turn, never guesses it. These exercise the two pure pieces -- turn_from_
events (raw SSE -> normalized turn) and classify_route (turn -> route label) --
on SYNTHETIC events only; no battery questions, answers or client figures live
here or in the runner. Full dotted import so the mutation gate selects this.
"""
from __future__ import annotations

import pytest

from scripts.ui_battery_runner import (
    INTENT_COMPUTE,
    INTENT_DOC_OP,
    INTENT_LOOKUP,
    ROUTE_ASKS_INPUTS,
    ROUTE_CALCULATOR,
    ROUTE_GENERAL_RAG,
    ROUTE_PROJECT_RAG,
    ROUTE_TOOL_ON_DOC,
    ROUTE_UNCLASSIFIED,
    _grounded,
    _pick_project,
    battery_exit_code,
    classify_intent,
    classify_route,
    turn_from_events,
)


@pytest.mark.parametrize("passed,total,expected_code", [
    (20, 20, 0),   # all routes passed -> exit 0 (green)
    (4, 20, 1),    # a partial score must be RED, never a silent green
    (0, 0, 1),     # an empty battery is a setup error -> non-zero
])
def test_battery_exit_code_is_zero_only_on_a_full_pass(passed, total, expected_code):
    assert battery_exit_code(passed, total) == expected_code


def _turn(tools=(), sources=(), provenance=(), answer=""):
    return {"tools": list(tools), "sources": list(sources),
            "provenance": list(provenance), "answer": answer}


@pytest.mark.parametrize("turn,expected", [
    # a calculator run -> LLM + calculator
    (_turn(tools=["construction_calc"], sources=[{"source_class": "project_corpus"}]),
     ROUTE_CALCULATOR),
    # a tool over a document / deliverable
    (_turn(tools=["generate_wbs"]), ROUTE_TOOL_ON_DOC),
    (_turn(tools=["fetch_document"]), ROUTE_TOOL_ON_DOC),
    # retrieved project corpus, no tool -> project RAG
    (_turn(sources=[{"source_class": "project_corpus"}]), ROUTE_PROJECT_RAG),
    # retrieved general/master corpus -> general RAG
    (_turn(sources=[{"source_class": "master_corpus"}]), ROUTE_GENERAL_RAG),
    # no tool, no retrieved source, no provenance -> nothing to attribute the
    # answer to: unclassified (a FAIL), NOT silently passed as general RAG
    (_turn(answer="Curing keeps concrete moist for 7 days."), ROUTE_UNCLASSIFIED),
    # an input request wins even when a calculator was attempted but came back
    # empty (no credited calc figure in provenance)
    (_turn(tools=["construction_calc"],
           answer="The delay damages calculator needs the rate and contract amount."),
     ROUTE_ASKS_INPUTS),
    # the inverse: the answer carries the same "needs"/"what values" phrasing but
    # a calculator figure WAS credited -> a calculator turn, not an input request
    (_turn(tools=["construction_calc"],
           answer="You asked what values I used; the rate needs no change.",
           provenance=[{"figure": "SAR 10000", "source": "calculator",
                        "inputs": [{"name": "rate_percent", "source": "project_data"}]}]),
     ROUTE_CALCULATOR),
])
def test_classify_route_reads_the_observed_route(turn, expected):
    assert classify_route(turn) == expected


# ── S2: question-intent layer (ORDERS_03NOV_0152Z/0345Z/0420Z) ───────────────


@pytest.mark.parametrize("message,expected_intent", [
    # LOOKUP: interrogative asking what a value IS (>=2 phrasings)
    ("What is the Defects Notification Period?", INTENT_LOOKUP),
    ("What is the HSE lighting (lux) requirement for general site areas?", INTENT_LOOKUP),
    # COMPUTE: a calculate verb, arithmetic, or a quantity produced from operands
    ("Calculate the delay damages.", INTENT_COMPUTE),
    ("Concrete volume for a raft 30 x 20 x 1.5 m?", INTENT_COMPUTE),
    ("Scaffold load capacity for the stated configuration?", INTENT_COMPUTE),
    # DOC_OPERATION: an imperative over a document / deliverable
    ("Summarise the named specification document.", INTENT_DOC_OP),
    ("Build the WBS / work-breakdown for the project.", INTENT_DOC_OP),
])
def test_classify_intent_reads_the_question_structure(message, expected_intent):
    assert classify_intent(message) == expected_intent


@pytest.mark.parametrize("question,turn,expected", [
    # Q1: a LOOKUP answered from the project is project_rag even though a document
    # tool ran -- the DOC_OP branch must not claim it.
    ("What is the Time for Completion under the contract?",
     _turn(tools=["fetch_document"], sources=[{"source_class": "project_document"}]),
     ROUTE_PROJECT_RAG),
    # Q2: a LOOKUP answered from the project that also offers to compute stays
    # project_rag (not asks_for_inputs).
    ("What delay-damages rate and cap does the contract state?",
     _turn(sources=[{"source_class": "project_corpus"}],
           answer="Clause 8.7 ... tell me the Accepted Contract Amount and I'll compute it."),
     ROUTE_PROJECT_RAG),
    # Q17: a COMPUTE whose operand was never resolved is asks_for_inputs even when
    # the answer quotes project clauses (no calculator figure was produced).
    ("Calculate the delay damages.",
     _turn(sources=[{"source_class": "project_corpus"}],
           answer="Clause 8.7 sets 0.1%/day; give me the Accepted Contract Amount."),
     ROUTE_ASKS_INPUTS),
    # a COMPUTE whose calculator DID run is llm_calculator
    ("Concrete volume for a raft 30 x 20 x 1.5 m?",
     _turn(tools=["construction_calc"],
           provenance=[{"figure": "945 m3", "source": "calculator", "inputs": []}]),
     ROUTE_CALCULATOR),
    # a DOC_OPERATION is tool_on_document
    ("Summarise the named specification document.",
     _turn(sources=[{"source_class": "project_document"}]),
     ROUTE_TOOL_ON_DOC),
])
def test_classify_route_keys_on_intent_first(question, turn, expected):
    assert classify_route(turn, question) == expected


# ── R6: grounding gate (ORDERS_03NOV_0310Z/0420Z) ────────────────────────────


@pytest.mark.parametrize("expected,provenance,sources,is_grounded", [
    # a project route must stand on a project_document provenance
    (ROUTE_PROJECT_RAG, [{"source": "project_document"}], [], True),
    (ROUTE_PROJECT_RAG, [{"source": "general_knowledge"}], [], False),
    # general route -> general_knowledge
    (ROUTE_GENERAL_RAG, [{"source": "general_knowledge"}], [], True),
    (ROUTE_GENERAL_RAG, [{"source": "calculator"}], [], False),
    # calculator route -> a calculator figure
    (ROUTE_CALCULATOR, [{"source": "calculator"}], [], True),
    # asks_for_inputs may carry only the user's own figures (or none), never a
    # figure it got elsewhere
    (ROUTE_ASKS_INPUTS, [{"source": "user_input"}], [], True),
    (ROUTE_ASKS_INPUTS, [], [], True),
    (ROUTE_ASKS_INPUTS, [{"source": "project_document"}], [], False),
    # tool_on_document is evidenced by the sources panel (a summary/WBS need not
    # carry a figure)
    (ROUTE_TOOL_ON_DOC, [], ["project_corpus"], True),
    (ROUTE_TOOL_ON_DOC, [], ["knowledge_base"], False),
])
def test_grounded_requires_the_route_s_own_evidence(expected, provenance, sources, is_grounded):
    assert _grounded(expected, provenance, sources) is is_grounded


@pytest.mark.parametrize("events,exp_tools,exp_classes,exp_answer", [
    (
        [{"type": "tool_call", "tool": "construction_calc", "args_preview": "{}"},
         {"type": "tool_result"},
         {"type": "end", "content": "SAR 10,000/day",
          "sources": [{"source_class": "project_corpus"}],
          "provenance": [{"figure": "SAR 10,000/day", "source": "calculator",
                          "inputs": [{"name": "rate_percent", "value": 0.1,
                                      "unit": "%", "source": "project_data"}]}]}],
        ["construction_calc"], {"project_corpus"}, "SAR 10,000/day",
    ),
    (
        [{"type": "tool_call", "tool": "fetch_document"},
         {"type": "end", "content": "The spec says ...",
          "sources": [{"source_class": "project_document"}], "provenance": []}],
        ["fetch_document"], {"project_document"}, "The spec says ...",
    ),
])
def test_turn_from_events_collects_the_observable_signals(events, exp_tools, exp_classes, exp_answer):
    turn = turn_from_events(events)
    assert turn["tools"] == exp_tools
    assert {s["source_class"] for s in turn["sources"]} == exp_classes
    assert turn["answer"] == exp_answer


@pytest.mark.parametrize("project_name,expected_id", [
    # a name is given -> the project with that exact name, regardless of size
    ("Riyadh Tower", "p_named"),
    # no name -> the project with the most indexed documents
    (None, "p_big"),
])
def test_pick_project_by_name_else_most_indexed(project_name, expected_id):
    projects = [
        {"id": "p_named", "name": "Riyadh Tower", "indexed_documents": 3},
        {"id": "p_big", "name": "Other Site", "indexed_documents": 40},
        {"id": "p_small", "name": "Shed", "indexed_documents": 1},
    ]
    assert _pick_project(projects, project_name)["id"] == expected_id


@pytest.mark.parametrize("projects", [
    [],                    # nothing to choose from
    ["not-a-dict", 5],     # no usable (dict) rows after filtering
])
def test_pick_project_is_none_when_no_usable_rows(projects):
    assert _pick_project(projects, None) is None


@pytest.mark.parametrize("rows,expected_id", [
    # a general/master corpus with the MOST indexed docs must NOT be auto-picked;
    # the real document project wins even with far fewer docs
    ([{"id": "master_corpus", "name": "General knowledge", "indexed_documents": 500},
      {"id": "p_real", "name": "Riyadh Tower", "indexed_documents": 12}], "p_real"),
    # when ONLY general corpora are visible, the auto-pick returns None so the
    # caller fails loudly instead of grading the general corpus
    ([{"id": "master_corpus", "indexed_documents": 500},
      {"id": "knowledge_base", "indexed_documents": 9}], None),
])
def test_pick_project_never_auto_picks_the_general_corpus(rows, expected_id):
    got = _pick_project(rows, None)
    assert (got["id"] if got else None) == expected_id


@pytest.mark.parametrize("name,expected_id", [
    # a DELIBERATE project_name is honoured even when it names a general corpus
    ("General knowledge", "master_corpus"),
    # a different explicit name selects its own row, not the auto-pick
    ("Riyadh Tower", "p_real"),
])
def test_pick_project_honours_an_explicit_name(name, expected_id):
    rows = [{"id": "master_corpus", "name": "General knowledge", "indexed_documents": 500},
            {"id": "p_real", "name": "Riyadh Tower", "indexed_documents": 12}]
    assert _pick_project(rows, name)["id"] == expected_id


# ── the seed step's pure decisions (ORDERS_02NOV_1508Z/1545Z) ────────────────

from scripts.ui_battery_runner import (  # noqa: E402  (grouped with their tests)
    DEFAULT_BATTERY_PROJECT,
    _fixtures_needed,
    _seed_decision,
    _wait_until_indexed,
)


@pytest.mark.parametrize("projects,expected", [
    # the target already exists -> found, never duplicated (idempotent re-run)
    ([{"id": "p1", "name": DEFAULT_BATTERY_PROJECT, "indexed_documents": 2},
      {"id": "p2", "name": "Other", "indexed_documents": 9}], "found"),
    # no project of that name -> the seed creates it
    ([{"id": "p2", "name": "Other", "indexed_documents": 9}], "create"),
])
def test_seed_decision_finds_or_creates_never_duplicates(projects, expected):
    decision, row = _seed_decision(projects, DEFAULT_BATTERY_PROJECT)
    assert decision == expected
    assert (row is not None) == (expected == "found")


@pytest.mark.parametrize("row,expected", [
    ({"id": "p1", "indexed_documents": 0}, True),    # empty -> upload fixtures
    ({"id": "p1", "indexed_documents": 7}, False),   # already indexed -> skip
])
def test_fixtures_needed_only_when_nothing_is_indexed(row, expected):
    assert _fixtures_needed(row) is expected


@pytest.mark.parametrize("snapshots,should_raise", [
    # indexing lands on the third poll -> returns the row
    ([0, 0, 3], False),
    # never indexes -> a CLEAR error, not a silent pass against an empty project
    ([0, 0, 0], True),
])
def test_wait_until_indexed_polls_to_done_or_fails_loud(snapshots, should_raise):
    calls = {"n": 0}

    def fetch_rows():
        count = snapshots[min(calls["n"], len(snapshots) - 1)]
        calls["n"] += 1
        return [{"id": "p1", "name": DEFAULT_BATTERY_PROJECT, "indexed_documents": count}]

    if should_raise:
        with pytest.raises(RuntimeError) as err:
            _wait_until_indexed(fetch_rows, DEFAULT_BATTERY_PROJECT, len(snapshots), lambda: None)
        assert "did not reach an indexed state" in str(err.value)
        assert "ADMIN" in str(err.value)
    else:
        row = _wait_until_indexed(fetch_rows, DEFAULT_BATTERY_PROJECT, len(snapshots), lambda: None)
        assert row["indexed_documents"] == 3
