"""S-T2: a question about what the documents say is answered from them.

A calculator runs only when the user supplies its inputs or asks for a
computation. A records question that merely mentions a calculator's subject
(which record fixes a lap length) or carries figures that only describe the
item (how many pads of a size are billed) must reach retrieval, never a
calculator asking the user for inputs.

Every case is a pair: the same subject with and without the records frame,
or the same records frame with and without the user's inputs / compute ask.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agents import runtime
from app.lib.construction_formulas import formula_inputs_supplied_by_text

CALC = {"construction_calc"}


def _forced(text: str) -> str | None:
    return runtime._forced_specific_tool([{"role": "user", "content": text}], CALC)


def _calc_ask(text: str) -> bool:
    return bool(
        runtime._message_is_formula_style_ask(text)
        or runtime._message_wants_named_calculator(text)
        or _forced(text)
    )


# --- 1. A record picked by its kind: the subject alone is no calculation ----

@pytest.mark.parametrize("subject_ask, records_ask", [
    ("Lap length for the column bars",
     "Which specification sets the lap length for the column bars?"),
    ("Lap length for the beam bars",
     "What drawing fixes the lap length for the beam bars?"),
])
def test_a_record_chosen_by_kind_is_a_documents_question(subject_ask, records_ask):
    assert _calc_ask(subject_ask)
    assert _forced(subject_ask) == "construction_calc"

    assert runtime._document_question(records_ask)
    assert runtime._document_question_without_operands(records_ask)
    assert not runtime._message_is_formula_style_ask(records_ask)
    assert not runtime._message_wants_named_calculator(records_ask)
    assert _forced(records_ask) is None


# --- 2. Figures that only describe the item a record is asked about ---------

@pytest.mark.parametrize("compute_ask, records_ask", [
    ("How much concrete is in a 4 m x 3 m x 0.5 m pad?",
     "How many 4 m x 3 m x 0.5 m pads are billed?"),
    ("How much concrete is in a 2.5 m x 2.5 m x 1.2 m chamber?",
     "How many 2.5 m x 2.5 m x 1.2 m chambers are scheduled?"),
])
def test_describing_figures_in_a_records_ask_are_no_operands(compute_ask, records_ask):
    assert runtime._looks_like_self_contained_calculation(compute_ask)
    assert _forced(compute_ask) == "construction_calc"

    assert not runtime._looks_like_self_contained_calculation(records_ask)
    assert runtime._figures_only_describe_record_ask(records_ask)
    assert runtime._document_question_without_operands(records_ask)
    assert not _calc_ask(records_ask)


# --- 3. The user's own request to compute keeps the calculator --------------

@pytest.mark.parametrize("records_ask, with_request", [
    ("How many 4 m x 3 m x 0.5 m pads are billed?",
     "How many 4 m x 3 m x 0.5 m pads are billed? Calculate the concrete in one."),
    ("How many 4 m x 3 m x 0.5 m pads are billed?",
     "How many 4 m x 3 m x 0.5 m pads are billed? Work out the concrete in one."),
    ("How many pads are billed?",
     "How many pads are billed at (4*3)*0.5 m3 each?"),
])
def test_a_compute_request_keeps_the_calculator(records_ask, with_request):
    assert not _calc_ask(records_ask)
    assert not runtime._figures_only_describe_record_ask(with_request)
    assert _forced(with_request) == "construction_calc"


# --- 4. Inputs the user wrote that complete the calculator keep it ----------

@pytest.mark.parametrize("without, with_inputs", [
    ("Which drawing shows the concrete volume for the slab?",
     "Which drawing shows the concrete volume for a slab 10 m long, 5 m wide and 0.2 m deep?"),
    ("Which specification gives the concrete volume for the raft?",
     "Which specification gives the concrete volume for a raft 30 m long, 20 m wide and 1.5 m deep?"),
])
def test_supplied_inputs_keep_the_calculator_inside_a_records_frame(without, with_inputs):
    assert runtime._document_question(without)
    assert runtime._document_question(with_inputs)
    assert not _calc_ask(without)
    assert not runtime._figures_only_describe_record_ask(with_inputs)
    assert _forced(with_inputs) == "construction_calc"


def test_formula_inputs_supplied_by_text_is_the_calculators_own_binding():
    full = "a slab 10 m long, 5 m wide and 0.2 m deep"
    assert formula_inputs_supplied_by_text("concrete_volume", full) is True
    assert formula_inputs_supplied_by_text("concrete_volume", "the slab") is False
    # A calculator that needs no input is not supplied by a user who gave none.
    assert formula_inputs_supplied_by_text("guardrail_top_rail_height",
                                           "guardrail top rail height") is False
    assert formula_inputs_supplied_by_text("not_a_registered_formula", full) is False
    assert formula_inputs_supplied_by_text(None, full) is False
    assert formula_inputs_supplied_by_text("concrete_volume", "  ") is False


# --- 5. The predispatch never runs a calculator on a records question -------

class _Agent:
    allowed_blocks = ("construction",)

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def _run_tool_call(self, tc: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(tc)
        return {"ok": True, "result": {"status": "needs_input", "needs_input": True,
                                       "missing": ["x"], "question": "What x?"}}


def _predispatch(text: str) -> tuple[Any, _Agent]:
    agent = _Agent()
    messages = [{"role": "user", "content": text}]
    out = asyncio.run(runtime._predispatch_formula_calc(agent, messages, "p1", operator_text=text))
    return out, agent


@pytest.mark.parametrize("calc_ask, records_ask", [
    ("Lap length for the column bars",
     "Which specification sets the lap length for the column bars?"),
    ("How much concrete is in a 4 m x 3 m x 0.5 m pad?",
     "How many 4 m x 3 m x 0.5 m pads are billed?"),
])
def test_predispatch_leaves_a_records_question_to_retrieval(monkeypatch, calc_ask, records_ask):
    monkeypatch.setenv("AGENT_FORMULA_PREDISPATCH", "1")
    out, agent = _predispatch(calc_ask)
    assert out is not None and out["name"] == "construction_calc"
    assert len(agent.calls) == 1

    out, agent = _predispatch(records_ask)
    assert out is None
    assert agent.calls == []
    # So the loop's missing-input short-circuit has nothing to answer with.
    assert runtime._calc_input_question(out) == ""


# --- 6. Each records frame on its own ---------------------------------------

@pytest.mark.parametrize("text, frame", [
    ("which drawing fixes the cover", "_RECORD_CHOICE_RE"),
    ("what contract specification volume governs the works", "_RECORD_CHOICE_RE"),
    ("how many are billed", "_RECORD_ACT_RE"),
    ("what grade is specified", "_RECORD_ACT_RE"),
    ("how many were itemised", "_RECORD_ACT_RE"),
])
def test_each_records_frame_matches(text, frame):
    assert getattr(runtime, frame).search(text)
    assert runtime._document_question(text)


@pytest.mark.parametrize("text", [
    "which is the larger bar",          # no record named
    "what is the lap length",           # a quantity, not a record
    "how many are required",            # not something a record does
    "how many bars are needed",
])
def test_no_records_frame_without_a_record(text):
    assert not runtime._RECORD_CHOICE_RE.search(text)
    assert not runtime._RECORD_ACT_RE.search(text)


# --- 7. A calculator that lacks inputs ends the turn only on a compute ask ---
# The recogniser above can miss a records question. Whatever elected the
# calculator, a missing-input answer closes the turn only when the user asked
# for a computation; otherwise the turn goes on to retrieval.

@pytest.mark.parametrize("compute_ask, plain_ask", [
    ("Calculate the lap length for the column bars",
     "Lap length for the column bars"),
    ("Work out the lap length for the column bars",
     "Lap length for the column bars"),
    ("Lap length for the column bars: (40*16) mm",
     "Lap length for the column bars"),
])
def test_missing_inputs_ask_only_when_a_computation_was_asked(monkeypatch, compute_ask, plain_ask):
    monkeypatch.setenv("AGENT_FORMULA_PREDISPATCH", "1")
    out, agent = _predispatch(compute_ask)
    assert out is not None and len(agent.calls) == 1
    assert out["falls_to_retrieval"] is False
    assert runtime._calc_input_question(out) == "What x?"

    out, agent = _predispatch(plain_ask)
    assert out is not None and len(agent.calls) == 1
    assert out["falls_to_retrieval"] is True
    # The turn does not end with the calculator's question ...
    assert runtime._calc_input_question(out) == ""


class _AskingAgent(_Agent):
    """As the real tool wrapper reports it: an ask is not ``ok``."""

    async def _run_tool_call(self, tc: dict[str, Any]) -> dict[str, Any]:
        out = await super()._run_tool_call(tc)
        return {**out, "ok": False}


def test_the_model_is_sent_to_the_documents_with_the_note_alongside(monkeypatch):
    monkeypatch.setenv("AGENT_FORMULA_PREDISPATCH", "1")
    for text, to_docs in (("Calculate the lap length for the column bars", False),
                          ("Lap length for the column bars", True)):
        agent = _AskingAgent()
        messages = [{"role": "user", "content": text}]
        asyncio.run(runtime._predispatch_formula_calc(agent, messages, "p1", operator_text=text))
        note = messages[-1]["content"]
        assert ("from the project documents" in note) is to_docs
        assert ("ask the user" in note.lower()) is not to_docs
        # ... and the missing-input note rides along either way.
        assert "What x?" in note


@pytest.mark.parametrize("result, falls", [
    ({"status": "needs_input", "needs_input": True, "missing": ["x"], "question": "q"}, True),
    # A value the user wrote and the calculator rejected is still asked about.
    ({"status": "error", "needs_input": True, "missing": [], "question": "q",
      "rejected": [{"name": "x", "reason": "range"}]}, False),
    ({"status": "needs_input", "needs_input": True, "missing": ["x"], "question": "q",
      "rejected": [{"name": "y", "reason": "unreadable"}]}, False),
    # Nothing missing: nothing to fall through on.
    ({"status": "needs_input", "needs_input": True, "missing": [], "question": "q"}, False),
    ({"status": "success", "calculation": "x"}, False),
    (None, False),
])
def test_only_a_pure_missing_input_falls_to_retrieval(result, falls):
    plain = "Lap length for the column bars"
    assert runtime._missing_inputs_fall_to_retrieval(result, plain) is falls
    # The same result on a compute request never falls through.
    assert runtime._missing_inputs_fall_to_retrieval(result, "Calculate the " + plain) is False


@pytest.mark.parametrize("text, computes", [
    ("Calculate the lap length", True),
    ("compute the lap length", True),
    ("estimate the lap length", True),
    ("Lap length is (40*16) mm", True),
    ("What is the volume of a raft 30 m x 20 m?", True),
    ("Lap length for the column bars", False),
    ("How many manholes", False),
    ("", False),
])
def test_message_requests_computation(text, computes):
    assert runtime._message_requests_computation(text) is computes
