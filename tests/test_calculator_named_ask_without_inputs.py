"""A calculator the ask names but gives no figures for asks for them by label.

Live (S4): an ask about a reference-table calculator was answered "the
pre-dispatch run errored: missing required points" and then refused. The
formula pre-dispatch handed the model the tool's raw bind error as an
"authoritative draft" of a run that had "ALREADY been run", followed by "No
registry calculator matched this question", although one had. The calculator's
own question for the input, by its label, sat unused in the envelope. Asks that
wrote the calculator's display name ("<display name> for 72 points") were not
recognised as naming it at all.

Generated from the registry, not known strings:

* an ask that writes a calculator's display name (three words or more) names
  that calculator and is a calculator ask;
* the pre-dispatch of a named calculator with no figures tells the model the
  calculator was not run and gives it the calculator's question; it is never
  told that no calculator matched, never handed the bind error, and never told
  the calculator already ran;
* an answer that only narrates the failed step, or copies the tool's error,
  leaves as that calculator's question;
* the platform's pre-dispatch wording never reaches the user;
* a whole turn with no figures ends on the question by label.
"""
from __future__ import annotations

import asyncio
import re
from unittest.mock import patch

import pytest

from app.agents import answer_exit
from app.agents import runtime
from app.lib.construction_formulas import (
    calculator_name_from_text,
    run_calculation,
)
from app.lib.formula_registry import all_specs, parameters
from app.lib.source_labels import plain_registry_text
from tests._fixture_values import ENV_SECRET_01

_FRAMES = (
    "What is the {d}?",
    "Work out the {d} for this element.",
    "{D} please",
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _claimed_elsewhere(ask: str) -> bool:
    """An ask another rule owns on purpose (contract data, a deliverable,
    a document question) is not a calculator ask whatever it names."""
    from app.core.contract_lookup_intent import message_is_contract_data_lookup

    return bool(
        message_is_contract_data_lookup(ask)
        or runtime._message_is_schedule_or_programme_deliverable(ask)
        or runtime._document_question_without_operands(ask)
        or runtime._message_wants_inline_boq(ask)
    )


def _display_asks():
    rows = []
    for spec in all_specs():
        if len(_words(spec.display_name)) < 3:
            continue
        for frame in _FRAMES:
            ask = frame.format(d=spec.display_name.lower(), D=spec.display_name)
            if not _claimed_elsewhere(ask):
                rows.append((spec.name, ask))
    return rows


_DISPLAY_ASKS = _display_asks()


def _required(name: str) -> list[str]:
    spec = next(s for s in all_specs() if s.name == name)
    return [p.name for p in parameters(spec).values() if p.required]


def _named_without_inputs():
    rows = []
    seen = set()
    for name, ask in _DISPLAY_ASKS:
        if name in seen or not _required(name) or not ask.startswith("Work out"):
            continue
        env = run_calculation(name, {})
        if env.get("missing") and env.get("question"):
            seen.add(name)
            rows.append((name, ask))
    return rows


_NAMED_WITHOUT_INPUTS = _named_without_inputs()


def test_the_registry_gives_cases():
    assert len(_DISPLAY_ASKS) >= 100, len(_DISPLAY_ASKS)
    assert len(_NAMED_WITHOUT_INPUTS) >= 30, len(_NAMED_WITHOUT_INPUTS)


@pytest.mark.parametrize("name,ask", _DISPLAY_ASKS)
def test_display_name_names_its_calculator(name, ask):
    assert calculator_name_from_text(ask) == name, ask
    assert runtime._message_is_formula_style_ask(ask), ask
    assert runtime._message_names_unambiguous_calculator(ask), ask


def _agent():
    return runtime.Agent(name="project-assistant", description="t", system_prompt="t",
                         allowed_blocks=["construction"])


@pytest.mark.parametrize("name,ask", _NAMED_WITHOUT_INPUTS)
def test_predispatch_without_figures_hands_the_model_the_question(name, ask):
    messages = [{"role": "user", "content": ask}]
    pre = asyncio.run(runtime._predispatch_formula_calc(_agent(), messages, None, operator_text=ask))
    assert pre and pre["name"] == "construction_calc" and not pre["ok"], pre
    question = pre["result"]["question"]
    note = messages[-1]["content"]
    assert messages[-1]["role"] == "user"
    assert note.startswith(runtime._PREDISPATCH_PREFIX), note
    assert "was not run" in note and question in note, note
    for wrong in ("No registry calculator matched", "ALREADY been run", "Authoritative draft",
                  pre["result"]["error"], f" {name} ", f"{name}("):
        assert wrong not in note, (wrong, note)
    assert not any(m.get("role") == "tool" for m in messages), "a run that did not happen was recorded"


def _mechanics_replies(env: dict) -> list[str]:
    err = env["error"]
    return [
        "The pre-dispatch run errored, so I cannot answer this.",
        f"Predispatch returned: {err}",
        err,
        f"I could not run it. {plain_registry_text(err)}",
    ]


@pytest.mark.parametrize("name,ask", _NAMED_WITHOUT_INPUTS)
def test_an_answer_of_step_failure_leaves_as_the_question(name, ask):
    env = run_calculation(name, {})
    for reply in _mechanics_replies(env):
        turn = answer_exit.Turn()
        answer_exit.note_tool_result("construction_calc", env, turn)
        out = answer_exit.check_text_or_fallback(reply, turn=turn)
        expected = answer_exit.check_text(env["question"], turn=turn)
        assert expected in out, (reply, out)
        for leak in ("pre-dispatch", "predispatch", "Bad parameters", "missing required", name):
            assert leak.lower() not in out.lower(), (reply, out)


_SITE_TEXT = (
    "The dispatch note lists 4 trucks of ready-mix.",
    "Materials are dispatched from the batching plant at 06:00.",
    "The supplier's dispatch schedule allows 2 deliveries per hour.",
)


@pytest.mark.parametrize("form", ("pre-dispatch", "Pre-dispatch", "predispatch", "pre-dispatched",
                                  "PRE-DISPATCH"))
def test_pre_dispatch_wording_never_reaches_the_user(form):
    fact = "The register lists 4 open risks."
    out = answer_exit.check_text(f"The {form} step returned nothing useful. {fact}")
    assert form.lower() not in out.lower() and fact in out, out


@pytest.mark.parametrize("text", _SITE_TEXT)
def test_dispatch_in_site_text_is_left_as_written(text):
    assert answer_exit.check_text(text) == text


@pytest.mark.parametrize("name,ask", _NAMED_WITHOUT_INPUTS[::2])
def test_a_turn_with_no_figures_ends_on_the_question(monkeypatch, name, ask):
    """The whole stream, with a model that repeats what the platform told it."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", ENV_SECRET_01)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("SYNTHESIS_STREAMING", "0")
    question = run_calculation(name, {})["question"]

    async def call_llm(_self, messages, api_key, **kwargs):
        told = str(messages[-1].get("content") or "")
        if "No registry calculator matched" in told or "missing required" in told:
            reply = "The pre-dispatch run errored: missing required inputs, so I cannot answer."
        else:
            reply = "The pre-dispatch run did not complete, so I cannot answer."
        return {"status": "success", "choice": {"message": {"role": "assistant", "content": reply}}}

    async def drain():
        with patch.object(runtime.Agent, "_call_llm", call_llm):
            return [ev async for ev in _agent().chat_stream(
                user_message=ask, history=[], project_id=None, conversation_id=None, user_id=None)]

    end = asyncio.run(drain())[-1]
    assert end["type"] == "end", end
    content = (end.get("content") or "").strip()
    assert content.startswith(answer_exit.check_text(question)), content
    assert "pre-dispatch" not in content.lower() and name not in content, content
