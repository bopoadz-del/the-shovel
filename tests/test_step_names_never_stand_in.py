"""A step name never stands in a sentence the user reads, and a turn never
ends empty because of it.

Live (S3 UI re-check, two agents, first ask):

* A removed tool or step name was replaced by "a platform step", so the
  sentence kept a placeholder where its subject or object had been ("I cannot
  report a platform step (IFC2X3 / IFC4 / IFC4.3)", "fall back to a platform
  step for a take-off"), and a code span whose value was removed showed two
  ticks.
* Tool talk with no tool name got through: retry narration ("Retrying the
  extractors once more."), asking the user for formula IDs or a project id,
  and a payload field with a word value ("confidence tier: BOQ").
* One agent answered "I was unable to generate a response for this turn": the
  model's reply was only about the platform's steps, the exit check took every
  sentence out, and nothing asked the model again. Another ended on its own
  retry narration, which no check recognised as unfinished. A calculator ask
  with no usable reply ended on "The answer was cut off" because the note that
  the calculator had NOT run was read as a run.

Generated from the registries and grammar frames, not known strings.
"""
from __future__ import annotations

import asyncio
import re
from unittest.mock import patch

import pytest

from app.agents import answer_exit
from app.agents import runtime
from app.lib.construction_formulas import run_calculation
from app.lib.formula_registry import all_specs, parameters
from app.lib.source_labels import _registry_maps
from tests._fixture_values import ENV_SECRET_01

FACT = "The slab is 250 mm thick."
TOTAL = "the BOQ total is 1,250,000 AED."


def _step_names() -> list[str]:
    displays, params, _units = _registry_maps()
    names = sorted(
        n for n in answer_exit._static_step_names()
        if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", n)
        and answer_exit._is_word_ident(n) and n not in displays and n not in params
    )
    return names[::max(1, len(names) // 120)]


STEPS = _step_names()

#: (frame with the step, what the user reads). A frame whose sentence is about
#: the step goes whole; one whose step sits in a phrase loses the phrase; a
#: step that did the finding is said as what was found.
FRAMES = (
    ("I cannot report the {s} (IFC2X3 / IFC4 / IFC4.3). " + FACT, FACT),
    ("If the take-off fails, fall back to {s} for a take-off. " + FACT, FACT),
    ("The {s} returned `` on two attempts. " + FACT, FACT),
    ("Please share the {s} summary for the relevant drawings. " + FACT, FACT),
    ("The `{s}` exists for this. " + FACT, FACT),
    ("Using the {s}, " + TOTAL, TOTAL[:1].upper() + TOTAL[1:]),
    (FACT + " The total, from the {s} block, is 1,250,000 AED.", FACT + " The total is 1,250,000 AED."),
    ("The {s} found 42 line items totalling 1,250,000 AED.", "I found 42 line items totalling 1,250,000 AED."),
    ("The {s} tool reports 12 doors on level 2.", "I found 12 doors on level 2."),
)


def _words(name: str) -> str:
    return name.replace("_", " ")


def test_the_registries_give_cases():
    assert len(STEPS) >= 60, len(STEPS)


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("frame,expected", FRAMES)
def test_a_step_name_leaves_without_a_placeholder(step, frame, expected):
    out = answer_exit.check_text(frame.format(s=step), turn=answer_exit.Turn())
    assert out == expected, (step, out)
    for leak in ("platform step", "``", step, _words(step)):
        assert leak not in out, (leak, out)


@pytest.mark.parametrize("phrase", ("tool calls", "tool results", "function outputs", "tool responses"))
def test_tool_call_phrases_leave_the_same_way(phrase):
    out = answer_exit.check_text(f"The {phrase} came back empty. {FACT}", turn=answer_exit.Turn())
    assert out == FACT, out
    out = answer_exit.check_text(f"Based on the {phrase}, {TOTAL}", turn=answer_exit.Turn())
    assert out == TOTAL[:1].upper() + TOTAL[1:], out


# -- tool talk without a tool name -------------------------------------------

_RETRIES = (
    "Retrying the {n} once more.",
    "Re-running the {n} now.",
    "Trying again with the {n}.",
    "I'll retry the {n}.",
    "Let me re-run the {n}.",
    "Checking the {n} one more time.",
)
_NOUNS = ("extractors", "drawing take-off", "BOQ read", "schedule parse")


@pytest.mark.parametrize("noun", _NOUNS)
@pytest.mark.parametrize("retry", _RETRIES)
def test_retry_narration_never_reaches_the_user(retry, noun):
    out = answer_exit.check_text(f"{FACT} {retry.format(n=noun)}", turn=answer_exit.Turn())
    assert out == FACT, out


_ASKS = ("Please provide the {n} IDs so I can continue.", "Which {n} ID should I use?",
         "Share the {n} identifiers and I will look them up.")


@pytest.mark.parametrize("noun", ("formula", "tool", "block", "operation", "action", "chunk",
                                  "calculator", "project", "document", "file"))
@pytest.mark.parametrize("ask", _ASKS)
def test_the_user_is_never_asked_for_an_internal_id(ask, noun):
    out = answer_exit.check_text(f"{FACT} {ask.format(n=noun)}", turn=answer_exit.Turn())
    assert out == FACT, out


@pytest.mark.parametrize("noun", ("project", "document", "file"))
def test_an_id_a_document_shows_is_left_as_written(noun):
    text = f"The {noun} ID column on the transmittal is blank for 3 rows."
    assert answer_exit.check_text(text, turn=answer_exit.Turn()) == text


def _field_keys() -> list[str]:
    displays, params, _units = _registry_maps()
    steps = answer_exit._static_step_names()
    keys = [f"{a}_{b}" for a in ("confidence", "promotion", "source", "match", "review")
            for b in ("tier", "state", "kind", "mode", "flag")]
    return [k for k in keys if k not in displays and k not in params and k not in steps]


@pytest.mark.parametrize("key", _field_keys())
@pytest.mark.parametrize("value", ("BOQ", "high", "pending", "Tier3"))
def test_a_payload_field_with_a_word_value_goes(key, value):
    turn = answer_exit.Turn()
    assert answer_exit.check_text(f"{FACT} {key}: {value}.", turn=turn) == FACT
    out = answer_exit.check_text(f"- {key}: {value}\n- slab thickness: 250 mm", turn=turn)
    assert out == "- slab thickness: 250 mm", out


def _registered_params() -> list[str]:
    _displays, params, _units = _registry_maps()
    return sorted(p for p in params if "_" in p)[::40]


@pytest.mark.parametrize("param", _registered_params())
def test_a_registered_field_with_a_figure_stays(param):
    out = answer_exit.check_text(f"- {param}: 72", turn=answer_exit.Turn())
    assert out.endswith(": 72") and param not in out, out


def test_an_error_figure_is_a_figure_not_a_tool_error():
    text = "Correction recorded:\n- predicted: 450 SAR/m3\n- actual: 480 SAR/m3\n- error: 6.7%"
    assert answer_exit.check_text(text, turn=answer_exit.Turn()) == text
    assert "not found" not in answer_exit.check_text("Error: file not found.", turn=answer_exit.Turn())


def test_a_data_fence_with_figures_is_read_not_dropped():
    out = answer_exit.check_text("```\nsample_count: 3\nerror_pct: 6.7\n```", turn=answer_exit.Turn())
    assert "```" not in out and "3" in out and "6.7" in out, out


# -- a turn never ends empty because of it -------------------------------------

def _internal_only() -> list[str]:
    rows = [frame.format(s=step).replace(" " + FACT, "").replace(FACT + " ", "")
            for step in STEPS[::6] for frame, expected in FRAMES if expected == FACT]
    rows += [retry.format(n="extractors") for retry in _RETRIES]
    rows += [ask.format(n=noun) for ask in _ASKS for noun in ("formula", "project")]
    return rows


@pytest.mark.parametrize("text", _internal_only())
def test_an_answer_the_exit_would_empty_is_asked_again(text):
    assert answer_exit.check_text(text, turn=answer_exit.Turn()).strip() == "", text
    assert runtime._final_text_needs_forced_retry(text, user_message="q") is True
    assert runtime._forced_retry_nudge(text) in (runtime._USER_TERMS_RETRY_NUDGE,
                                                 runtime._SEARCH_PREAMBLE_RETRY_NUDGE)


_STATEMENTS = ("Both files are registered under exactly those names.",
               "The drawing and the BOQ are both in the project.",
               "I have the schedule.")
_NARRATIONS = ("Retrying the extractors once more.", "Re-running the take-off.",
               "Trying again with the BOQ.", "Reading the next window.", "Calling the parser again.")


@pytest.mark.parametrize("narration", _NARRATIONS)
@pytest.mark.parametrize("statement", _STATEMENTS)
def test_a_short_reply_that_ends_on_more_work_is_unfinished(statement, narration):
    text = f"{statement} {narration}"
    assert runtime._looks_like_search_preamble(text) is True, text
    assert runtime._final_text_needs_forced_retry(text, user_message="q") is True


@pytest.mark.parametrize("text", (
    FACT,
    "I found 42 line items totalling 1,250,000 AED.",
    "The rate is 0.1%.\nReading the cap now.\nThe cap is 10%.",
    "Delay Damages are 0.1% of the Contract Price per calendar day. Let me know if you need more.",
    "Pouring resumes at 06:00 once the formwork is checked.",
))
def test_a_real_answer_is_not_asked_again(text):
    assert runtime._final_text_needs_forced_retry(text, user_message="q") is False, text


def _calculators_asking() -> list[tuple[str, str, str]]:
    rows = []
    for spec in all_specs():
        if len(re.findall(r"[a-z0-9]+", spec.display_name.lower())) < 3:
            continue
        if not any(p.required for p in parameters(spec).values()):
            continue
        env = run_calculation(spec.name, {})
        if env.get("missing") and env.get("question"):
            rows.append((spec.name, spec.display_name, env["question"]))
    return rows


CALCS = _calculators_asking()


def test_calculators_give_cases():
    assert len(CALCS) >= 30, len(CALCS)


@pytest.mark.parametrize("name,display,question", CALCS)
def test_the_empty_turn_reply_becomes_the_tools_question(name, display, question):
    turn = answer_exit.Turn()
    answer_exit.note_tool_result("construction_calc", run_calculation(name, {}), turn)
    out = answer_exit.check_text_or_fallback(runtime._EMPTY_RESPONSE_FALLBACK, turn=turn)
    assert out == answer_exit.check_text(question, turn=turn), out


@pytest.mark.parametrize("name,display,question", CALCS)
def test_a_calculator_that_did_not_run_is_never_called_cut_off(name, display, question):
    env = run_calculation(name, {})
    messages = [{"role": "user", "content": f"Calculate the {display.lower()}."},
                {"role": "user", "content": runtime._formula_not_run_bubble(name, {"result": env})}]
    turn, token = answer_exit.begin_turn()
    try:
        answer_exit.note_tool_result("construction_calc", env, turn)
        out = runtime._nonblank_after_empty_synthesis("", messages)
    finally:
        answer_exit.reset_turn(token)
    assert runtime._SYNTH_CUTOFF_NOTICE not in out and question in out, out


def _agent(name="project-assistant", blocks=("construction",)):
    return runtime.Agent(name=name, description="t", system_prompt="t", allowed_blocks=list(blocks))


def _stream(monkeypatch, ask, replies, agent=None):
    monkeypatch.setenv("DEEPSEEK_API_KEY", ENV_SECRET_01)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("SYNTHESIS_STREAMING", "0")
    told: list[str] = []

    async def call_llm(_self, messages, api_key, **kwargs):
        told.append(str(messages[-1].get("content") or ""))
        reply = replies[min(len(told), len(replies)) - 1]
        return {"status": "success", "choice": {"message": {"role": "assistant", "content": reply}}}

    async def drain():
        with patch.object(runtime.Agent, "_call_llm", call_llm):
            return [ev async for ev in (agent or _agent()).chat_stream(
                user_message=ask, history=[], project_id=None, conversation_id=None, user_id=None)]

    return asyncio.run(drain())[-1], told


@pytest.mark.parametrize("name,display,question", CALCS[::3])
def test_a_calculator_turn_with_no_usable_reply_asks_by_label(monkeypatch, name, display, question):
    end, _told = _stream(monkeypatch, f"Calculate the {display.lower()}.", [""])
    content = (end.get("content") or "").strip()
    assert end["type"] == "end" and content.startswith(answer_exit.check_text(question)), content
    assert runtime._SYNTH_CUTOFF_NOTICE not in content, content


@pytest.mark.parametrize("text", _internal_only()[::4])
def test_a_reply_only_about_steps_is_asked_again_in_user_terms(monkeypatch, text):
    end, told = _stream(monkeypatch, "What has been recorded so far?", [text, FACT])
    assert (end.get("content") or "").strip() == FACT, end.get("content")
    assert any(runtime._USER_TERMS_RETRY_NUDGE == t or runtime._SEARCH_PREAMBLE_RETRY_NUDGE == t
               for t in told), told


@pytest.mark.parametrize("statement", _STATEMENTS)
def test_a_reply_that_ends_on_its_retry_is_not_the_answer(monkeypatch, statement):
    end, _told = _stream(monkeypatch, "Compare the BOQ with the drawing quantities.",
                         [f"{statement} Retrying the extractors once more.", FACT])
    assert (end.get("content") or "").strip() == FACT, end.get("content")
