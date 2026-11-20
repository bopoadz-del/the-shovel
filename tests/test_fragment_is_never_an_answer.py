"""A model reply is withheld only on POSITIVE evidence that it is not the
answer (GRADER 1250Z on #449): the platform's stages cut more than
``CUT_SHARE`` of it, or the completion did not end normally (a length limit,
a tool call left pending, a stream cut without a reason) and is under
``SHORT_REPLY_CHARS``. Never on its words, length or punctuation alone: a
whole short reply that ended normally ("Yes", "No", "Grade B", one word, a
short Arabic answer) is shown as it is.

What replaces a withheld reply: the computed result's statement when the
turn has one; otherwise "the reply was incomplete; please ask again" in the
question's language -- never a not-found sentence.

Every turn goes through the real HTTP route (/v1/chat/stream, driver mode)
with the model stubbed at its last hop (``llm_client.provider_stream``), so
the provider's finish reason travels the real parsing. Invented input only:
a synthetic formula, invented words and figures; two inputs per decision.
"""
from __future__ import annotations

import json
import logging
import re

import pytest

from app.agents.core import computed_answer
from app.agents.driver import loop, render
from app.core import llm_client
from app.lib import construction_formulas as cf
from app.lib import formula_registry as fr
from tests._dev_key import DEV_KEY

NAME = "syn_plank_area"
EN = computed_answer.LABELS["en"]["incomplete"]
AR = computed_answer.LABELS["ar"]["incomplete"]


@pytest.fixture(autouse=True)
def synthetic_formula():
    def syn_plank_area(plank_length_m: float, plank_width_m: float):
        return {"plank_area_m2": plank_length_m * plank_width_m}
    fr._REGISTRY.pop(NAME, None)
    fr.formula(owner="base", description="Area of a synthetic plank.", display_name="Synthetic plank area",
               inputs={"plank_length_m": fr.Param(unit="m", lo=0.01, hi=50, label="plank length"),
                       "plank_width_m": fr.Param(unit="m", lo=0.01, hi=50, label="plank width")},
               outputs={"plank_area_m2": "m2"})(syn_plank_area)
    cf.CALCULATORS[NAME] = syn_plank_area
    render._MAP.clear()
    yield
    fr._REGISTRY.pop(NAME, None)
    cf.CALCULATORS.pop(NAME, None)
    render._MAP.clear()


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _sse(choice):
    return "data: " + json.dumps({"choices": [choice]})


def text_reply(content, finish="stop"):
    """A provider completion of ``content`` in small deltas, ending with
    ``finish`` (None: the stream ends without the provider saying why)."""
    lines = [_sse({"delta": {"content": content[i:i + 5]}}) for i in range(0, len(content), 5)]
    if finish is not None:
        lines.append(_sse({"delta": {}, "finish_reason": finish}))
    return lines


def calc_reply(length, width):
    args = json.dumps({"calculation": NAME, "params": {"plank_length_m": length, "plank_width_m": width}})
    return [_sse({"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                            "function": {"name": "construction_calc", "arguments": args}}]}}),
            _sse({"delta": {}, "finish_reason": "tool_calls"})]


def _turn(monkeypatch, user, replies):
    """One turn through POST /v1/chat/stream; the provider answers each model
    call with the next scripted SSE completion. Returns (answer text, the
    DRIVER_TURN trace)."""
    from fastapi.testclient import TestClient

    from app.main import app
    from tests.test_driver_mode import _no_retrieval

    _no_retrieval(monkeypatch)
    monkeypatch.setenv("DRIVER_MODE", "request")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "invented-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("USAGE_DAILY_CAP_USD", raising=False)
    replies = list(replies)

    async def provider_stream(url, payload, headers, *, timeout=None, provider="", model="",
                              user_id=None, agent_name=""):
        for line in replies.pop(0):
            yield line
        yield "data: [DONE]"

    monkeypatch.setattr(llm_client, "provider_stream", provider_stream)
    log = logging.getLogger(loop.__name__)
    monkeypatch.setattr(log, "disabled", False)
    seen = _Lines()
    log.addHandler(seen)
    log.setLevel(logging.INFO)
    try:
        with TestClient(app) as client:
            body = client.post("/v1/chat/stream", json={"message": user, "history": []},
                               headers={"Authorization": "Bearer " + DEV_KEY, "X-Fork-Driver": "on"}).text
    finally:
        log.removeHandler(seen)
    events = [json.loads(line[len("data: "):]) for line in body.split("\n") if line.startswith("data: ")]
    assert any(e.get("type") == "start" and e.get("mode") == "driver" for e in events), body[:400]
    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    line = next(entry for entry in seen.lines if entry.startswith("DRIVER_TURN "))
    return text, json.loads(line[len("DRIVER_TURN "):])


# ── the rule itself ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("written,kept,finish,why", [
    (30, 3, "stop", "stage_cut"), (120, 50, "stop", "stage_cut"),
    (3, 3, "length", "unfinished"), (12, 12, "content_filter", "unfinished"),
    (3, 3, "tool_calls", "unfinished"),
    (3, 3, "stop", ""), (2, 2, "stop", ""),
    # no finish reason from the provider: normal, unless nothing was written
    (2, 2, "unreported", ""), (12, 12, "", ""), (0, 0, "unreported", "unfinished"), (0, 0, "", "unfinished"),
    (60, 60, "length", ""), (90, 90, "tool_calls", ""),
])
def test_only_positive_evidence_withholds(written, kept, finish, why):
    assert computed_answer.not_the_answer(written, kept, finish) == why


# ── a cut completion ────────────────────────────────────────────────────────

@pytest.mark.parametrize("length,width,area", [(2, 3, "6 m"), (4, 5, "20 m")])
def test_a_cut_completion_with_a_computed_result_shows_the_tools_statement(monkeypatch, length, width, area):
    text, trace = _turn(monkeypatch, f"Plank {length} m by {width} m, what area?",
                        [calc_reply(length, width), text_reply("QRZ", finish="length")])
    assert "QRZ" not in text
    assert area in text
    assert EN not in text
    assert trace["withheld"] == {"reason": "unfinished", "finish": "length", "written": 3, "kept": 3}
    assert "QRZ" not in json.dumps(trace)  # lengths and reasons only, never the text


@pytest.mark.parametrize("user,stub,finish,expected", [
    ("What do the plank rows add up to?", "QRZ", "length", EN),
    ("ما مجموع بنود الألواح؟", "Vexl", "content_filter", AR),
])
def test_a_cut_completion_without_a_result_says_the_reply_was_incomplete(monkeypatch, user, stub,
                                                                         finish, expected):
    text, trace = _turn(monkeypatch, user, [text_reply(stub, finish=finish)])
    assert stub not in text
    assert expected in text
    assert "could not find" not in text.lower()  # never "not found": no search found nothing
    assert trace["withheld"]["reason"] == "unfinished"
    assert trace["steps"][0]["finish"] == finish
    assert trace["steps"][0]["reply_chars"] == len(stub)


@pytest.mark.parametrize("figure", ["7,391", "15,207"])
def test_an_answer_a_stage_cut_down_is_not_shown(monkeypatch, figure):
    # A stage that removes every figure (standing in for any stage that cuts
    # too much) leaves "Plank total" of a longer answer: the stage lengths
    # show which stage cut it, and the reply is withheld though it ended
    # normally.
    monkeypatch.setattr(loop, "_tidy", lambda text: re.sub(r"\s*[\d.,]{3,}[^.]*", "", text).strip())
    answer = f"Plank {figure} m2 is the sum of the plank rows on every sheet."
    text, trace = _turn(monkeypatch, "What do the plank rows add up to?", [text_reply(answer)])
    assert "Plank" not in text
    assert EN in text
    stages = trace["stage_chars"]
    # the lengths name the stage that cut it: not the citation stage
    assert stages["citation_gate"] > stages["figure_provenance"] or stages["figure_provenance"] > stages["tidy"]
    assert trace["withheld"]["reason"] == "stage_cut"


# ── what the citation stage removes is not a cut ────────────────────────────

@pytest.mark.parametrize("stages,kept", [
    # the citation stage alone removed most of it: all is kept
    ([("model", 34), ("readable", 34), ("citation_gate", 7), ("figure_provenance", 7), ("tidy", 7)], 34),
    ([("model", 20), ("citation_gate", 4), ("tidy", 4)], 20),
    # another stage's removal counts; a lengthening stage adds nothing
    ([("model", 30), ("readable", 40), ("citation_gate", 40), ("figure_provenance", 10)], 0),
    ([("model", 30), ("citation_gate", 20), ("tidy", 12)], 22),
    ([], 0),
])
def test_only_non_citation_stages_count_as_cut(stages, kept):
    assert computed_answer.kept_after_stages(stages) == kept


@pytest.mark.parametrize("cited,shown", [
    ("Grade B\nSource: plank_log.pdf p.4", "Grade B"),
    ("Larch\nSource: timber_schedule.xlsx sheet 2", "Larch"),
])
def test_a_short_answer_whose_citation_a_stage_strips_is_shown(monkeypatch, cited, shown):
    # The turn read nothing, so the citation gate strips the citation, which
    # is more than half the reply. That is a citation removed, not a cut.
    text, trace = _turn(monkeypatch, "What grade is the plank timber?", [text_reply(cited)])
    assert text.strip().splitlines()[0] == shown  # then the platform's own removed-attribution note
    assert EN not in text
    assert trace["withheld"] == {}
    stages = trace["stage_chars"]
    assert stages["answer_exit"] > 2 * stages["citation_gate"]  # the citation stage removed most of it


@pytest.mark.parametrize("user,reply", [
    ("How many planks are in the log?", "88"),
    ("How many plank rows are on sheet two?", "14"),
])
def test_a_bare_number_with_a_normal_stop_is_not_withheld_as_cut(monkeypatch, user, reply):
    text, trace = _turn(monkeypatch, user, [text_reply(reply)])
    assert trace["withheld"] == {}
    assert text.strip() == reply


@pytest.mark.parametrize("user,stub,expected", [
    ("What do the plank rows add up to?", "Let", EN),
    ("ما مجموع بنود الألواح؟", "دع", AR),
])
def test_a_tool_call_finish_with_no_call_to_run_is_incomplete(monkeypatch, user, stub, expected):
    text, trace = _turn(monkeypatch, user, [text_reply(stub, finish="tool_calls")])
    assert stub not in text
    assert expected in text
    assert trace["withheld"]["finish"] == "tool_calls"


@pytest.mark.parametrize("user,reply", [
    ("Which material are the planks?", "Larch"),
    ("هل سجل الألواح معتمد؟", "نعم"),
])
def test_a_reply_whose_provider_gave_no_finish_reason_is_shown(monkeypatch, user, reply):
    text, trace = _turn(monkeypatch, user, [text_reply(reply, finish=None)])
    assert text.strip() == reply
    assert trace["withheld"] == {}
    assert trace["steps"][0]["finish"] == "unreported"


@pytest.mark.parametrize("user,expected", [
    ("What do the plank rows add up to?", EN),
    ("ما مجموع بنود الألواح؟", AR),
])
def test_an_empty_reply_with_no_finish_reason_says_incomplete(monkeypatch, user, expected):
    text, trace = _turn(monkeypatch, user, [text_reply("", finish=None)])
    assert expected in text


# ── LEFT ALONE: a whole short reply that ended normally ─────────────────────

@pytest.mark.parametrize("user,reply", [
    ("Is the plank log approved?", "Yes"),
    ("Is the plank log signed?", "No"),
    ("Was the plank submittal accepted?", "Yes, approved"),
    ("What grade is the plank timber?", "Grade B"),
    ("Which material are the planks?", "Larch"),
    ("Which finish do the planks take?", "Varnish"),
    ("هل سجل الألواح معتمد؟", "نعم معتمد"),
    ("ما مادة الألواح؟", "خشب الصنوبر"),
])
def test_a_whole_short_reply_with_a_normal_stop_is_shown_as_it_is(monkeypatch, user, reply):
    text, trace = _turn(monkeypatch, user, [text_reply(reply)])
    assert text.strip() == reply
    assert EN not in text and AR not in text
    assert trace["withheld"] == {}


@pytest.mark.parametrize("reply", [
    "Plank total\n\nThe planks add up to the area stated above.",
    "## Plank area\nEach plank is listed below, row by row.",
])
def test_a_long_reply_streams_whole(monkeypatch, reply):
    text, trace = _turn(monkeypatch, "What do the plank rows add up to?", [text_reply(reply)])
    assert text.strip() == reply.strip()
    assert trace["withheld"] == {}


@pytest.mark.parametrize("reply", [
    "Plank rows total forty-two square metres in all\n\nEach row is listed below.",
    "Planks were counted on every floor of the block\n\nThe count is in the table.",
])
def test_a_paragraph_break_after_a_shown_line_reaches_the_user(monkeypatch, reply):
    # The first line is long enough to be shown at once; the blank line after
    # it is layout and is streamed as it is, between the two paragraphs.
    text, trace = _turn(monkeypatch, "What do the plank rows add up to?", [text_reply(reply)])
    assert text.strip() == reply
    assert trace["withheld"] == {}


@pytest.mark.parametrize("shared,stages", [
    (True, ["model", "readable", "answer_exit", "citation_gate"]),
    (False, ["model", "readable", "citation_gate"]),
])
def test_exit_check_names_each_stage_the_text_passed(shared, stages):
    msgs = [{"role": "user", "content": "What is the plank area?"}]
    _, _, report = loop.exit_check("The plank area is stated in the log.", msgs, shared=shared)
    assert [name for name, _ in report["stages"]][:len(stages)] == stages


# ── the model call: no configured key, no provider call ─────────────────────

class _Agent:
    name, model, temperature, max_tokens = "invented-agent", "", 0.2, 64


def _provider_key(monkeypatch, configured):
    from app.agents.runtime import _llm_config

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    env_key = _llm_config()["env_key"]
    if configured:
        monkeypatch.setenv(env_key, "invented-key")
    else:
        monkeypatch.delenv(env_key, raising=False)


@pytest.mark.parametrize("configured,status", [(False, "error"), (True, "success")])
def test_call_reaches_the_provider_only_with_a_key(monkeypatch, configured, status):
    import asyncio

    from app.agents.driver import llm

    reached = []

    class _Resp:
        status_code = 200

    async def provider_post(url, payload, headers, **kw):
        reached.append(url)
        return _Resp(), {"choices": [{"message": {"content": "Larch"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(llm_client, "provider_post", provider_post)
    _provider_key(monkeypatch, configured)
    out = asyncio.run(llm.call(_Agent(), [{"role": "user", "content": "Which wood?"}], None))
    assert out["status"] == status
    assert bool(reached) == configured


@pytest.mark.parametrize("configured,status", [(False, "error"), (True, "success")])
def test_stream_reaches_the_provider_only_with_a_key(monkeypatch, configured, status):
    import asyncio

    from app.agents.driver import llm

    reached = []

    async def provider_stream(url, payload, headers, **kw):
        reached.append(url)
        for line in text_reply("Larch"):
            yield line

    monkeypatch.setattr(llm_client, "provider_stream", provider_stream)
    _provider_key(monkeypatch, configured)

    async def run():
        return [item async for item in llm.stream(_Agent(), [{"role": "user", "content": "Which wood?"}], None)]

    done = [value for kind, value in asyncio.run(run()) if kind == "done"]
    assert len(done) == 1 and done[0]["status"] == status
    assert bool(reached) == configured
