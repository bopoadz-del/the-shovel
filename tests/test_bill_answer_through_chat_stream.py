"""#411 (c)/(d): a computed bill answer through the REAL stream route.

Every test POSTs one turn to ``/v1/chat/stream`` (the route and its whole
wrapper chain: ``with_answer_exit``, ``with_hat_signals``,
``guarantee_terminal``, ``redact_frames``) and reads every SSE frame the
user's browser would read. Only the chat block's token source is replaced:
it records the bill tool's real result on the turn (as the agent loop does)
and then streams a model reply. The bill, the ask and the reply are built
here; the expected figure is the sum of the rows built here.

What is asserted on the frames before ``end``:
* the tool's statement is the first answer text once the tool has spoken,
  each figure-stating call in call order, an identical statement once;
* the model's own tokens after that are held: no pre-end frame states a
  figure the tool did not;
* ``end`` leads with the same statements and carries the model's text whole
  or withholds it whole.
"""
from __future__ import annotations

import json
import uuid

import pytest

from tests._fixture_values import PW_06
from tests.test_bill_sum_selection_from_ask import (
    ASK,
    _bill_payload,
    _invented_bill,
)


def _cited(text):
    """``text`` as the end frame carries it: the exit contract turns a
    ``(source: ...)`` into a numbered citation marker, nothing else."""
    import re

    return re.sub(r" \(source: [^)]*\)| \[\d+\]", "", text)

pytestmark = pytest.mark.asyncio


def _scripted(steps):
    """The turn's answer source below the route, scripted: ``steps`` is a
    list of ("tool", payload) and ("say", text) run in order; a "say" is
    streamed word by word as token frames; an ``end`` frame closes it."""

    async def stream(*, prompt, **_kw):
        from app.agents import answer_exit

        for kind, value in steps:
            if kind == "tool":
                turn = answer_exit.current_turn()
                turn.question = prompt
                answer_exit.note_tool_result("bill_rows_aggregate", value, turn)
            else:
                for word in value.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'sources': []})}\n\n"

    return stream


def _turn(monkeypatch, steps, message=ASK):
    """POST one turn to /v1/chat/stream; returns (pre-end token texts, end)."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.routers import chat as chat_router

    from app.routers.exit_frames import with_answer_exit

    # The route under test answers through the exit wrapper under test.
    assert chat_router.with_answer_exit is with_answer_exit
    monkeypatch.setattr(chat_router, "_predispatch_concurrent_stream", _scripted(steps))
    with TestClient(app) as c:
        email = f"bill-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        r = c.post("/v1/chat/stream", headers={"Authorization": f"Bearer {token}"},
                   json={"message": message, "history": []})
    frames = []
    for line in r.text.splitlines():
        if line.startswith("data:"):
            try:
                frames.append(json.loads(line[5:]))
            except ValueError:
                continue
    return _split(frames)


def _split(frames):
    ends = [i for i, f in enumerate(frames) if f.get("type") == "end"]
    assert len(ends) == 1, [f.get("type") for f in frames]
    pre = frames[:ends[0]]
    return [f["content"] for f in pre if f.get("type") == "token"], frames[ends[0]]


def _leads_for(payloads):
    from app.agents import answer_exit

    turn = answer_exit.Turn()
    turn.question = ASK
    for p in payloads:
        answer_exit.note_tool_result("bill_rows_aggregate", p, turn)
    return answer_exit.computed_leads(turn)


_ROWS_A = (633, 488, 828, 957, 844, 221, 552, 200, 135)
_ROWS_B = (1250, 740, 310)


@pytest.mark.parametrize("quantities, off", [
    (_ROWS_A, 1000),
    (_ROWS_B, 10),
    (_ROWS_A, 0),
    (_ROWS_B, 0),
], ids=["a-foreign-total", "b-foreign-total", "a-right", "b-right"])
async def test_the_tools_statement_is_emitted_first_and_the_model_is_held(
        monkeypatch, tmp_path, quantities, off):
    from app.agents import answer_exit

    payload = _bill_payload(_invented_bill(tmp_path, quantities), ASK)
    right = sum(quantities)
    reply = f"The clay pipe in the runs comes to {right + off:,} m."
    tokens, end = _turn(monkeypatch, [("say", "Reading the bill."), ("tool", payload), ("say", reply)])
    (lead,) = _leads_for([payload])
    assert f"{right:,}" in lead
    # Before the tool spoke the model's words streamed; the first frame after
    # is the tool's statement, and nothing after it streams.
    assert "".join(tokens[:-1]).strip() == "Reading the bill."
    assert tokens[-1] == lead + "\n\n"
    shown = "".join(tokens)
    assert shown.count(lead) == 1
    if off:
        assert f"{right + off:,}" not in shown
    # The end frame leads with the statement; the model's text whole or not at all.
    content = _cited(end["content"])
    assert content.startswith(_cited(lead))
    assert content.count(_cited(lead)) == 1
    assert (reply.strip() in content) is (not off)
    assert (answer_exit.FIGURE_REMOVED in content) is bool(off)


@pytest.mark.parametrize("first_quantities, same", [
    (_ROWS_B, False),
    (_ROWS_A, True),
], ids=["two-different-calls", "the-same-call-twice"])
async def test_every_figure_stating_call_leads_in_call_order_once(
        monkeypatch, tmp_path, first_quantities, same):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    first = _bill_payload(_invented_bill(tmp_path / "1", first_quantities), ASK)
    second = _bill_payload(_invented_bill(tmp_path / "2", _ROWS_A), ASK)
    tokens, end = _turn(monkeypatch, [("tool", first), ("say", "Two reads."), ("tool", second),
                                      ("say", f"Total {sum(_ROWS_A):,} m.")])
    leads = _leads_for([first, second])
    assert len(leads) == (1 if same else 2)
    assert tokens == [lead + "\n\n" for lead in leads]
    content = _cited(end["content"])
    assert content.startswith("\n\n".join(_cited(lead) for lead in leads))
    for lead in leads:
        assert content.count(_cited(lead)) == 1


@pytest.mark.parametrize("reply", [
    "The drainage section lists nine runs of clay pipe and a fence.",
    "The bill has 12 sections and 64 m was mentioned in the spec.",
])
async def test_a_turn_with_no_computed_result_streams_unchanged(monkeypatch, reply):
    from app.agents import answer_exit

    tokens, end = _turn(monkeypatch, [("say", reply)], message="What does the bill hold?")
    assert "".join(tokens).strip() == reply
    assert answer_exit.FIGURE_REMOVED not in (end.get("content") or "")
