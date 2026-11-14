"""M2: a multi-turn reply must not be bound to an earlier question.

The chat UI sends the full thread as ``history`` on every turn, and the server
loads the same thread back from the conversation store. Concatenating both
duplicates the thread; under the ``MAX_HISTORY_TURNS`` cap that replays an
older question right before the current one, so the model answers a previous
turn. The stored conversation is authoritative; the client's replay of it must
not double it.
"""
from __future__ import annotations

import importlib

import pytest

from app.agents.runtime import Agent


def _make_agent(name: str = "m2-agent") -> Agent:
    return Agent(
        name=name,
        description="Test agent for unit tests",
        system_prompt="You are a test agent.",
        allowed_blocks=[],
        can_delegate=False,
    )


def _isolate(tmp_path, monkeypatch):
    """Point DATA_DIR at tmp and reload agent_memory so no test touches the
    real DB. Returns the fresh agent_memory module."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.core.agent_memory as _am
    import app.core.projects as _proj
    import app.core.project_memory as _pm
    importlib.reload(_proj)
    importlib.reload(_pm)
    importlib.reload(_am)
    return _am


def _llm_capture(box, text: str = "ok"):
    """A scripted ``_call_llm`` that snapshots the messages it received."""
    async def fake(self, messages, api_key, project_id=None, **kwargs):
        box["messages"] = [dict(m) for m in messages]
        return {
            "status": "success",
            "choice": {"message": {"content": text}},
            "raw": {},
        }
    return fake


QA_PAIRS = [
    # Two unrelated two-turn threads: distinct questions and answers.
    [
        ("who is the Engineer", "The Engineer is Jacobs"),
        ("what is the Time for Completion", "852 days"),
    ],
    [
        ("what is the retention percentage", "10 percent"),
        ("what is the defect notification period", "365 days"),
    ],
]
CURRENTS = ["what is the performance bond", "what is the advance payment"]


@pytest.mark.asyncio
@pytest.mark.parametrize("qa_pairs,current", zip(QA_PAIRS, CURRENTS))
async def test_stored_thread_is_not_duplicated_by_the_client_replay(
    tmp_path, monkeypatch, qa_pairs, current
):
    am = _isolate(tmp_path, monkeypatch)
    am.get_or_create_conversation("cv-dup", "m2-agent")
    for q, a in qa_pairs:
        am.append_message("cv-dup", "user", q)
        am.append_message("cv-dup", "assistant", a)

    client_history = [
        {"role": role, "content": content}
        for q, a in qa_pairs
        for role, content in (("user", q), ("assistant", a))
    ]

    box: dict = {}
    monkeypatch.setattr(Agent, "_call_llm", _llm_capture(box))
    agent = _make_agent()
    await agent.chat(current, api_key="k", conversation_id="cv-dup", history=client_history)

    contents = [
        m["content"] for m in box["messages"] if m["role"] in ("user", "assistant")
    ]
    for q, a in qa_pairs:
        assert contents.count(q) == 1, contents
        assert contents.count(a) == 1, contents
    # The current turn is the last user-visible message in the prompt.
    assert contents[-1] == current, contents


@pytest.mark.asyncio
@pytest.mark.parametrize("qa_pairs,current", zip(QA_PAIRS, CURRENTS))
async def test_streamed_thread_is_not_duplicated_by_the_client_replay(
    tmp_path, monkeypatch, qa_pairs, current
):
    am = _isolate(tmp_path, monkeypatch)
    am.get_or_create_conversation("cv-dup-s", "m2-agent")
    for q, a in qa_pairs:
        am.append_message("cv-dup-s", "user", q)
        am.append_message("cv-dup-s", "assistant", a)

    client_history = [
        {"role": role, "content": content}
        for q, a in qa_pairs
        for role, content in (("user", q), ("assistant", a))
    ]

    box: dict = {}
    monkeypatch.setattr(Agent, "_call_llm", _llm_capture(box))
    agent = _make_agent()
    async for _ev in agent.chat_stream(
        current, api_key="k", conversation_id="cv-dup-s", history=client_history
    ):
        pass

    contents = [
        m["content"] for m in box["messages"] if m["role"] in ("user", "assistant")
    ]
    for q, a in qa_pairs:
        assert contents.count(q) == 1, contents
        assert contents.count(a) == 1, contents
    assert contents[-1] == current, contents
