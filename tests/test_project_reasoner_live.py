"""LIVE LLM end-to-end test — Reasoning Engine Plan 5.

Off by default — even with a cloud key in .env, this test stays skipped
unless LIVE_LLM_TESTS=1 is explicitly set. Two-key gate prevents a routine
`pytest` run from silently burning the LLM credit.
"""

import os

import pytest

from app.blocks.project_reasoner import ProjectReasonerBlock
from app.core.session_store import InMemorySessionStore

pytestmark = pytest.mark.skipif(
    not ((os.getenv("KIMI_API_KEY") or os.getenv("GROQ_API_KEY")) and os.getenv("LIVE_LLM_TESTS") == "1"),
    reason="live LLM tests off — set LIVE_LLM_TESTS=1 (plus KIMI_API_KEY or GROQ_API_KEY) to arm",
)


@pytest.mark.asyncio
async def test_live_reasoner_answers_critical_path_question():
    session = await InMemorySessionStore().get_or_create("live1")
    session.data["activities"] = [
        {"id": "A", "duration": 3, "predecessors": []},
        {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
        {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
    ]
    block = ProjectReasonerBlock()
    out = await block.process({
        "request": "What is the project duration and the critical path?",
        "session": session,
    })
    assert out["status"] == "success"
    assert "10" in out["answer"]


@pytest.mark.asyncio
async def test_live_reasoner_follow_up_uses_prior_state():
    session = await InMemorySessionStore().get_or_create("live2")
    session.data["activities"] = [
        {"id": "A", "duration": 3, "predecessors": []},
        {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
        {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
    ]
    block = ProjectReasonerBlock()
    await block.process({"request": "Compute the critical path.",
                         "session": session})
    out = await block.process({
        "request": "Now shorten B by 3 days — what is the new duration?",
        "session": session,
    })
    assert out["status"] == "success"
    assert "7" in out["answer"]
