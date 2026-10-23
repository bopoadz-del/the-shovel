"""The user's turn must be persisted even when the LLM call fails mid-loop,
so the conversation history never silently drops the question."""

from app.agents.runtime import Agent, AGENT_REGISTRY, load_agents
from tests._fixture_values import STUB_TOKEN_04


async def test_user_turn_persisted_when_llm_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)

    from app.core import agent_memory
    agent_memory.init_db()
    load_agents()
    agent = AGENT_REGISTRY["smart-orchestrator"]

    async def failing_llm(self, messages, api_key, project_id=None, with_tools=True, **kwargs):
        return {"status": "error", "error": "simulated LLM outage"}

    monkeypatch.setattr(Agent, "_call_llm", failing_llm)

    conv = "conv-persist-on-error"
    result = await agent.chat("remember this question", conversation_id=conv)

    assert result["status"] == "error"  # the chat surfaced the LLM failure
    # ...but the user's question was still saved to the conversation.
    roles = [(m["role"], m["content"]) for m in agent_memory.get_messages(conv)]
    assert ("user", "remember this question") in roles


async def test_user_turn_persisted_once_on_success(tmp_path, monkeypatch):
    """On a normal turn the user message is stored exactly once (no double
    write now that persistence moved up front)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)

    from app.core import agent_memory
    agent_memory.init_db()
    load_agents()
    agent = AGENT_REGISTRY["smart-orchestrator"]

    async def ok_llm(self, messages, api_key, project_id=None, with_tools=True, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": "the answer", "tool_calls": []}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", ok_llm)

    conv = "conv-persist-once"
    await agent.chat("my question", conversation_id=conv)

    roles = [(m["role"], m["content"]) for m in agent_memory.get_messages(conv)]
    assert roles.count(("user", "my question")) == 1
    assert ("assistant", "the answer") in roles
