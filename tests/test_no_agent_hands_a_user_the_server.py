"""An agent is only as safe as the blocks it may call.

`code` and `sandbox` were admin-only at /v1/execute and /v1/chain -- but the
chat accepted ANY agent name a browser sent (`"agent": "self-coding"`), the
runtime ran that agent's blocks without asking who the caller was, and
/v1/agents/{name}/chat did the same. A plain user could therefore reach, by
naming an agent, the very blocks they were refused by name:

  * self-coding     -> code, sandbox      (runs the Python it is handed)
  * external-mcp    -> mcp_consumer       (starts the command it is handed)
  * document-ingestion -> local_drive     (reads/lists/WRITES the data disk,
                                           which holds every user's uploads)

and `web` / `webhook` (any URL, from inside the network) were not admin-only
anywhere. The gate now lives in the runtime, where every door ends, and the
doors that name an agent refuse early.
"""
import asyncio
import json
import pathlib
import re
import uuid

import pytest
from tests._fixture_values import PW_06

from app.core.privileges import (
    PRIVILEGED_BLOCKS,
    agent_is_privileged,
    caller_may_use_agent,
    caller_may_use_block,
    set_caller_role,
)

BLOCKS_DIR = pathlib.Path(__file__).resolve().parent.parent / "app" / "blocks"

# A block that starts a process is privileged, or it is listed here with the
# reason its command cannot come from a user. (None today: drawing_qto's DWG
# converter, the one reviewed entry, is removed.)
_REVIEWED_PROCESS_BLOCKS: dict[str, str] = {}
_STARTS_A_PROCESS_RE = re.compile(
    r"subprocess\.(?:run|Popen|call|check_output|check_call)|os\.system\(|"
    r"os\.popen\(|StdioServerParameters\(|create_subprocess_(?:exec|shell)\("
)
# A new block that opens the network must be admin-only, or listed here
# with why the destination is not a caller-chosen host (same shape as
# _REVIEWED_PROCESS_BLOCKS). Caller-URL downloaders are listed as a known
# gap — they are not privileged yet; a *new* file with httpx/requests
# still fails CI until someone puts it in PRIVILEGED_BLOCKS or here.
_REVIEWED_NETWORK_BLOCKS = {
    "chat": "LLM provider URL from env/config, not a caller-chosen host",
    "project_reasoner": "same LLM hop as chat",
    "formula_executor_v2": "same LLM hop as chat",
    "translate": "fixed Google translate endpoint; query is text, not a URL",
    "search": "fixed Serper/DDGS endpoints; query is text, not a URL",
    "ocr": "downloads a caller http(s) URL then local Tesseract — SSRF-class, not privileged yet",
    "ocr_v2": "same caller-URL download as ocr",
    "pdf": "same caller-URL download as ocr",
    "pdf_v2": "same caller-URL download as ocr",
    "image": "same caller-URL download as ocr",
}
_REACHES_THE_NETWORK_RE = re.compile(
    r"\b(?:httpx|requests|aiohttp|urllib3)\.|"
    r"urllib\.request|"
    r"AsyncOpenAI|OpenAI\(|"
    r"\bwebsockets\.|\bftplib\.|\bsmtplib\."
)


@pytest.fixture(autouse=True)
def _agents_loaded_and_role_reset():
    from app.agents.runtime import load_agents

    load_agents()
    yield
    set_caller_role(None)


# ── the rule ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("block", sorted(PRIVILEGED_BLOCKS))
def test_a_privileged_block_is_refused_to_everyone_but_an_admin(block):
    assert caller_may_use_block(block, "admin")
    assert not caller_may_use_block(block, "user")
    # An unknown caller is not an admin: fail closed.
    set_caller_role(None)
    assert not caller_may_use_block(block)


@pytest.mark.parametrize("block", ["mcp_consumer", "local_drive", "web", "webhook",
                                   "google_drive", "onedrive"])
def test_the_blocks_that_reach_past_the_app_are_privileged(block):
    assert block in PRIVILEGED_BLOCKS


def test_an_ordinary_block_stays_open():
    assert caller_may_use_block("construction", "user")
    assert caller_may_use_block("formula_executor_v2", "user")


def test_every_block_that_starts_a_process_is_privileged_or_reviewed():
    offenders = []
    for path in sorted(BLOCKS_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        m = re.search(r'^\s+name\s*=\s*"([^"]+)"', src, re.M)
        if not m or not _STARTS_A_PROCESS_RE.search(src):
            continue
        name = m.group(1)
        if name not in PRIVILEGED_BLOCKS and name not in _REVIEWED_PROCESS_BLOCKS:
            offenders.append(f"{path.name} ({name})")
    assert not offenders, (
        "these blocks start a process but are not admin-only: " + ", ".join(offenders)
    )


def test_every_block_that_reaches_the_network_is_privileged_or_reviewed():
    offenders = []
    for path in sorted(BLOCKS_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        m = re.search(r'^\s+name\s*=\s*"([^"]+)"', src, re.M)
        if not m or not _REACHES_THE_NETWORK_RE.search(src):
            continue
        name = m.group(1)
        if name not in PRIVILEGED_BLOCKS and name not in _REVIEWED_NETWORK_BLOCKS:
            offenders.append(f"{path.name} ({name})")
    assert not offenders, (
        "these blocks reach the network but are not admin-only: "
        + ", ".join(offenders)
    )


def test_every_agent_holding_a_privileged_block_is_refused_to_a_user():
    from app.agents import AGENT_REGISTRY

    held = {n: a for n, a in AGENT_REGISTRY.items() if agent_is_privileged(a)}
    # The three that were reachable. If one stops holding a privileged block
    # the rule still holds, but this pins what the fix was about.
    for name in ("self-coding", "external-mcp", "document-ingestion"):
        assert name in held, name
    for name, agent in held.items():
        assert not caller_may_use_agent(agent, "user"), name
        assert caller_may_use_agent(agent, "admin"), name
    # The user-facing specialists are untouched.
    for name in ("project-assistant", "quantity-surveyor", "contracts-manager",
                 "construction-pm", "safety-officer"):
        if name in AGENT_REGISTRY:
            assert caller_may_use_agent(AGENT_REGISTRY[name], "user"), name


# ── the runtime: where every door ends ──────────────────────────────────────

def _tool_call(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def test_the_runtime_refuses_a_privileged_block_to_a_user(monkeypatch):
    from app.agents import AGENT_REGISTRY
    from app.blocks import BLOCK_REGISTRY

    ran = []

    async def _process(self, input_data, params=None):
        ran.append(input_data)
        return {"status": "success", "output": "ran"}

    from app.dependencies import block_instances

    monkeypatch.setattr(BLOCK_REGISTRY["code"], "process", _process)
    # Startup creates every block instance; do the same for this one.
    monkeypatch.setitem(block_instances, "code", BLOCK_REGISTRY["code"]())
    agent = AGENT_REGISTRY["self-coding"]
    call = _tool_call("code", {"input": {"code": "print(1)"}})

    set_caller_role("user")
    out = asyncio.run(agent._run_tool_call(call))
    assert out["ok"] is False
    assert "admin-only" in out["result"]["error"]
    assert ran == []

    set_caller_role("admin")
    out = asyncio.run(agent._run_tool_call(call))
    assert ran, "an admin must still reach the block"


def test_a_tool_the_caller_would_be_refused_is_not_offered():
    """The auto-router still hands unmatched arithmetic to self-coding, for
    anyone. A plain user's turn there must see formula_executor_v2 and not
    `code`, or the model spends a round on a refusal."""
    from app.agents import AGENT_REGISTRY
    from app.blocks import BLOCK_REGISTRY

    agent = AGENT_REGISTRY["self-coding"]
    set_caller_role("user")
    offered = {t["function"]["name"] for t in agent.tool_definitions()}
    assert not offered & PRIVILEGED_BLOCKS, offered
    # Offered wherever it is loaded; a virgin boot (no domain kit) has none.
    if "formula_executor_v2" in BLOCK_REGISTRY:
        assert "formula_executor_v2" in offered
    set_caller_role("admin")
    offered = {t["function"]["name"] for t in agent.tool_definitions()}
    assert "code" in offered


def test_a_user_cannot_delegate_to_a_privileged_agent(monkeypatch):
    from app.agents import AGENT_REGISTRY

    delegating = next(a for a in AGENT_REGISTRY.values() if a.can_delegate)
    target = AGENT_REGISTRY["self-coding"]
    reached = []

    async def _chat(*a, **k):
        reached.append(True)
        return {"status": "success", "response": "x"}

    monkeypatch.setattr(target, "chat", _chat)
    set_caller_role("user")
    out = asyncio.run(delegating._run_tool_call(
        _tool_call("delegate_to_agent", {"agent_name": "self-coding", "message": "x"})))
    assert out["ok"] is False
    assert reached == []
    assert "self-coding" not in out["result"]["hint"]


# ── the doors ───────────────────────────────────────────────────────────────

@pytest.fixture
def user_client():
    # One user per test: the suite empties the database between tests, so a
    # module-wide user is gone by the second one ("Token user no longer exists").
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        email = f"pin-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post(
            "/v1/users/login", json={"email": email, "password": PW_06}
        ).json()["token"]
        yield c, {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("block", ["mcp_consumer", "local_drive", "web", "webhook",
                                   "google_drive", "onedrive"])
def test_execute_refuses_the_newly_privileged_blocks(user_client, block):
    from app.blocks import BLOCK_REGISTRY

    if block not in BLOCK_REGISTRY:
        # Not loaded in this build: /v1/execute answers 404 before any gate.
        # The rule above still covers it the day it is loaded.
        assert block in PRIVILEGED_BLOCKS
        return
    c, user = user_client
    r = c.post("/v1/execute", headers=user,
               json={"block": block, "input": {}, "params": {"operation": "list"}})
    assert r.status_code == 403, (r.status_code, r.text)


def test_the_agent_picker_does_not_offer_privileged_agents(user_client):
    c, user = user_client
    body = c.get("/v1/agents", headers=user).json()
    names = {a["name"] for a in body["agents"]}
    assert not names & {"self-coding", "external-mcp", "document-ingestion"}, names
    assert "project-assistant" in names
    assert body["count"] == len(body["agents"])


@pytest.mark.parametrize("path,method", [
    ("/v1/agents/self-coding", "get"),
    ("/v1/agents/self-coding/chat", "post"),
    ("/v1/agents/external-mcp/chat/stream", "post"),
    ("/v1/agents/document-ingestion/chat", "post"),
])
def test_naming_a_privileged_agent_in_the_url_is_refused(user_client, path, method):
    c, user = user_client
    r = getattr(c, method)(path, headers=user, **({"json": {"message": "hi"}} if method == "post" else {}))
    assert r.status_code == 404, (r.status_code, r.text)


@pytest.mark.parametrize("agent", ["self-coding", "external-mcp", "document-ingestion"])
def test_pinning_a_privileged_agent_in_the_chat_is_refused(user_client, agent):
    c, user = user_client
    r = c.post("/v1/chat/stream", headers=user,
               json={"message": "list the files", "agent": agent})
    events = [json.loads(line[5:]) for line in r.text.splitlines()
              if line.startswith("data:") and line[5:].strip().startswith("{")]
    assert not any(e.get("type") == "route" and e.get("final") == agent for e in events), events
    errors = [e for e in events if e.get("type") == "error"]
    assert errors and "not available" in errors[0]["message"], events
