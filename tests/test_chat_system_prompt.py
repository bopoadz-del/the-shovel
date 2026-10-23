"""ChatBlock — optional system prompt (literal + file) tests.

Covers the two new params introduced for prompt injection without
touching the user message:

  - ``system_prompt`` — literal string used as the system role verbatim
  - ``system_prompt_file`` — filename loaded from ``app/prompts/``

Failure modes (missing file, path traversal) must NOT abort the chat —
they fall through with no system prompt applied.

Default behavior with neither param must be byte-for-byte unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.blocks.chat import ChatBlock
from tests._fixture_values import STUB_TOKEN_05


# ── Helpers ───────────────────────────────────────────────────────────────


def _install_cloud_stub(monkeypatch, captured: dict):
    """Replace ChatBlock._call_cloud with an async stub that captures the
    system_prompt kwarg + the messages list it would have sent. Returns
    success so process() short-circuits cleanly."""

    async def fake_call(
        self,
        message,
        model,
        max_tokens,
        temperature,
        stream,
        key,
        cfg=None,
        system_prompt=None,
    ):
        captured["called"] = True
        captured["message"] = message
        captured["system_prompt"] = system_prompt
        captured["messages"] = self._build_messages(message, system_prompt)
        return {
            "status": "success",
            "text": "ok",
            "provider": "deepseek",
            "model": model,
        }

    monkeypatch.setattr(ChatBlock, "_call_cloud", fake_call)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_05)


# ── 1. Literal system_prompt ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_literal_system_prompt_prepends_system_message(monkeypatch):
    """A literal system_prompt string lands as a system role message
    immediately before the user message, and the user message is left
    completely untouched."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    cb = ChatBlock()
    result = await cb.process(
        {"text": "what is concrete cover?"},
        {"system_prompt": "You are an EVM expert. Be concise."},
    )

    assert result["status"] == "success"
    assert captured["system_prompt"] == "You are an EVM expert. Be concise."
    # User message is verbatim — system prompt did NOT mutate it.
    assert captured["message"] == "what is concrete cover?"
    msgs = captured["messages"]
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "You are an EVM expert. Be concise."}
    assert msgs[1] == {"role": "user", "content": "what is concrete cover?"}


@pytest.mark.asyncio
async def test_literal_via_input_data_dict(monkeypatch):
    """Either input_data OR params can carry the new params, matching the
    convention used by use_rag / use_local_model."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    cb = ChatBlock()
    await cb.process(
        {"text": "hello", "system_prompt": "from input_data"},
        {},
    )
    assert captured["system_prompt"] == "from input_data"


# ── 2. system_prompt_file loads from app/prompts/ ─────────────────────────


@pytest.mark.asyncio
async def test_system_prompt_file_loads_construction_evm(monkeypatch):
    """system_prompt_file='construction_evm.md' loads the real file from
    app/prompts/ and passes its content through as the system role."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    cb = ChatBlock()
    await cb.process(
        {"text": "estimate this trade"},
        {"system_prompt_file": "construction_evm.md"},
    )

    sp = captured["system_prompt"]
    assert sp is not None, "construction_evm.md should have been loaded"
    # The file is a real markdown file; check a stable distinctive line.
    assert "CEREBRUM CONSTRUCTION AI" in sp
    # And it must have been prepended as a system message.
    msgs = captured["messages"]
    assert msgs[0]["role"] == "system"
    assert "CEREBRUM CONSTRUCTION AI" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "estimate this trade"}


@pytest.mark.asyncio
async def test_literal_wins_over_file_when_both_supplied(monkeypatch):
    """Precedence: literal system_prompt wins over system_prompt_file."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    cb = ChatBlock()
    await cb.process(
        {"text": "hello"},
        {
            "system_prompt": "literal wins",
            "system_prompt_file": "construction_evm.md",
        },
    )
    assert captured["system_prompt"] == "literal wins"


# ── 3. Path-traversal is rejected ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_path_traversal_rejected(monkeypatch):
    """system_prompt_file='../etc/passwd' must be rejected. No system
    prompt is applied, but the chat call still succeeds."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    cb = ChatBlock()
    result = await cb.process(
        {"text": "hello"},
        {"system_prompt_file": "../etc/passwd"},
    )
    assert result["status"] == "success"
    assert captured["system_prompt"] is None
    assert captured["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_absolute_path_rejected(monkeypatch, tmp_path):
    """Absolute paths must NOT escape app/prompts/, even if the target
    file exists and is readable."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    bait = tmp_path / "evil.md"
    bait.write_text("SHOULD NOT LOAD")

    cb = ChatBlock()
    result = await cb.process(
        {"text": "hi"},
        {"system_prompt_file": str(bait)},
    )
    assert result["status"] == "success"
    assert captured["system_prompt"] is None


# ── 4. Missing file falls through quietly ─────────────────────────────────


@pytest.mark.asyncio
async def test_missing_file_falls_through(monkeypatch):
    """Pointing at a non-existent file in app/prompts/ logs a warning and
    continues with NO system prompt — chat must not go dark."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    # Sanity: confirm the file really doesn't exist so the test is honest.
    prompts_dir = Path(__file__).resolve().parent.parent / "app" / "prompts"
    assert not (prompts_dir / "nonexistent.md").exists()

    cb = ChatBlock()
    result = await cb.process(
        {"text": "hi"},
        {"system_prompt_file": "nonexistent.md"},
    )
    assert result["status"] == "success"
    assert captured["system_prompt"] is None
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


# ── 5. Default behavior: generic ChatBlock does NOT inject domain prompts ─


@pytest.mark.asyncio
async def test_default_does_not_inject_construction_expert(monkeypatch):
    """Generic ChatBlock must not auto-inject construction_expert.txt (fa76e0a).
    Domain prompts are container-owned — ConstructionContainer.chat() sets
    construction_evm.md when the construction kit handles chat."""
    captured: dict = {}
    _install_cloud_stub(monkeypatch, captured)

    cb = ChatBlock()
    result = await cb.process({"text": "hello"})
    assert result["status"] == "success"
    assert captured["system_prompt"] is None
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
