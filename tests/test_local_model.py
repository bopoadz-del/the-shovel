"""Tests for the PR 3a scaffolding — local model loader + chat block opt-in.

What's covered here:
  - local_model.available() reports correctly on missing deps, missing
    adapter dir, missing adapter_config.json, and the LOCAL_MODEL_UNAVAILABLE
    override
  - generate() returns None when unavailable (gating works)
  - chat block's use_local_model flag falls back to cloud when local stack
    is unavailable
  - chat block's use_local_model flag uses the local generation when it
    returns text
  - chat block falls back when generate() returns None mid-flight
  - export_training_set.py: agent_memory pair extraction, seed file
    reading, filter thresholds

What's NOT covered (because it needs real torch + a real model):
  - The actual _build_pipeline() — model + adapter loading
  - generate() against a real model — output quality
  - The finetune_router.py training loop itself
  These are verified by running on a GPU host, not in CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests._fixture_values import STUB_TOKEN_05


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Standard isolation pattern + reset local_model's cache."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le_state.json"))
    monkeypatch.setenv("LEARNING_ENGINE_ENABLED", "1")
    from app.core.learning import local_model as _lm
    _lm.invalidate_cache()
    from app.core import agent_memory as _am
    if hasattr(_am, "_initialized"):
        _am._initialized = False
    yield tmp_path
    _lm.invalidate_cache()


# ── local_model.available() — the gate ────────────────────────────────────


def test_available_false_when_override_set(isolated_data_dir, monkeypatch):
    """LOCAL_MODEL_UNAVAILABLE=1 short-circuits the check. Critical for
    tests + operators forcing the cloud path."""
    from app.core.learning import local_model as _lm
    monkeypatch.setenv("LOCAL_MODEL_UNAVAILABLE", "1")
    assert _lm.available() is False


def test_available_false_when_no_adapter_dir(isolated_data_dir, monkeypatch):
    """No adapter dir = nothing to load even if torch/peft are installed."""
    from app.core.learning import local_model as _lm
    monkeypatch.setenv("LOCAL_ADAPTER_DIR", str(isolated_data_dir / "nonexistent"))
    assert _lm.available() is False


def test_available_false_when_no_adapter_config(isolated_data_dir, monkeypatch):
    """Adapter dir exists but no adapter_config.json — incomplete state."""
    from app.core.learning import local_model as _lm
    adapter_dir = isolated_data_dir / "fake_adapter"
    adapter_dir.mkdir()
    # adapter_model.safetensors but NO adapter_config.json
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
    monkeypatch.setenv("LOCAL_ADAPTER_DIR", str(adapter_dir))
    assert _lm.available() is False


def test_get_pipeline_returns_none_when_unavailable(isolated_data_dir, monkeypatch):
    """The "gate then build" pattern. Callers can rely on get_pipeline()
    returning None to mean "fall back to cloud" rather than catching
    exceptions."""
    from app.core.learning import local_model as _lm
    monkeypatch.setenv("LOCAL_MODEL_UNAVAILABLE", "1")
    assert _lm.get_pipeline() is None


def test_generate_returns_none_when_unavailable(isolated_data_dir, monkeypatch):
    from app.core.learning import local_model as _lm
    monkeypatch.setenv("LOCAL_MODEL_UNAVAILABLE", "1")
    assert _lm.generate("anything") is None


def test_format_prompt_matches_training_template():
    """Single source of truth: the inference wrap MUST equal the trainer's
    wrap. Both finetune_router._format_prompt and local_model.format_prompt
    delegate to the same function — if anyone redefines either in a way
    that drifts, this test fails. Reviewer-flagged bug on PR #24."""
    from app.core.learning.local_model import format_prompt
    from scripts.finetune_router import _format_prompt

    instruction = "what is the typical concrete cover for a slab?"
    response = "25-40mm depending on exposure class"

    # 1. The two wrappers must produce identical output for identical input.
    assert format_prompt(instruction, response) == _format_prompt(instruction, response)
    assert format_prompt(instruction) == _format_prompt(instruction)

    # 2. The inference-time wrap (no response) MUST end with the response
    # prefix so the model continues generating from after "### Response:\n".
    inference_prompt = format_prompt(instruction)
    assert inference_prompt.endswith("### Response:\n"), (
        f"inference prompt must end at the response anchor; got: {inference_prompt!r}"
    )
    assert "### Instruction:" in inference_prompt


def test_generate_wraps_prompt_with_training_template(isolated_data_dir, monkeypatch):
    """The actual fix: generate() wraps the raw prompt in the Alpaca-style
    template before calling the pipeline. Without this, the model sees a
    different prompt shape at inference than during training, silently
    degrading quality. We mock the pipeline to capture exactly what the
    wrapper passed in."""
    from app.core.learning import local_model as _lm

    captured = {}
    def fake_pipe(text, **kwargs):
        captured["text"] = text
        captured["kwargs"] = kwargs
        return [{"generated_text": "  some answer  "}]

    monkeypatch.setattr(_lm, "get_pipeline", lambda: fake_pipe)
    result = _lm.generate("what is the typical slab cover?", max_new_tokens=100)

    # The wrapped prompt must start with the instruction marker
    assert captured["text"].startswith("### Instruction:")
    assert "what is the typical slab cover?" in captured["text"]
    # And end at the response anchor so the model continues from there
    assert captured["text"].endswith("### Response:\n")
    # The output is returned (and stripped of leading/trailing whitespace)
    assert result == "some answer"


def test_generate_truncates_at_next_instruction_boundary(isolated_data_dir, monkeypatch):
    """Defense-in-depth: if the adapter ever overruns EOS and starts a
    fresh '### Instruction:' block, we cut at the next turn boundary so
    callers don't see hallucinated follow-ups."""
    from app.core.learning import local_model as _lm

    def fake_pipe(text, **kwargs):
        # Simulate the model producing a real answer then leaking into a
        # hallucinated new turn
        return [{
            "generated_text": (
                "the slab cover is 30mm.\n\n"
                "### Instruction:\nand for columns?\n\n"
                "### Response:\n40mm typical"
            )
        }]
    monkeypatch.setattr(_lm, "get_pipeline", lambda: fake_pipe)

    result = _lm.generate("slab cover?")
    assert "slab cover is 30mm" in result
    assert "and for columns" not in result, (
        "must cut at the next ### Instruction: boundary, not return hallucinated turn"
    )


def test_invalidate_cache_drops_pipeline(isolated_data_dir, monkeypatch):
    """After swapping an adapter, operators call invalidate_cache(). Verify
    it actually clears the module-level reference."""
    from app.core.learning import local_model as _lm
    _lm._PIPELINE_CACHE = "fake_cached_pipeline"
    _lm.invalidate_cache()
    assert _lm._PIPELINE_CACHE is None


# ── chat block use_local_model opt-in ─────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_falls_back_to_cloud_when_local_unavailable(
    isolated_data_dir, monkeypatch,
):
    """use_local_model=true but the local stack is missing → silently
    falls back to the existing cloud path. Chat never goes dark because
    of a local-model gap."""
    from app.blocks.chat import ChatBlock

    monkeypatch.setenv("LOCAL_MODEL_UNAVAILABLE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_05)

    captured = {}
    async def fake_call(self, message, model, max_tokens, temperature, stream, key, cfg=None, **kwargs):
        captured["called"] = True
        captured["message"] = message
        return {"status": "success", "response": "cloud answer", "provider": "deepseek"}
    monkeypatch.setattr(ChatBlock, "_call_cloud", fake_call)

    cb = ChatBlock()
    result = await cb.process({"text": "hello"}, {"use_local_model": True})
    assert captured.get("called") is True, "cloud path should run when local unavailable"
    assert result["provider"] == "deepseek"
    assert result["response"] == "cloud answer"


@pytest.mark.asyncio
async def test_chat_uses_local_model_when_available(isolated_data_dir, monkeypatch):
    """use_local_model=true + local stack returns text → cloud path is
    bypassed entirely, response carries provider="local_lora"."""
    from app.blocks.chat import ChatBlock
    from app.core.learning import local_model as _lm

    # Force available() True and stub generate() to return canned text
    monkeypatch.setattr(_lm, "available", lambda: True)
    monkeypatch.setattr(_lm, "generate", lambda prompt, **kw: "fine-tuned answer")

    # Spy on the cloud path to assert it WASN'T called
    cloud_called = {"yes": False}
    async def fake_call(self, *a, **kw):
        cloud_called["yes"] = True
        return {"status": "success", "response": "cloud"}
    monkeypatch.setattr(ChatBlock, "_call_cloud", fake_call)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_05)

    cb = ChatBlock()
    result = await cb.process({"text": "hello"}, {"use_local_model": True})
    assert result["provider"] == "local_lora"
    assert result["response"] == "fine-tuned answer"
    assert cloud_called["yes"] is False


@pytest.mark.asyncio
async def test_chat_falls_back_when_generate_returns_none(
    isolated_data_dir, monkeypatch,
):
    """Mid-flight failure case: available() is True but generate()
    returns None (OOM, decode error, etc.). The chat block must fall
    through to the cloud path rather than returning an empty response."""
    from app.blocks.chat import ChatBlock
    from app.core.learning import local_model as _lm

    monkeypatch.setattr(_lm, "available", lambda: True)
    monkeypatch.setattr(_lm, "generate", lambda prompt, **kw: None)

    cloud_called = {"yes": False}
    async def fake_call(self, *a, **kw):
        cloud_called["yes"] = True
        return {"status": "success", "response": "cloud rescue", "provider": "deepseek"}
    monkeypatch.setattr(ChatBlock, "_call_cloud", fake_call)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_05)

    cb = ChatBlock()
    result = await cb.process({"text": "hello"}, {"use_local_model": True})
    assert cloud_called["yes"] is True
    assert result["response"] == "cloud rescue"


@pytest.mark.asyncio
async def test_chat_without_use_local_model_unchanged(isolated_data_dir, monkeypatch):
    """Default chat behavior is byte-for-byte unchanged when
    use_local_model is absent. Critical for not regressing existing
    deploys."""
    from app.blocks.chat import ChatBlock
    from app.core.learning import local_model as _lm

    # Even with the local stack "available", absence of the flag means
    # the local path must NOT be touched.
    monkeypatch.setattr(_lm, "available", lambda: True)
    monkeypatch.setattr(_lm, "generate", lambda *a, **kw: pytest.fail(
        "generate should NOT be called when use_local_model is absent"
    ))

    async def fake_call(self, *a, **kw):
        return {"status": "success", "response": "cloud", "provider": "deepseek"}
    monkeypatch.setattr(ChatBlock, "_call_cloud", fake_call)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_05)

    cb = ChatBlock()
    result = await cb.process({"text": "hello"})  # no use_local_model param
    assert result["provider"] == "deepseek"


# ── export_training_set.py ────────────────────────────────────────────────


def test_export_from_seed_file(isolated_data_dir, tmp_path):
    """Seed file path: --seed-data writes its rows to the output JSONL
    with source="seed_file" appended."""
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        '{"instruction": "what is concrete cover?", "response": "typically 40mm"}\n'
        '{"instruction": "how do you cure concrete?", "response": "moist cure 7 days"}\n'
    )
    out = tmp_path / "training.jsonl"

    result = subprocess.run([
        sys.executable, "scripts/export_training_set.py",
        "--out", str(out),
        "--seed-data", str(seed),
        "--no-agent-memory",
    ], capture_output=True, text=True, cwd=Path.cwd())
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(r["source"] == "seed_file" for r in rows)
    assert rows[0]["instruction"] == "what is concrete cover?"


def test_export_from_agent_memory(isolated_data_dir, tmp_path, monkeypatch):
    """Conversations in agent_memory.db become (instruction, response)
    pairs. The walker joins each user message with the immediately-
    following assistant response."""
    from app.core import agent_memory
    agent_memory._ensure_db()

    conv = agent_memory.get_or_create_conversation(
        conversation_id="c1", agent_name="chat", project_id="p1",
    )
    # Long-enough messages clear the 20-char min threshold
    agent_memory.append_message(conv["id"], "user", "what's the typical slab cover thickness?")
    agent_memory.append_message(conv["id"], "assistant", "typically 25 to 40 millimeters depending on exposure class")
    agent_memory.append_message(conv["id"], "user", "and for columns")
    # 16 chars — below the 20-char minimum; pair should be skipped
    agent_memory.append_message(conv["id"], "assistant", "40 to 75 mm")

    out = tmp_path / "training.jsonl"
    result = subprocess.run([
        sys.executable, "scripts/export_training_set.py",
        "--out", str(out),
        "--min-message-chars", "20",
    ], capture_output=True, text=True, cwd=Path.cwd(), env={**os.environ, "DATA_DIR": str(isolated_data_dir)})
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    # Only the first pair survives (the second assistant response is too short)
    assert len(rows) == 1
    assert rows[0]["source"] == "agent_memory"
    assert "slab cover" in rows[0]["instruction"]


def test_export_seed_and_memory_combined(isolated_data_dir, tmp_path):
    """Both sources at once: seed rows precede agent_memory rows in the
    output (deterministic ordering)."""
    from app.core import agent_memory
    agent_memory._ensure_db()
    conv = agent_memory.get_or_create_conversation(
        conversation_id="c1", agent_name="chat", project_id="p1",
    )
    agent_memory.append_message(conv["id"], "user", "what aggregate ratio for class C30?")
    agent_memory.append_message(conv["id"], "assistant", "approximately 1:2:4 cement:sand:aggregate")

    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        '{"instruction": "what is the curing duration?", "response": "minimum 7 days"}\n'
    )
    out = tmp_path / "training.jsonl"
    result = subprocess.run([
        sys.executable, "scripts/export_training_set.py",
        "--out", str(out),
        "--seed-data", str(seed),
    ], capture_output=True, text=True, cwd=Path.cwd(), env={**os.environ, "DATA_DIR": str(isolated_data_dir)})
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["source"] == "seed_file"
    assert rows[1]["source"] == "agent_memory"


def test_export_returns_nonzero_when_empty(isolated_data_dir, tmp_path):
    """Empty corpus → exit code 1 + warning. Helps catch silent
    misconfigurations (--no-agent-memory but no --seed-data)."""
    out = tmp_path / "training.jsonl"
    result = subprocess.run([
        sys.executable, "scripts/export_training_set.py",
        "--out", str(out),
        "--no-agent-memory",
    ], capture_output=True, text=True, cwd=Path.cwd(), env={**os.environ, "DATA_DIR": str(isolated_data_dir)})
    assert result.returncode == 1
    assert "ZERO rows" in result.stderr or "ZERO rows" in result.stdout
