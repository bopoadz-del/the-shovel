"""U-2b: the retired LLM vendor cannot come back as a provider, fallback or option.

Owner ruling: never, as provider, fallback or option. #474 (U-2) retired the
pin from the 15 agent configs. This file is a literal guard: the vendor's
names cannot return to ``app/agents/configs``. It also checks that the
``llm_enhancer`` block carries no model or provider default of its own. With
no caller ``model`` it leaves the choice to the chat block, which uses the
active provider's default. Every input below is constructed here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _ROOT / "app" / "agents" / "configs"

# The retired vendor and its model family, matched case-insensitively.
_RETIRED = ("kimi", "moonshot")


def _retired_hits(root: Path) -> list[str]:
    """Every ``file:line`` under ``root`` that names a retired vendor."""
    hits: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if any(name in low for name in _RETIRED):
                hits.append(f"{path.relative_to(root)}:{n}")
    return hits


def test_agent_configs_never_name_the_retired_vendor():
    assert _CONFIGS.is_dir()
    assert any(_CONFIGS.iterdir()), "the configs directory must not be empty"
    assert _retired_hits(_CONFIGS) == []


@pytest.mark.parametrize(
    "line",
    ["model: Kimi-K2.6\n", "fallback: MOONSHOT-v1-32k\n"],
)
def test_guard_flags_a_planted_mention(tmp_path, line):
    (tmp_path / "hat.md").write_text("---\nname: hat\n" + line + "---\n", encoding="utf-8")
    assert _retired_hits(tmp_path) == ["hat.md:3"]


@pytest.mark.parametrize(
    "line",
    ["model: deepseek-chat\n", "notes: a kimono is not a model\n"],
)
def test_guard_passes_a_clean_config(tmp_path, line):
    (tmp_path / "hat.md").write_text("---\nname: hat\n" + line + "---\n", encoding="utf-8")
    assert _retired_hits(tmp_path) == []


def test_probe_script_is_gone():
    assert not (_ROOT / "scripts" / "kimi_probe.py").exists()


# ── llm_enhancer: no model/provider default of its own ─────────────────────


class _FakeChat:
    """Records what the enhancer sends and reports a served model."""

    def __init__(self, served_model, text='{"k": "v"}'):
        self.served_model = served_model
        self.text = text
        self.sent: list[dict] = []

    async def execute(self, prompt, params):
        self.sent.append(dict(params))
        result = {"text": self.text}
        if self.served_model is not None:
            result["model"] = self.served_model
        return {"status": "success", "result": result}


def _enhancer(chat):
    from app.blocks.llm_enhancer import LLMEnhancerBlock

    block = LLMEnhancerBlock()
    block.wire("chat", chat)
    return block


def test_enhancer_config_has_no_provider_or_model_default():
    from app.blocks.llm_enhancer import LLMEnhancerBlock

    cfg = LLMEnhancerBlock.default_config
    assert "default_provider" not in cfg and "model" not in cfg
    assert not any(name in str(cfg).lower() for name in _RETIRED)


@pytest.mark.parametrize("served", ["served-model-one", "served-model-two"])
def test_enhancer_without_caller_model_sends_none_and_reports_served(served):
    chat = _FakeChat(served)
    out = asyncio.run(_enhancer(chat).process("Pump P-7 rated 40 kW", {"action": "extract"}))
    assert "model" not in chat.sent[0]
    assert out["status"] == "success"
    assert out["model_used"] == served


@pytest.mark.parametrize("caller", ["caller-model-a", "caller-model-b"])
def test_enhancer_passes_the_callers_model_through(caller):
    chat = _FakeChat(served_model=None, text="not json")
    out = asyncio.run(
        _enhancer(chat).process("Wall W3 is 200 mm", {"action": "summarize", "model": caller})
    )
    assert chat.sent[0]["model"] == caller
    assert out["model_used"] == caller


def test_enhancer_reports_the_served_model_over_the_callers():
    chat = _FakeChat(served_model="served-after-remap")
    out = asyncio.run(
        _enhancer(chat).process("Door D2 width 900", {"action": "extract", "model": "asked-model"})
    )
    assert chat.sent[0]["model"] == "asked-model"
    assert out["model_used"] == "served-after-remap"
