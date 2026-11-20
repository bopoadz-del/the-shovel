"""U-2: no agent config pins a model its provider would refuse, and an
agent file without a ``model:`` line follows the operator's model setting.

Every input is built by the test (the env values and the stub file); the
configs themselves are read through the real loader.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import app.agents.runtime as rt


def _shipped_agents():
    return [rt._parse_agent_file(p) for p in sorted(rt.CONFIGS_DIR.glob("*.md"))]


def _provider_cfg(monkeypatch, provider: str, model_env: str, value: str):
    monkeypatch.setenv(model_env, value)
    return rt._llm_config(provider)


def test_the_shipped_configs_are_found():
    agents = _shipped_agents()
    assert len(agents) >= 10, "the agent configs were not read"


def test_no_shipped_config_pins_a_model_no_provider_keeps(monkeypatch):
    """A pin is legitimate only if some provider would actually send it.

    A retired id (one every provider remaps away) is a dead line that only
    the runtime remap keeps from 404ing the turn.
    """
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/free")
    cfgs = [rt._llm_config("deepseek"), rt._llm_config("openrouter")]
    dead = sorted(
        f"{a.name}: {a.model}"
        for a in _shipped_agents()
        if a.model and all(rt._resolve_attempt_model(c, a.model) != a.model for c in cfgs)
    )
    assert not dead, f"configs pin a model no provider sends: {dead}"


@pytest.mark.parametrize("value", ["deepseek-reasoner", "deepseek-chat"])
def test_every_shipped_agent_follows_deepseek_model(monkeypatch, value):
    cfg = _provider_cfg(monkeypatch, "deepseek", "DEEPSEEK_MODEL", value)
    sent = {a.name: rt._resolve_attempt_model(cfg, a.model) for a in _shipped_agents()}
    wrong = {n: m for n, m in sent.items() if m != value}
    assert not wrong, f"DEEPSEEK_MODEL={value} overridden for: {wrong}"


@pytest.mark.parametrize("value", ["openrouter/free", "some-vendor/some-model:free"])
def test_every_shipped_agent_follows_openrouter_model(monkeypatch, value):
    cfg = _provider_cfg(monkeypatch, "openrouter", "OPENROUTER_MODEL", value)
    sent = {a.name: rt._resolve_attempt_model(cfg, a.model) for a in _shipped_agents()}
    wrong = {n: m for n, m in sent.items() if m != value}
    assert not wrong, f"OPENROUTER_MODEL={value} overridden for: {wrong}"


def _write_agent(tmp_path: Path, name: str, model_line: str) -> Path:
    path = tmp_path / f"{name}.md"
    path.write_text(
        f"---\nname: {name}\ndescription: d\n{model_line}temperature: 0.1\n---\n"
        "you are a test agent\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("value", ["deepseek-reasoner", "deepseek-chat"])
def test_a_file_without_a_model_line_is_unpinned(tmp_path, monkeypatch, value):
    agent = rt._parse_agent_file(_write_agent(tmp_path, "nomodel", ""))
    assert agent.model == ""
    cfg = _provider_cfg(monkeypatch, "deepseek", "DEEPSEEK_MODEL", value)
    assert rt._resolve_attempt_model(cfg, agent.model) == value


@pytest.mark.parametrize("pin", ["deepseek-reasoner", "deepseek-chat"])
def test_an_explicit_pin_is_still_kept(tmp_path, monkeypatch, pin):
    agent = rt._parse_agent_file(_write_agent(tmp_path, "pinned", f"model: {pin}\n"))
    assert agent.model == pin
    other = "deepseek-chat" if pin == "deepseek-reasoner" else "deepseek-reasoner"
    cfg = _provider_cfg(monkeypatch, "deepseek", "DEEPSEEK_MODEL", other)
    assert rt._resolve_attempt_model(cfg, agent.model) == pin
