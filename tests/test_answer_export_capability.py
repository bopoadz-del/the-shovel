"""The model is told which files the user can download from its answer.

Live defect (17 Nov): the assistant told the user "I can't produce a .docx
file from here" while a Word button sat under its own answer. No agent prompt
said that every finished answer carries download controls, so the model
guessed.

Mechanism under test (one declaration, three consumers):
- ``app.core.answer_report_intent.ANSWER_EXPORT_FORMATS`` declares the formats;
- ``app.core.answer_report_intent.answer_export_capability_note`` builds the
  prompt section from it, and ``app.agents.runtime._apply_answer_export_capability``
  appends that section to every agent loaded by ``_parse_agent_file``;
- ``app.core.answer_report_intent.answer_export_format_pattern`` and
  ``answer_export_declined_message`` drive the route
  ``app.routers.exports.export_conversation_message``;
- the workspace controls (``frontend/src/chat/ChatBubble.tsx``) must offer
  exactly the declared formats (drift test).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import answer_report_intent as ari

REPO = Path(__file__).resolve().parents[1]


def _agent_file(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.md"
    path.write_text(
        f"---\nname: {name}\ndescription: test agent\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


# --- the prompt every agent is loaded with ---------------------------------

@pytest.mark.parametrize("name,body", [
    ("plain-agent", "You answer questions about the project."),
    ("other-agent", "You are a quantity surveyor. Never invent numbers."),
])
def test_every_parsed_agent_is_told_the_download_formats(tmp_path, name, body):
    from app.agents.runtime import _parse_agent_file

    agent = _parse_agent_file(_agent_file(tmp_path, name, body))
    prompt = agent.system_prompt
    assert prompt.startswith(body)
    for ext, label in ari.ANSWER_EXPORT_FORMATS.items():
        assert f"{label} (.{ext})" in prompt
        assert f"'{label}'" in prompt
    for ext, label in ari.ANSWER_EXPORT_DECLINED_FORMATS.items():
        assert re.search(rf"Not offered: .*{label} \(\.{ext}\)", prompt)
    assert "Never tell the user you cannot produce" in prompt


@pytest.mark.parametrize("agent_name", ["project-assistant", "quantity-surveyor"])
def test_shipped_agents_carry_the_section(agent_name):
    from app.agents.runtime import _parse_agent_file

    cfg = REPO / "app" / "agents" / "configs" / f"{agent_name}.md"
    agent = _parse_agent_file(cfg)
    assert agent.system_prompt.count("## Files you can hand the user") == 1
    assert ari.answer_export_capability_note().strip() in agent.system_prompt


@pytest.mark.parametrize("formats,absent", [
    ({"docx": "Word"}, "Excel"),
    ({"xlsx": "Excel"}, "Word"),
])
def test_note_follows_the_declaration(monkeypatch, formats, absent):
    """The section is derived, not written: drop a format and it is gone."""
    from app.agents.runtime import _apply_answer_export_capability

    monkeypatch.setattr(ari, "ANSWER_EXPORT_FORMATS", formats)
    out = _apply_answer_export_capability("BODY")
    assert out.startswith("BODY")
    (ext, label), = formats.items()
    assert f"{label} (.{ext})" in out
    assert absent not in out


@pytest.mark.parametrize("declined,expect", [
    ({}, False),
    ({"pdf": "PDF"}, True),
])
def test_declined_line_only_when_something_is_declined(monkeypatch, declined, expect):
    monkeypatch.setattr(ari, "ANSWER_EXPORT_DECLINED_FORMATS", declined)
    assert ("Not offered:" in ari.answer_export_capability_note()) is expect


# --- the route follows the same declaration --------------------------------

@pytest.mark.parametrize("fmt,ok", [
    ("docx", True), ("xlsx", True), ("pdf", True),
    ("csv", False), ("docxx", False),
])
def test_route_pattern_is_offered_plus_declined(fmt, ok):
    assert bool(re.fullmatch(ari.answer_export_format_pattern(), fmt)) is ok


@pytest.mark.parametrize("fmt", ["pdf", "PDF-like"])
def test_declined_message_names_every_offered_format(fmt):
    msg = ari.answer_export_declined_message(fmt)
    assert f"format='{fmt}'" in msg
    for ext in ari.ANSWER_EXPORT_FORMATS:
        assert f"'{ext}'" in msg


class TestRoute:
    """export_conversation_message serves each declared format, refuses the rest."""

    @pytest.fixture(autouse=True)
    def _wired(self, monkeypatch):
        from app.dependencies import require_user
        from app.main import app
        from app.routers import exports as mod

        app.dependency_overrides[require_user] = lambda: {
            "user_id": "u1", "role": "user",
        }
        monkeypatch.setattr(mod, "_check_owner", lambda *_a: {"name": "Proj"})
        monkeypatch.setattr(
            mod, "_require_conversation_in_project", lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            mod.agent_memory, "get_messages",
            lambda *_a, **_k: [
                {"role": "user", "content": "Summarise the scope."},
                {"role": "assistant", "content": "Scope:\n\n| A | B |\n|---|---|\n| 1 | 2 |"},
            ],
        )
        self.app = app
        yield
        app.dependency_overrides.clear()

    def _post(self, fmt: str):
        with TestClient(self.app) as client:
            return client.post(
                f"/v1/projects/p1/conversations/conv12345678/export?format={fmt}"
            )

    @pytest.mark.parametrize("fmt", sorted(ari.ANSWER_EXPORT_FORMATS))
    def test_each_declared_format_downloads(self, fmt):
        res = self._post(fmt)
        assert res.status_code == 200
        assert res.headers["content-disposition"].endswith(f'.{fmt}"')
        assert res.content[:2] == b"PK"  # a real OOXML zip, not an error page

    @pytest.mark.parametrize("fmt", sorted(ari.ANSWER_EXPORT_DECLINED_FORMATS))
    def test_declined_format_is_a_plain_501(self, fmt):
        res = self._post(fmt)
        assert res.status_code == 501
        assert res.json()["detail"] == ari.answer_export_declined_message(fmt)

    @pytest.mark.parametrize("fmt", ["csv", "txt"])
    def test_undeclared_format_is_rejected(self, fmt):
        assert self._post(fmt).status_code == 422


# --- the controls the user presses offer exactly the declared formats -------

def _bubble_source() -> str:
    return (REPO / "frontend" / "src" / "chat" / "ChatBubble.tsx").read_text(
        encoding="utf-8",
    )


@pytest.mark.parametrize("pattern,expected", [
    (r"onDownload\('(\w+)'\)", lambda: set(ari.ANSWER_EXPORT_FORMATS)),
    (r"<span>(\w+)</span>", lambda: set(ari.ANSWER_EXPORT_FORMATS.values())),
])
def test_answer_controls_match_the_declaration(pattern, expected):
    assert set(re.findall(pattern, _bubble_source())) == expected()
