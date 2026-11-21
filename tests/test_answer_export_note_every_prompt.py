"""Every prompt a turn opens with tells the model what the user can download.

Follow-ups to #406 (grader, 17 Nov):
1. the buttons exist only on a finished answer -- the section says so;
2. the answer is written in the user's language -- the section says so;
3. a test that fails if any agent prompt is assembled without the section.

There are two places a turn's opening system prompt is assembled:
- ``app.agents.runtime._parse_agent_file`` (every configured agent; the turn
  sends ``agent.system_prompt`` first, ``app.agents.core.context._build_messages``);
- ``app.agents.driver.context.system_message`` (driver mode builds its own,
  from the standing rules, the hat catalogue and the chosen hat).
Before this change the driver path carried no download section at all.
Each test below builds its own inputs; nothing reads a recorded turn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import answer_report_intent as ari

REPO = Path(__file__).resolve().parents[1]
HEADING = "## Files you can hand the user"
SHIPPED = sorted((REPO / "app" / "agents" / "configs").glob("*.md"))


def _note_core() -> str:
    return ari.answer_export_capability_note().strip()


# --- what the section says --------------------------------------------------

@pytest.mark.parametrize("formats", [
    {"docx": "Word", "xlsx": "Excel"},
    {"xlsx": "Excel"},
])
def test_section_says_buttons_only_on_a_finished_answer(monkeypatch, formats):
    monkeypatch.setattr(ari, "ANSWER_EXPORT_FORMATS", formats)
    note = ari.answer_export_capability_note()
    assert "appear only once your answer has finished" in note
    assert "still being written" in note
    assert "ends in an error has none" in note
    assert "once the answer is complete" in note
    for label in formats.values():
        assert f"'{label}'" in note


@pytest.mark.parametrize("formats", [
    {"docx": "Word", "xlsx": "Excel"},
    {"docx": "Word"},
])
def test_section_says_write_in_the_users_language(monkeypatch, formats):
    monkeypatch.setattr(ari, "ANSWER_EXPORT_FORMATS", formats)
    note = ari.answer_export_capability_note()
    assert "in the language the user writes in" in note
    # The button names are the one thing kept as labelled.
    assert "only the button names stay as shown above" in note


# --- the agent path: every shipped config, and invented ones ----------------

@pytest.mark.parametrize("cfg", SHIPPED, ids=[p.stem for p in SHIPPED])
def test_every_shipped_agent_opens_its_turn_with_the_section_once(cfg):
    from app.agents.runtime import _parse_agent_file

    agent = _parse_agent_file(cfg)
    msgs = agent._build_messages("How much concrete is in the slab?", [], None)
    opening = msgs[0]
    assert opening["role"] == "system"
    assert opening["content"].count(HEADING) == 1
    assert _note_core() in opening["content"]


def test_there_are_shipped_agents_to_check():
    # Guards the parametrisation above against an empty glob passing vacuously.
    assert len(SHIPPED) >= 2


@pytest.mark.parametrize("name,body", [
    ("invented-estimator", "You price earthworks from the user's own rates."),
    ("invented-reader", "You read the user's documents and quote them."),
])
def test_an_invented_agent_opens_its_turn_with_the_section_once(tmp_path, name, body):
    from app.agents.runtime import _parse_agent_file

    path = tmp_path / f"{name}.md"
    path.write_text(f"---\nname: {name}\ndescription: t\n---\n\n{body}\n", encoding="utf-8")
    agent = _parse_agent_file(path)
    opening = agent._build_messages("Give me this as a Word file", [], None)[0]
    assert opening["content"].startswith(body)
    assert opening["content"].count(HEADING) == 1


# --- the driver path: no hat, and every hat the catalogue loads -------------

def _driver_hats() -> list:
    from app.agents.driver import context

    return [None] + [h["discipline"] for h in context.hats()]


@pytest.mark.parametrize("hat", _driver_hats())
def test_driver_mode_opens_its_turn_with_the_section_once(hat):
    from app.agents.driver import context

    msgs = context.messages("Can I have that as Excel?", [], None, None, hat)
    opening = msgs[0]
    assert opening["role"] == "system"
    assert opening["content"].count(HEADING) == 1
    assert _note_core() in opening["content"]


def test_driver_mode_has_hats_to_check():
    assert len(_driver_hats()) >= 2


@pytest.mark.parametrize("formats,absent", [
    ({"docx": "Word"}, "Excel (.xlsx)"),
    ({"xlsx": "Excel"}, "Word (.docx)"),
])
def test_driver_section_follows_the_declaration(monkeypatch, formats, absent):
    """Driver mode reads the same declaration, not a copy of the text."""
    from app.agents.driver import context

    monkeypatch.setattr(ari, "ANSWER_EXPORT_FORMATS", formats)
    content = context.system_message(None, None, None)["content"]
    (ext, label), = formats.items()
    assert f"{label} (.{ext})" in content
    assert absent not in content


@pytest.mark.parametrize("proj", ["proj-alpha", "proj-beta"])
def test_driver_section_is_in_the_static_lead_before_the_profile(proj):
    """The section is the same for every project, so it sits before the
    per-project profile (the shared, cacheable lead stays shared)."""
    from app.agents.driver import context

    content = context.system_message(proj, None, None)["content"]
    assert content.index(HEADING) < content.rindex(proj)
    assert content.index("Hats you can wear") < content.index(HEADING)
