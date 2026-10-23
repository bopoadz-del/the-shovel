"""UI-PHYS H1: 'Export A1-A9 answers as a docx report' is a conversation export.

Live on ``567147a`` / theshovel.ai: the ask produced no docx. ``A1-A9`` is
identifier-shaped, retrieval treated it as RFP appendix/attachment codes,
and the turn listed those documents instead of compiling the chat answers.

Every assertion here reads a real predicate, a real pair list, or a real
docx ZIP — never a mocked renderer return value.
"""

from __future__ import annotations

import asyncio
import zipfile

import pytest

from app.core.answer_report_intent import (
    ANSWER_REPORT_BLOCKED_ACTIONS,
    answer_report_export_descriptor,
    collect_answer_pairs,
    compose_answer_report_reply,
    export_workspace_project_id,
    message_wants_answer_report,
    parse_answer_report_range,
)
from app.core.rag.retriever import extract_query_identifiers
from app.routers.exports import _render_answers_docx
from tests.conftest import requires_construction_kit

# Verbatim live H1 ask (UI-PHYS "Question (ask exactly)").
H1_EXPORT_ASK = "Export A1-A9 answers as a docx report"

# Fixture-only figures — never live client amounts.
A1_ANSWER = (
    "The Accepted Contract Amount, **excluding VAT**, is:"
    "**SAR 8,640,000.00**."
)
A5_ANSWER = "Delay Damages are **0.2%** of the Accepted Contract Amount per day."
A9_ANSWER = "The Engineer is **Named Engineer** of Example Consulting."


def _turns(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for q, a in pairs:
        out.append({"role": "user", "content": q})
        out.append({"role": "assistant", "content": a})
    return out


A1_A9_HISTORY = _turns(
    ("What is the Accepted Contract Amount?", A1_ANSWER),
    ("What is the Time for Completion?", "420 days."),
    ("What is the Defects Notification Period?", "365 days."),
    ("What is the performance bond?", "10 percent."),
    ("What are the Delay Damages?", A5_ANSWER),
    ("How many milestones?", "Five."),
    ("What is Milestone 5 Time for Completion?", "400 days."),
    ("What is the method of communication?", "Aconex."),
    ("Who is the Engineer?", A9_ANSWER),
)


# --------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------


def test_verbatim_h1_ask_is_an_answer_report():
    assert message_wants_answer_report(H1_EXPORT_ASK)
    assert message_wants_answer_report(
        "Export A1-A9 answers as a docx report"
    )
    assert parse_answer_report_range(H1_EXPORT_ASK) == (1, 9)


def test_conversation_word_export_is_an_answer_report():
    assert message_wants_answer_report(
        "Export this conversation as a Word document"
    )


@pytest.mark.parametrize("msg", [
    "Prepare an RFP for the landscaping subcontract package",
    "Export the RFP attachments as a zip",
    "List the RFP attachments A1-A9",
    "What does attachment A1 of the RFP say?",
    "give me a docx copy",
    "What is the export value of the works?",
    "Who is the Engineer?",
    "What is the Time for Completion for the whole of the Works?",
    "Download times are slow on site",
])
def test_rfp_and_lookups_are_not_answer_reports(msg):
    assert not message_wants_answer_report(msg), msg


def test_a1_a9_is_identifier_shaped_that_is_the_landmine():
    """Documents the 567147a failure mode: do not 'fix' extraction here.

    The fence is ``message_wants_answer_report``, which must fire BEFORE
    RAG so this identifier never searches the RFP corpus.
    """
    found = extract_query_identifiers(H1_EXPORT_ASK)
    assert "a1-a9" in found


# --------------------------------------------------------------------------
# Pair collection
# --------------------------------------------------------------------------


def test_collects_a1_through_a9_with_figures_intact():
    pairs = collect_answer_pairs(A1_A9_HISTORY, 1, 9)
    assert [p["label"] for p in pairs] == [f"A{i}" for i in range(1, 10)]
    assert "8,640,000.00" in pairs[0]["answer"]
    assert "0.2%" in pairs[4]["answer"]
    assert "Named Engineer" in pairs[8]["answer"]


def test_skips_the_export_ask_and_the_ready_confirmation():
    msgs = A1_A9_HISTORY + [
        {"role": "user", "content": H1_EXPORT_ASK},
        {"role": "assistant", "content": compose_answer_report_reply(
            collect_answer_pairs(A1_A9_HISTORY, 1, 9), 1, 9,
        )},
    ]
    pairs = collect_answer_pairs(msgs, 1, 9)
    assert len(pairs) == 9
    assert "8,640,000.00" in pairs[0]["answer"]
    assert "Prepared a Word report" not in pairs[-1]["answer"]


def test_honest_when_the_thread_is_shorter_than_a9():
    pairs = collect_answer_pairs(A1_A9_HISTORY[:6], 1, 9)  # 3 Q&A
    assert [p["label"] for p in pairs] == ["A1", "A2", "A3"]
    reply = compose_answer_report_reply(pairs, 1, 9)
    assert "A1–A3" in reply
    assert "A1–A9 was asked" in reply


# --------------------------------------------------------------------------
# Docx bytes (ZIP walk — same contract as H1/H2 footer tests)
# --------------------------------------------------------------------------


def _report_zip(pairs=None, base_url="https://theshovel.ai"):
    pairs = pairs or collect_answer_pairs(A1_A9_HISTORY, 1, 9)
    path = _render_answers_docx(
        "Master Corpus", pairs, "ws-master-corpus-h1", base_url, "A1–A9",
    )
    return zipfile.ZipFile(path)


def test_report_docx_opens_and_keeps_figures():
    z = _report_zip()
    assert z.testzip() is None
    doc = z.read("word/document.xml").decode()
    assert "A1" in doc and "A9" in doc
    assert "8,640,000.00" in doc
    assert "0.2%" in doc
    assert "Named Engineer" in doc
    assert "**" not in doc


def test_report_docx_has_a_real_live_url_footer():
    z = _report_zip()
    names = z.namelist()
    footer_parts = [
        n for n in names if n.startswith("word/footer") and n.endswith(".xml")
    ]
    assert footer_parts, names
    footer = z.read(footer_parts[0]).decode()
    assert "theshovel.ai" in footer
    assert "Generated by The Shovel" in footer
    assert "localhost" not in footer


def test_report_docx_never_stamps_localhost():
    z = _report_zip(base_url="")
    footer_parts = [
        n for n in z.namelist()
        if n.startswith("word/footer") and n.endswith(".xml")
    ]
    footer = z.read(footer_parts[0]).decode()
    body = z.read("word/document.xml").decode()
    assert "localhost" not in footer
    assert "localhost" not in body


# --------------------------------------------------------------------------
# Descriptor + runtime short-circuit
# --------------------------------------------------------------------------


def test_descriptor_points_at_scope_answers_not_an_rfp_endpoint():
    desc = answer_report_export_descriptor("master_corpus", "ws-x-1", 1, 9)
    assert desc["format"] == "docx"
    assert "scope=answers" in desc["endpoint"]
    assert "range=A1-A9" in desc["endpoint"]
    assert "rfp" not in desc["endpoint"].lower()
    assert "schedule-from-document" not in desc["endpoint"]


def test_descriptor_uses_ui_alias_not_drive_archive_source(monkeypatch):
    """Live H1: chat remaps master_corpus → drive_archive; the button must not.

    The 404 was ``Project 'drive_archive' not found`` while the user was
    on Master Corpus. Fail if the offer looks up the archive corpus id.
    """
    from app.core import projects as projects_mod

    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_PROJECT_ID", "master_corpus")
    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_SOURCE_PROJECT_ID", "drive_archive")

    assert export_workspace_project_id(
        "drive_archive", "ws-master_corpus-1740000000000",
    ) == "master_corpus"
    assert export_workspace_project_id("drive_archive", "ws-other-1") == "master_corpus"
    assert export_workspace_project_id("p-owned", "ws-p-owned-1") == "p-owned"

    desc = answer_report_export_descriptor(
        "drive_archive", "ws-master_corpus-1740000000000", 1, 9,
    )
    assert "/v1/projects/master_corpus/" in desc["endpoint"]
    assert "drive_archive" not in desc["endpoint"]
    assert "scope=answers" in desc["endpoint"]
    assert "range=A1-A9" in desc["endpoint"]


def test_chat_short_circuit_offers_the_docx_and_does_not_mention_rfp_files():
    from app.agents.runtime import Agent

    agent = Agent(
        name="project-assistant",
        description="test",
        system_prompt="test",
        allowed_blocks=[],
    )
    cid = "ws-h1-answer-report-1"
    from app.core import agent_memory
    agent_memory.get_or_create_conversation(cid, "project-assistant", "p-h1")
    for m in A1_A9_HISTORY:
        agent_memory.append_message(cid, m["role"], m["content"])

    out = asyncio.run(agent.chat(
        H1_EXPORT_ASK, project_id="p-h1", conversation_id=cid,
    ))
    assert out["status"] == "success"
    assert "Prepared a Word report of answers A1–A9" in out["answer"]
    assert "not an RFP" in out["answer"]
    assert out["exports"], "no download offer — the docx cannot land"
    offer = out["exports"][0]
    assert offer["format"] == "docx"
    assert "scope=answers" in offer["endpoint"]
    assert out["tool_calls"] == []


def test_short_circuit_stamps_ui_project_when_runtime_has_source_id(monkeypatch):
    """Reproduce the live remap: agent sees drive_archive, button must not."""
    from app.agents.runtime import Agent
    from app.core import agent_memory
    from app.core import projects as projects_mod

    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_PROJECT_ID", "master_corpus")
    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_SOURCE_PROJECT_ID", "drive_archive")

    agent = Agent(
        name="project-assistant",
        description="test",
        system_prompt="test",
        allowed_blocks=[],
    )
    cid = "ws-master_corpus-h1-scope"
    agent_memory.get_or_create_conversation(cid, "project-assistant", "drive_archive")
    for m in A1_A9_HISTORY:
        agent_memory.append_message(cid, m["role"], m["content"])

    out = asyncio.run(agent.chat(
        H1_EXPORT_ASK, project_id="drive_archive", conversation_id=cid,
    ))
    assert out["status"] == "success"
    assert out["exports"], "no download offer — the docx cannot land"
    offer = out["exports"][0]
    assert "/v1/projects/master_corpus/" in offer["endpoint"]
    assert "drive_archive" not in offer["endpoint"]


def test_stream_short_circuit_emits_exports_on_end():
    from app.agents.runtime import Agent

    agent = Agent(
        name="project-assistant",
        description="test",
        system_prompt="test",
        allowed_blocks=[],
    )
    cid = "ws-h1-answer-report-stream"
    from app.core import agent_memory
    agent_memory.get_or_create_conversation(cid, "project-assistant", "p-h1")
    for m in A1_A9_HISTORY:
        agent_memory.append_message(cid, m["role"], m["content"])

    async def _collect():
        events = []
        async for ev in agent.chat_stream(
            H1_EXPORT_ASK, project_id="p-h1", conversation_id=cid,
        ):
            events.append(ev)
        return events

    events = asyncio.run(_collect())
    ends = [e for e in events if e.get("type") == "end"]
    assert ends
    exports = ends[-1].get("exports") or []
    assert exports and "scope=answers" in exports[0]["endpoint"]
    tokens = "".join(e.get("content") or "" for e in events if e.get("type") == "token")
    assert "Prepared a Word report" in tokens
    assert not any(e.get("type") == "error" for e in events)


def test_kill_switch_restores_the_old_path(monkeypatch):
    from app.agents.runtime import Agent, _fulfill_answer_report

    monkeypatch.setenv("ANSWER_REPORT_EXPORT", "0")
    from app.core.answer_report_intent import answer_report_export_enabled
    assert answer_report_export_enabled() is False

    agent = Agent(
        name="project-assistant",
        description="test",
        system_prompt="test",
        allowed_blocks=[],
    )
    # Without a key the un-short-circuited path errors — that proves the
    # kill-switch did not take the no-LLM export path.
    out = asyncio.run(agent.chat(H1_EXPORT_ASK, project_id="p-h1"))
    assert out.get("status") == "error" or "Prepared a Word report" not in (
        out.get("answer") or ""
    )
    # The helper itself still works; only the gate is off.
    assert _fulfill_answer_report


def test_select_agent_stays_on_project_assistant():
    from app.agents.runtime import Agent, select_agent_for_message

    pa = Agent(name="project-assistant", description="t", system_prompt="t")
    final, info = asyncio.run(select_agent_for_message(H1_EXPORT_ASK, pa))
    assert final.name == "project-assistant"
    assert info["reason"] == "answer_report_export"


def test_rfp_draft_predicate_does_not_steal_h1():
    from app.agents import runtime
    assert not runtime._message_wants_rfp_draft(H1_EXPORT_ASK)
    assert runtime._asks_for_export(H1_EXPORT_ASK)


@pytest.mark.asyncio
async def test_file_predispatch_does_not_fetch_a_docx_for_h1(monkeypatch):
    from app.agents import runtime

    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("list_documents must not run for an answer report")

    monkeypatch.setattr("app.core.projects.list_documents", _boom)
    agent = runtime.Agent(
        name="project-assistant", description="t", system_prompt="t",
        allowed_blocks=["bim_extractor"],
    )
    rec = await runtime._predispatch_file_tool(
        agent,
        [{"role": "user", "content": H1_EXPORT_ASK}],
        "p-h1",
    )
    assert rec is None
    assert called["n"] == 0


# --------------------------------------------------------------------------
# HTTP endpoint
# --------------------------------------------------------------------------


@pytest.fixture
def _export_http(monkeypatch):
    from fastapi.testclient import TestClient

    from app.core import agent_memory
    from app.dependencies import require_user
    from app.main import app
    from app.routers import exports as exp_mod

    app.dependency_overrides[require_user] = lambda: {
        "user_id": "u1", "role": "user",
    }
    monkeypatch.setattr(exp_mod, "_check_owner", lambda *_a, **_k: {"name": "Master Corpus"})
    cid = "convh1export12"
    agent_memory.get_or_create_conversation(cid, "project-assistant", "p1")
    for m in A1_A9_HISTORY:
        agent_memory.append_message(cid, m["role"], m["content"])
    with TestClient(app) as client:
        yield client, cid
    app.dependency_overrides.clear()


def test_scope_answers_returns_a_docx_of_a1_a9(_export_http):
    import io

    client, cid = _export_http
    res = client.post(
        f"/v1/projects/p1/conversations/{cid}/export"
        f"?format=docx&scope=answers&range=A1-A9",
        headers={"host": "theshovel.ai", "x-forwarded-proto": "https"},
    )
    assert res.status_code == 200, res.text
    assert "wordprocessingml" in res.headers["content-type"]
    assert res.content[:2] == b"PK"
    z = zipfile.ZipFile(io.BytesIO(res.content))
    doc = z.read("word/document.xml").decode()
    assert "8,640,000.00" in doc
    assert "Named Engineer" in doc
    footer = next(
        n for n in z.namelist()
        if n.startswith("word/footer") and n.endswith(".xml")
    )
    assert "theshovel.ai" in z.read(footer).decode()


def test_default_scope_is_still_last_message_only(_export_http):
    """Left-panel export must not silently become the A1–A9 report."""
    import io

    client, cid = _export_http
    res = client.post(
        f"/v1/projects/p1/conversations/{cid}/export?format=docx",
    )
    assert res.status_code == 200, res.text
    z = zipfile.ZipFile(io.BytesIO(res.content))
    doc = z.read("word/document.xml").decode()
    assert "Named Engineer" in doc
    assert "8,640,000.00" not in doc
    assert "Conversation Excerpt" in doc


def test_export_http_looks_up_master_corpus_not_drive_archive(monkeypatch):
    """Owner gate must resolve the UI project, never the RAG archive id.

    Live: the A1–A9 button POSTed ``/v1/projects/drive_archive/.../export``
    and ``_check_owner`` 404'd. New descriptors stamp ``master_corpus``;
    a leftover ``drive_archive`` URL must also remap before lookup.
    """
    from fastapi.testclient import TestClient

    from app.core import agent_memory
    from app.core import projects as projects_mod
    from app.dependencies import require_user
    from app.main import app

    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_PROJECT_ID", "master_corpus")
    monkeypatch.setattr(projects_mod, "MASTER_CORPUS_SOURCE_PROJECT_ID", "drive_archive")

    looked_up: list[str] = []
    real_get = projects_mod.get_project

    def _spy(project_id, *args, **kwargs):
        looked_up.append(project_id)
        if project_id == "master_corpus":
            return {"id": "master_corpus", "name": "Master Corpus"}
        if project_id == "drive_archive":
            return None
        return real_get(project_id, *args, **kwargs)

    monkeypatch.setattr(projects_mod, "get_project", _spy)

    app.dependency_overrides[require_user] = lambda: {
        "user_id": "u1", "role": "user",
    }
    cid = "ws-master_corpus-h1http"
    agent_memory.get_or_create_conversation(cid, "project-assistant", "drive_archive")
    for m in A1_A9_HISTORY:
        agent_memory.append_message(cid, m["role"], m["content"])

    desc = answer_report_export_descriptor("drive_archive", cid, 1, 9)
    assert "drive_archive" not in desc["endpoint"]
    assert "/v1/projects/master_corpus/" in desc["endpoint"]

    try:
        with TestClient(app) as client:
            res = client.post(
                desc["endpoint"],
                headers={"host": "theshovel.ai", "x-forwarded-proto": "https"},
            )
            assert res.status_code == 200, res.text
            assert res.content[:2] == b"PK"
            assert "master_corpus" in looked_up
            assert "drive_archive" not in looked_up

            looked_up.clear()
            stale = (
                f"/v1/projects/drive_archive/conversations/{cid}/export"
                f"?format=docx&scope=answers&range=A1-A9"
            )
            stale_res = client.post(
                stale,
                headers={"host": "theshovel.ai", "x-forwarded-proto": "https"},
            )
            assert stale_res.status_code == 200, stale_res.text
            assert "master_corpus" in looked_up
            assert "drive_archive" not in looked_up
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Orchestrator steal-guard
# --------------------------------------------------------------------------


def test_blocked_actions_include_rfp_and_process_document():
    assert "rfp_management" in ANSWER_REPORT_BLOCKED_ACTIONS
    assert "rfp_draft" in ANSWER_REPORT_BLOCKED_ACTIONS
    assert "process_document" in ANSWER_REPORT_BLOCKED_ACTIONS


@requires_construction_kit
def test_orchestrator_does_not_queue_rfp_for_h1():
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    result = asyncio.run(
        SmartOrchestratorBlock().process({"user_message": H1_EXPORT_ASK})
    )
    queued = result.get("action_queue") or []
    matched = [m["action"] for m in (result.get("matched_actions") or [])]
    for banned in ANSWER_REPORT_BLOCKED_ACTIONS:
        assert banned not in queued, queued
        assert banned not in matched, matched


@requires_construction_kit
def test_orchestrator_strips_document_listing_even_when_keywords_hit():
    """Mutation probe: the filter must be wired into ``_match_actions``.

    This phrasing is still the H1 export (range + answers + docx) but also
    names 'which documents' / 'RFP' — enough to score process_document and
    rfp_management above gate-1 if the steal-guard is deleted.
    """
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    msg = (
        H1_EXPORT_ASK
        + ". Prepare an RFP. Which documents and which drawings are in this project?"
    )
    assert message_wants_answer_report(msg)
    result = asyncio.run(
        SmartOrchestratorBlock().process({"user_message": msg})
    )
    matched = [m["action"] for m in (result.get("matched_actions") or [])]
    assert "process_document" not in matched, matched
    assert "rfp_management" not in matched, matched
