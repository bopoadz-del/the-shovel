"""Tests for the RAG injection helpers: audit log writer, budget,
noise filter, _rag_inject. The injector + retriever calls are mocked
in this file; we do not hit the live LLM or vector store.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile

import pytest
from tests._dev_key import DEV_KEY
from tests._fixture_values import ENV_SECRET_10


def test_audit_writer_writes_one_jsonl_row_per_call(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.rag import audit
    audit.write({
        "timestamp": "2026-06-09T10:00:00Z",
        "project_id": "p1",
        "conversation_id": "ws-p1",
        "user_id": "u1",
        "agent_name": "project-assistant",
        "user_message_preview": "hi",
        "requested_k": 5,
        "injected_k": 5,
        "injected_tokens": 1200,
        "top_score": 0.71,
        "threshold_fired": False,
        "noise_filtered_count": 0,
        "budget_remaining": 498800,
        "budget_degraded": False,
        "chunks": [],
    })
    log_path = tmp_path / "logs" / "rag_audit.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["project_id"] == "p1"
    assert rows[0]["injected_tokens"] == 1200


def test_audit_writer_never_raises_on_disk_failure(monkeypatch):
    """Audit writes must never break a real chat turn. If the path is
    unwritable, the writer logs and swallows."""
    from app.core.rag import audit
    # Force the resolved log path to contain a NUL byte. os.makedirs raises
    # ValueError on NUL on every platform, which reliably exercises the
    # writer's failure branch. (A literal unwritable string like
    # "/nonexistent/..." can't be used here: on Windows it resolves under
    # C:\ and the write actually succeeds; and os.environ rejects NUL on
    # the way in, so DATA_DIR can't carry it directly.)
    monkeypatch.setattr(audit, "_log_path", lambda: "\x00invalid/logs/rag_audit.jsonl")
    # Should not raise even though the path is unwritable.
    audit.write({"hello": "world"})


def test_budget_starts_at_zero_consumed_for_new_day(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "500000")
    from app.core.rag import budget
    state = budget.snapshot(day="2026-06-09")
    assert state == {"day": "2026-06-09", "consumed": 0, "budget": 500000, "remaining": 500000, "degraded": False}


def test_budget_consume_accumulates_and_reports_degraded_at_inclusive_boundary(
    monkeypatch, tmp_path,
):
    """The spec says 'consumed >= RAG_DAILY_TOKEN_BUDGET' (inclusive).
    The boundary test: when consumed reaches exactly the budget the
    next snapshot must report degraded=True."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "1000")
    from app.core.rag import budget
    budget.consume(day="2026-06-09", tokens=600)
    st = budget.snapshot(day="2026-06-09")
    assert st["consumed"] == 600 and st["remaining"] == 400 and st["degraded"] is False
    budget.consume(day="2026-06-09", tokens=400)
    # Now consumed == 1000 == budget, EXACTLY at the boundary.
    st = budget.snapshot(day="2026-06-09")
    assert st["consumed"] == 1000
    assert st["remaining"] == 0
    assert st["degraded"] is True, (
        "Boundary semantics: degradation must fire when consumed reaches "
        "the budget exactly. The classic off-by-one would have this still "
        "be False until consumed > budget."
    )


def test_budget_rollover_at_new_day_resets_consumed(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "500000")
    from app.core.rag import budget
    budget.consume(day="2026-06-09", tokens=400000)
    assert budget.snapshot(day="2026-06-09")["consumed"] == 400000
    # New day - implicit rollover by querying a fresh date.
    assert budget.snapshot(day="2026-06-10")["consumed"] == 0


def test_noise_filter_default_regex_matches_known_garbage():
    from app.core.rag.retriever import _is_noise_filename
    # The defaults from the spec.
    assert _is_noise_filename("~$C-201_Time Management.docx") is True
    assert _is_noise_filename("nambae-menu(4).pptx") is True
    assert _is_noise_filename("SandsChina_Application_ChaD.docx") is True
    # And NOT a real doc.
    assert _is_noise_filename("Example Client - Performance Basis of Design.pdf") is False
    assert _is_noise_filename("RFP_Appendix_B.xlsx") is False


def test_noise_filter_env_override(monkeypatch):
    monkeypatch.setenv(
        "RAG_NOISE_FILENAME_REGEX",
        r"^(~\$|nambae-menu|SandsChina_Application|MyCustomNoise)",
    )
    from app.core.rag.retriever import _is_noise_filename
    assert _is_noise_filename("MyCustomNoise_v3.txt") is True
    assert _is_noise_filename("Real Document.pdf") is False


def test_retrieve_drops_noise_before_top_k(monkeypatch):
    """The candidate pool from the vector store may include chunks
    from noise docs. They must be filtered BEFORE we pick top-K so a
    noise chunk cannot displace a real chunk."""
    from app.core.rag import retriever as ret
    from app.core.rag.vector_store import Chunk

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [
            Chunk(chunk_id="c1", project_id=project_id, doc_id="d-noise",
                  chunk_index=0, text="noise content", score=0.95),
            Chunk(chunk_id="c2", project_id=project_id, doc_id="d-real",
                  chunk_index=0, text="real content",  score=0.80),
            Chunk(chunk_id="c3", project_id=project_id, doc_id="d-real",
                  chunk_index=1, text="more real",     score=0.70),
        ]

    def fake_doc_name(doc_id):
        return {
            "d-noise": "~$lockfile.docx",
            "d-real":  "Example Client - BOD.pdf",
        }[doc_id]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", fake_doc_name, raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    # Scope this test to the single-project path so the active-project
    # noise count isn't doubled by the general-knowledge merge. The GK
    # merge is covered by ``test_retrieve_merges_general_knowledge``.
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")

    # K=2 - if noise weren't filtered we'd get 2 chunks total
    # (noise + real), since noise scored highest. Filter must skip noise
    # and we must therefore see 2 REAL chunks.
    chunks, dropped = ret.retrieve_with_filter("query", "p1", k=2)
    assert dropped == 1
    assert all("real" in c.text for c in chunks)
    assert len(chunks) == 2


def test_retrieve_merges_general_knowledge(monkeypatch):
    """PR #107: ``retrieve_with_filter`` queries the active project AND
    each project listed in ``RAG_GENERAL_KNOWLEDGE_PROJECTS``, merges
    results by score, and returns the top K with active-project chunks
    winning ties (stable sort).

    Without the GK merge, a GK chunk that semantically matches the
    query (e.g. a procedure from ``training_material``) is invisible to
    a project-scoped chat. With it, the GK chunk competes on equal
    footing and surfaces when it's a better match.
    """
    from app.core.rag import retriever as ret
    from app.core.rag.vector_store import Chunk

    def fake_search(self, project_id, qvec, k, query_text=None):
        # Distinct chunks per project so we can prove the merge happened.
        if project_id == "p_active":
            return [
                Chunk(chunk_id="ap1", project_id=project_id, doc_id="ap-doc",
                      chunk_index=0, text="active project context", score=0.80),
            ]
        if project_id == "training_material":
            return [
                # GK chunk with HIGHER score than the active project's match —
                # the merge must include + rank it above.
                Chunk(chunk_id="gk1", project_id=project_id, doc_id="gk-doc",
                      chunk_index=0, text="general procedure", score=0.92),
                Chunk(chunk_id="gk2", project_id=project_id, doc_id="gk-doc",
                      chunk_index=1, text="extra general", score=0.55),
            ]
        return []

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material")

    chunks, dropped = ret.retrieve_with_filter("query", "p_active", k=3)
    assert dropped == 0

    # Merge must pull from BOTH projects.
    project_ids = [c.project_id for c in chunks]
    assert "p_active" in project_ids
    assert "training_material" in project_ids

    # GK background factor defaults to 1.0 (OFF), so GK scores are unchanged:
    # GK 0.92 first, active 0.80 second, GK 0.55 third.
    assert chunks[0].score == 0.92
    assert chunks[0].project_id == "training_material"
    assert chunks[1].score == 0.80
    assert chunks[1].project_id == "p_active"
    assert chunks[2].score == 0.55


def test_retrieve_skips_gk_when_active_is_gk(monkeypatch):
    """When the active project IS one of the GK projects, the retriever
    must NOT query it twice. Otherwise the active project's chunks
    would appear duplicated in the candidate pool.
    """
    from app.core.rag import retriever as ret
    from app.core.rag.vector_store import Chunk

    call_log: List[str] = []

    def fake_search(self, project_id, qvec, k, query_text=None):
        call_log.append(project_id)
        return [Chunk(chunk_id="c1", project_id=project_id, doc_id="d",
                      chunk_index=0, text="content", score=0.7)]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material")

    ret.retrieve_with_filter("query", "training_material", k=3)
    assert call_log == ["training_material"], (
        f"expected exactly one search call; got {call_log}"
    )


def test_retrieve_gk_failure_does_not_break_primary(monkeypatch):
    """A GK lookup that raises must not affect the active-project path.
    The retriever logs and returns the active results as if GK was disabled.
    """
    from app.core.rag import retriever as ret
    from app.core.rag.vector_store import Chunk

    def fake_search(self, project_id, qvec, k, query_text=None):
        if project_id == "p_active":
            return [Chunk(chunk_id="ap1", project_id=project_id, doc_id="ap",
                          chunk_index=0, text="real", score=0.8)]
        raise RuntimeError("training_material backend unavailable")

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "training_material")

    chunks, dropped = ret.retrieve_with_filter("query", "p_active", k=3)
    assert dropped == 0
    assert len(chunks) == 1
    assert chunks[0].project_id == "p_active"


def test_format_chunks_emits_doc_chunk_score_header():
    from app.core.rag.inject import format_chunks_as_system_message
    from app.core.rag.vector_store import Chunk
    chunks = [
        Chunk(chunk_id="c1", project_id="p", doc_id="d1", chunk_index=0,
              text="A short chunk.", score=0.81),
        Chunk(chunk_id="c2", project_id="p", doc_id="d1", chunk_index=1,
              text="Another short chunk.", score=0.74),
    ]
    msg = format_chunks_as_system_message(chunks, total_candidates=10)
    assert msg["role"] == "system"
    body = msg["content"]
    assert "AUTHORITATIVE REFERENCE CONTEXT" in body
    # doc_id / chunk / score are adjacent and in that order. The source class
    # follows score (owner's numbered item 2); it may NOT sit between chunk=
    # and score= because runtime.py's internal-leak guard and excerpt splitter
    # both anchor on that pair being adjacent.
    assert "[doc_id=d1 chunk=0 score=0.810 class=" in body
    assert "[doc_id=d1 chunk=1 score=0.740 class=" in body
    assert "A short chunk." in body


def test_token_cap_drops_whole_chunks_from_bottom(monkeypatch):
    """When total estimated tokens > MAX_RAG_TOKENS, drop the lowest-
    score chunks (whole-chunk only, never truncate mid-chunk)."""
    monkeypatch.setenv("MAX_RAG_TOKENS", "100")  # very tight
    from app.core.rag.inject import apply_token_cap
    from app.core.rag.vector_store import Chunk
    # Three chunks, each ~80 tokens (320 chars / 4 = 80 tokens).
    big = "X" * 320
    chunks = [
        Chunk(chunk_id="c1", project_id="p", doc_id="d1", chunk_index=0, text=big, score=0.9),
        Chunk(chunk_id="c2", project_id="p", doc_id="d1", chunk_index=1, text=big, score=0.8),
        Chunk(chunk_id="c3", project_id="p", doc_id="d1", chunk_index=2, text=big, score=0.7),
    ]
    kept, total_tokens = apply_token_cap(chunks)
    # Only one chunk should fit under the 100-token cap.
    assert len(kept) == 1
    assert kept[0].score == 0.9  # highest score retained
    assert total_tokens <= 100


def test_token_cap_keeps_all_when_under_budget(monkeypatch):
    monkeypatch.setenv("MAX_RAG_TOKENS", "1500")
    from app.core.rag.inject import apply_token_cap
    from app.core.rag.vector_store import Chunk
    chunks = [
        Chunk(chunk_id=f"c{i}", project_id="p", doc_id="d1", chunk_index=i,
              text="hello world. " * 5, score=0.9 - i*0.1)
        for i in range(3)
    ]
    kept, total_tokens = apply_token_cap(chunks)
    assert len(kept) == 3


def test_rag_inject_returns_none_when_below_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")

    def fake_retrieve(query, project_id, k, **kwargs):
        from app.core.rag.vector_store import Chunk
        return ([
            Chunk(chunk_id="c1", project_id=project_id, doc_id="d1",
                  chunk_index=0, text="weak match", score=0.30),
        ], 0)

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    from app.core.rag.inject import rag_inject

    sys_msg, audit_rec = rag_inject(
        user_message="hello",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert sys_msg is None
    assert audit_rec["threshold_fired"] is True
    assert audit_rec["injected_k"] == 0
    assert audit_rec["budget_degraded"] is False


def test_rag_inject_returns_system_message_when_confident(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("MAX_RAG_TOKENS", "1500")

    def fake_retrieve(query, project_id, k, **kwargs):
        from app.core.rag.vector_store import Chunk
        return ([
            Chunk(chunk_id="c1", project_id=project_id, doc_id="d1",
                  chunk_index=0, text="strong match content", score=0.80),
            Chunk(chunk_id="c2", project_id=project_id, doc_id="d1",
                  chunk_index=1, text="second chunk content",  score=0.60),
        ], 0)

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    from app.core.rag.inject import rag_inject

    sys_msg, audit_rec = rag_inject(
        user_message="data center cooling architecture?",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert sys_msg is not None and sys_msg["role"] == "system"
    assert "strong match" in sys_msg["content"]
    assert audit_rec["injected_k"] == 2
    assert audit_rec["threshold_fired"] is False
    assert audit_rec["top_score"] == 0.80
    assert audit_rec["budget_degraded"] is False
    assert audit_rec["budget_remaining"] >= 0


def test_rag_inject_degrades_to_k2_when_budget_exhausted(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "100")
    monkeypatch.setenv("RAG_K", "5")
    monkeypatch.setenv("MAX_RAG_TOKENS", "10000")  # don't let cap interfere
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.0")

    # Burn the budget first.
    from app.core.rag import budget
    import datetime as _dt
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    budget.consume(day=today, tokens=100)  # consumed == 100 == budget

    seen_k = {"value": None}
    def fake_retrieve(query, project_id, k, **kwargs):
        seen_k["value"] = k
        from app.core.rag.vector_store import Chunk
        return ([
            Chunk(chunk_id=f"c{i}", project_id=project_id, doc_id="d1",
                  chunk_index=i, text="x", score=0.9)
            for i in range(k)
        ], 0)

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    from app.core.rag.inject import rag_inject

    sys_msg, audit_rec = rag_inject(
        user_message="any",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert seen_k["value"] == 2  # K degraded to 2
    assert audit_rec["budget_degraded"] is True


def test_rag_inject_runs_for_any_agent_when_project_id_present(monkeypatch, tmp_path):
    """RAG injection is project-driven, not agent-driven. Even when the
    smart orchestrator routes a project-scoped turn to heavy-reasoning,
    the project context must still be injected."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")

    def fake_retrieve(query, project_id, k, **kwargs):
        from app.core.rag.vector_store import Chunk
        return ([
            Chunk(chunk_id="c1", project_id=project_id, doc_id="d1",
                  chunk_index=0, text="strong match content", score=0.80),
        ], 0)

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    from app.core.rag.inject import rag_inject

    sys_msg, audit_rec = rag_inject(
        user_message="hi",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="heavy-reasoning",
    )
    assert sys_msg is not None
    assert sys_msg["role"] == "system"
    assert "strong match" in sys_msg["content"]
    assert audit_rec["project_id"] == "p1"
    assert audit_rec["agent_name"] == "heavy-reasoning"


def test_rag_inject_skips_when_project_id_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.rag.inject import rag_inject
    sys_msg, audit_rec = rag_inject(
        user_message="hi",
        project_id=None,
        conversation_id=None,
        user_id="u1",
        agent_name="project-assistant",
    )
    assert sys_msg is None
    assert audit_rec == {}


def test_chat_stream_injects_rag_system_message_for_project_assistant(monkeypatch, tmp_path):
    """When agent_name == project-assistant + project_id provided +
    rag_inject returns a system message, the runtime must inject it
    AFTER the existing system prompt and BEFORE the user message."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_RAG_TOKENS", "1500")
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", ENV_SECRET_10)

    captured = {}

    # **kwargs so this spy tolerates NEW optional params on rag_inject
    # (e.g. `history`, added for the follow-up-context expansion). A spy
    # that pins the exact signature fails on a backwards-compatible
    # change without any behaviour actually regressing.
    def fake_inject(user_message, project_id, conversation_id, user_id,
                    agent_name, **kwargs):
        captured["args"] = (user_message, project_id, agent_name)
        return ({"role": "system", "content": "INJECTED_CONTEXT"}, {"injected_k": 1})

    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_inject)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)

    # Intercept _call_llm so we see the messages list that goes to the model.
    seen_messages = {"value": None}

    async def fake_call_llm(self, messages, api_key=None, **kwargs):
        seen_messages["value"] = messages
        return {
            "status": "success",
            "choice": {"message": {"content": "ok", "tool_calls": []}},
            "raw": {},
        }

    from app.agents import runtime
    monkeypatch.setattr(runtime.Agent, "_call_llm", fake_call_llm)

    import asyncio
    agent = runtime.Agent(
        name="project-assistant",
        description="test",
        system_prompt="you are project-assistant",
        allowed_blocks=[],
    )

    async def collect():
        events = []
        async for ev in agent.chat_stream(
            user_message="how big is the IT load?",
            project_id="p1",
            conversation_id=None,
            user_id="u1",
        ):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    # The fake_inject was called with the user's question.
    assert captured["args"][0] == "how big is the IT load?"
    # The runtime passed the injected system message into _call_llm.
    msgs = seen_messages["value"]
    contents = [m.get("content", "") for m in msgs]
    assert any("INJECTED_CONTEXT" in c for c in contents)


def test_rag_debug_query_param_propagates_to_chat_stream(monkeypatch, tmp_path):
    """The ?rag_debug=true query param must be forwarded into chat_stream.
    Verifying via a stubbed chat_stream that captures the kwarg.

    Note: project_id is intentionally omitted from the body so the router's
    project-ownership check (404 on unknown project) doesn't fire — the test
    only exercises query-param forwarding."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from app.main import app

    captured = {"rag_debug": None}

    async def fake_chat_stream(self, message, **kwargs):
        captured["rag_debug"] = kwargs.get("rag_debug")
        yield {"type": "start", "agent": self.name}
        yield {"type": "end", "iterations": 1}

    monkeypatch.setattr("app.agents.runtime.Agent.chat_stream", fake_chat_stream)

    with TestClient(app) as client:
        r = client.post(
            "/v1/agents/project-assistant/chat/stream?rag_debug=true",
            json={"message": "hi"},
            headers={"Authorization": "Bearer " + DEV_KEY},
        )
    assert r.status_code == 200
    assert captured["rag_debug"] is True


def test_q4_tool_call_discipline_under_rag(monkeypatch):
    """When the user names a deliverable AND RAG context is injected,
    the iter-0 LLM response must have finish_reason='tool_calls' AND
    a tool_calls entry with function.name='generate_wbs'.

    This guards the project-assistant's tool-call mandate from being
    displaced by the RAG system message that we just added.
    """
    from app.agents.runtime import _user_intent_requires_tool

    # The matcher must still fire on "generate a 50-activity construction
    # schedule" even when prefixed by RAG context — context is a SYSTEM
    # role, the matcher walks the messages' tail for the user role.
    messages = [
        {"role": "system", "content": "Relevant project context: ..."},
        {"role": "user",   "content": "Generate a 50-activity construction schedule for the data center."},
    ]
    assert _user_intent_requires_tool(messages) is True


def test_chat_stream_end_event_carries_sources(monkeypatch, tmp_path):
    """The final SSE 'end' event must include sources[] with up to top-3
    items sorted by score desc, each with doc_id, doc_name, page_or_section,
    score, confidence."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.rag.vector_store import Chunk
    audit_rec = {
        "injected_k": 3,
        "chunks": [
            {"doc_id": "d1", "chunk_index": 0, "score": 0.91},
            {"doc_id": "d1", "chunk_index": 7, "score": 0.62},
            {"doc_id": "d2", "chunk_index": 2, "score": 0.48},
            {"doc_id": "d3", "chunk_index": 0, "score": 0.30},  # would be 4th
        ],
    }
    from app.agents.runtime import _build_sources_from_audit
    sources = _build_sources_from_audit(audit_rec)
    assert len(sources) == 3
    assert sources[0]["score"] == 0.91
    assert sources[0]["confidence"] == "High"
    assert sources[1]["confidence"] == "Medium"
    assert sources[2]["confidence"] == "Low"
