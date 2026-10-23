"""Formula-style asks on a fixture project must not go no-tool / Master Corpus.

Phase 2 live: a formula audit fixture project received asks such as
rebar lap, pe_unit_convert, and formwork striking. The turn often
(1) answered without calling construction_calc, and/or (2) cited the
Master Corpus fallback instead of staying on the active project_id.

This file is synthetic only — no client names, no live fixture ids.
Agent A owns no-tool + master_corpus bleed (routing / runtime / inject).
"""
from __future__ import annotations

import pytest

from tests._dev_key import DEV_KEY
from app.agents.runtime import (
    _MASTER_CORPUS_FALLBACK_NOTE,
    _apply_rag_context,
    _build_sources_from_audit,
    _forced_specific_tool,
    _message_wants_named_calculator,
    _postprocess_answer,
    _predispatch_formula_calc,
    _project_has_non_rag_context,
    project_is_user_fixture,
    should_suppress_master_corpus_fallback,
)


FIXTURE_PID = "proj_formula_fixture_scope"
MASTER_PID = "master_src_formula_scope"
GK_PID = "curated_kb_formula_scope"
# Live UI project id is a hex slug; the name carries FIXTURE-.
# Synthetic id/name only — no live fixture identifiers.
HEX_FIXTURE_PID = "aa11bb22"
HEX_FIXTURE_NAME = "FIXTURE-c-formula-pack"

AVAILABLE = {"construction_calc", "search_project_documents"}

# Formula-shaped asks that do not carry L×W×D. Includes the live
# phone/UI phrasings (synthetic — no client names).
SHORT_UI_ASKS = (
    "rebar lap",
    "pe_unit_convert",
    "slab formwork striking",
)
FORMULA_ASKS = (
    "compute rebar lap for 20 mm bar fy 420",
    "What is the typical rebar lap length for 16mm bars in tension?",
    "pe_unit_convert 10 m to ft",
    "Convert 150 pe using pe_unit_convert",
    "formwork striking time for a slab",
    "What is the formwork striking time for a slab?",
) + SHORT_UI_ASKS

LOOKUP_ASKS = (
    "what is the backfilling specification for soft ground",
    "what is the concrete volume in the BOQ",
    "list the documents in this project",
)

# Agent C Phase-2 re-probe on empty FIXTURE-c (synthetic asks only).
# plumbing_flow_programme is a schedule-builder path — not in this set.
REPROBE_BLEED_ASKS = (
    "What is the rebar lap length for 16mm bars in tension?",
    "Convert 150 pe using pe_unit_convert",
    "What is the modulus of elasticity of concrete for fck 30 MPa?",
    "What is a typical productivity rate for concrete pouring m3 per crew-day?",
    "roi_calculator gain 1200000 cost 1000000",
    "What is the unit weight of reinforced concrete?",
)
PLUMBING_PROGRAMME_ASK = (
    "Build a plumbing flow programme for a 20-storey tower"
)


def _tail(text: str):
    return [{"role": "user", "content": text}]


def _chunk(chunk_id, project_id, score, text="filler content"):
    from app.core.rag.vector_store import Chunk
    return Chunk(
        chunk_id=chunk_id,
        project_id=project_id,
        doc_id=f"doc-{chunk_id}",
        chunk_index=0,
        text=text,
        score=score,
    )


def _install_empty_fixture_with_master(monkeypatch, *, record=None):
    """Empty fixture project + a high-scoring Master Corpus row.

    Same harness as tests/test_step0_retrieval_isolation.py — fake
    VectorStore.search so scores are exact and no live embedder is needed.
    """
    from app.core.rag import retriever as ret

    per_project = {
        FIXTURE_PID: [],
        MASTER_PID: [_chunk("m_bleed", MASTER_PID, 0.95, "generic formula notes")],
        GK_PID: [],
    }

    def fake_search(self, project_id, qvec, k, query_text=None):
        if record is not None:
            record.append(project_id)
        return list(per_project.get(project_id, []))

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK_PID)
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", MASTER_PID)
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    for var in (
        "RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
        "RAG_GK_LEXICAL_FOLD",
    ):
        monkeypatch.delenv(var, raising=False)
    return ret


# ── 1. formula-style ask must route to construction_calc ──────────────────


@pytest.mark.parametrize("q", FORMULA_ASKS)
def test_formula_style_ask_forces_construction_calc(q):
    """A formula ask on any project must force the calculator tool.

    Live miss: project-assistant answered in prose (no-tool) because
    tool_choice stayed auto and the dimension heuristic never fired.
    """
    assert (
        _forced_specific_tool(_tail(q), AVAILABLE) == "construction_calc"
    ), q


@pytest.mark.parametrize("q", LOOKUP_ASKS)
def test_document_lookups_are_not_forced_onto_the_calculator(q):
    """Lookups stay on RAG. Do not steal BOQ / spec questions."""
    assert _forced_specific_tool(_tail(q), AVAILABLE) is None, q


def test_schedule_critical_path_lookup_is_not_named_calculator():
    """CI: 'What is the critical path of this schedule?' must stay a
    schedule lookup (heavy-reasoning / primavera), not construction_calc."""
    q = "What is the critical path of this schedule?"
    assert _message_wants_named_calculator(q) is False
    assert _forced_specific_tool(_tail(q), AVAILABLE) is None


@pytest.mark.parametrize("q", REPROBE_BLEED_ASKS)
def test_reprobe_bleed_asks_force_construction_calc(q):
    """FIXTURE-c re-probe: these six must call the calculator, not RAG."""
    from app.agents.runtime import _message_is_formula_style_ask

    assert _message_is_formula_style_ask(q), q
    assert _message_wants_named_calculator(q), q
    assert _forced_specific_tool(_tail(q), AVAILABLE) == "construction_calc", q


def test_plumbing_flow_programme_is_not_stolen_onto_the_calculator():
    """Schedule-builder path. intent_map 'plumbing flow' must not win."""
    q = PLUMBING_PROGRAMME_ASK
    assert _message_wants_named_calculator(q) is False
    assert _forced_specific_tool(
        _tail(q), AVAILABLE | {"generate_wbs"},
    ) != "construction_calc"


def test_formula_force_requires_the_tool_to_be_available():
    q = FORMULA_ASKS[0]
    assert _forced_specific_tool(_tail(q), {"search_project_documents"}) is None


# ── 2. retrieved / cited sources stay on the fixture project_id ───────────


@pytest.mark.parametrize("q", FORMULA_ASKS)
def test_formula_ask_on_fixture_does_not_query_master_corpus(monkeypatch, q):
    asked: list = []
    ret = _install_empty_fixture_with_master(monkeypatch, record=asked)
    chunks, _ = ret.retrieve_with_filter(q, FIXTURE_PID, k=5)
    assert MASTER_PID not in asked, (
        f"formula ask queried Master Corpus fallback: {asked}"
    )
    assert all(getattr(c, "layer", "own") != "master_corpus" for c in chunks)
    assert all(c.project_id != MASTER_PID for c in chunks)


@pytest.mark.parametrize("q", FORMULA_ASKS)
def test_rag_inject_formula_ask_on_fixture_does_not_bleed_master(monkeypatch, q):
    _install_empty_fixture_with_master(monkeypatch)
    from app.core.rag import inject as inj

    sys_msg, audit = inj.rag_inject(
        user_message=q,
        project_id=FIXTURE_PID,
        conversation_id=None,
        user_id=None,
        agent_name="project-assistant",
    )
    assert audit.get("project_id") == FIXTURE_PID
    assert audit.get("fallback_used") is not True
    chunks = audit.get("chunks") or []
    assert all(c.get("layer") != "master_corpus" for c in chunks)
    assert all(c.get("project_id") != MASTER_PID for c in chunks)
    sources = _build_sources_from_audit(audit, final_text="")
    assert all(s.get("layer") != "master_corpus" for s in sources)
    assert all(s.get("project_id") != MASTER_PID for s in sources)
    # Empty fixture + no fallback: nothing to fold. That is correct —
    # the calculator tool is the answer path, not Master Corpus prose.
    if sys_msg and sys_msg.get("content"):
        assert "master_corpus" not in (sys_msg.get("content") or "").lower()


def test_document_lookup_on_empty_project_does_not_use_master_fallback(monkeypatch):
    """Project-own rows of another project are never fetched, lookups included."""
    asked: list = []
    ret = _install_empty_fixture_with_master(monkeypatch, record=asked)
    chunks, _ = ret.retrieve_with_filter(LOOKUP_ASKS[0], FIXTURE_PID, k=5)
    assert MASTER_PID not in asked
    assert all(c.project_id != MASTER_PID for c in chunks)
    assert all(getattr(c, "layer", "own") != "master_corpus" for c in chunks)


# ── 3. RAG fold must not clamp a formula ask to corpus-only / no-tool ─────


@pytest.mark.parametrize("q", FORMULA_ASKS)
def test_formula_ask_is_not_told_to_answer_only_from_corpus(q):
    """The strict lookup clamp is why the model answered no-tool from RAG.

    `_apply_rag_context` used to treat 'rebar lap' as a document lookup
    and append 'using ONLY the reference context' — so when fallback
    excerpts were present the model quoted Master Corpus and never
    called construction_calc.
    """
    context = {
        "content": (
            "[doc_id=abcd chunk=0] master_corpus notes about generic laps"
        ),
    }
    msgs = [{"role": "user", "content": q}]
    assert _apply_rag_context(msgs, context) is True
    folded = msgs[-1]["content"]
    assert "using ONLY the reference context" not in folded
    assert "CALCULATION REQUEST" in folded
    assert q in folded


@pytest.mark.parametrize("q", FORMULA_ASKS)
def test_formula_ask_is_named_calculator_not_below_routing_gate(q):
    """Live: 'typical rebar lap' routed reason=below_routing_gate."""
    assert _message_wants_named_calculator(q), q


@pytest.mark.parametrize("q", FORMULA_ASKS)
def test_empty_fixture_formula_ask_is_not_an_unindexed_refusal(q):
    """Empty FIXTURE-a must reach construction_calc, not the no-docs abort."""
    assert _project_has_non_rag_context(FIXTURE_PID, q) is True


@pytest.mark.parametrize("q", FORMULA_ASKS)
def test_formula_ask_does_not_wear_master_corpus_banner(q):
    """Live pe_unit_convert: tool ran and the answer still opened with MC."""
    msgs = [{"role": "user", "content": q}]
    out = _postprocess_answer(
        "calculator result line", None, msgs, fallback_used=True,
    )
    assert _MASTER_CORPUS_FALLBACK_NOTE.strip() not in out


def test_lookup_still_wears_master_corpus_banner_when_fallback_used():
    msgs = [{"role": "user", "content": LOOKUP_ASKS[0]}]
    out = _postprocess_answer(
        "Here is the spec.", None, msgs, fallback_used=True,
    )
    assert out.startswith(_MASTER_CORPUS_FALLBACK_NOTE)


@pytest.mark.asyncio
@pytest.mark.parametrize("q", (
    "What is the typical rebar lap length for 16mm bars in tension?",
    "What is the formwork striking time for a slab?",
    "Convert 150 pe using pe_unit_convert",
))
async def test_named_calculator_route_actually_invokes_construction_calc(q):
    """Kimi/Groq leave tool_choice=auto; predispatch must still call the tool."""
    from app.agents.runtime import Agent

    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    msgs = [{"role": "user", "content": q}]
    rec = await _predispatch_formula_calc(
        agent, msgs, FIXTURE_PID, operator_text=q,
    )
    assert rec is not None, q
    assert rec["name"] == "construction_calc"
    assert rec.get("predispatched") is True
    assert any(
        "PLATFORM PRE-DISPATCH: construction_calc" in str(m.get("content") or "")
        for m in msgs
    )


# Exact live gate-tester strings (empty FIXTURE-a). Synthetic project_id.
LIVE_REBAR_LAP = (
    "What is the typical rebar lap length for 16mm bars in tension?"
)
LIVE_PE_CONVERT = "Convert 150 pe using pe_unit_convert"
LIVE_FORMWORK_SLAB = "What is the formwork striking time for a slab?"
LIVE_GATE = (LIVE_REBAR_LAP, LIVE_PE_CONVERT, LIVE_FORMWORK_SLAB)


@pytest.mark.parametrize("q", LIVE_GATE)
def test_live_gate_strings_are_named_calculator_not_below_routing_gate(q):
    """Live #1 was reason=below_routing_gate; #3 was named_calculator."""
    from app.agents.runtime import _message_is_formula_style_ask

    assert _message_is_formula_style_ask(q), q
    assert _message_wants_named_calculator(q), q
    assert _forced_specific_tool(_tail(q), AVAILABLE) == "construction_calc", q


@pytest.mark.asyncio
@pytest.mark.parametrize("q", LIVE_GATE)
async def test_live_empty_fixture_chat_calls_calc_without_mc_bleed(
    q, tmp_path, monkeypatch,
):
    """Full chat path: LLM answers no-tool prose; predispatch must still
    put construction_calc on the turn and must not emit the MC banner.
    """
    from app.agents.runtime import Agent, _MASTER_CORPUS_FALLBACK_NOTE

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: False,
    )
    _install_empty_fixture_with_master(monkeypatch)

    async def _llm_no_tool(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": (
                        "This project has no documents of its own for this "
                        "question — answering from the Master Corpus. "
                        "Typical value from reference notes."
                    ),
                },
            },
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", _llm_no_tool)
    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    out = await agent.chat(q, api_key=DEV_KEY, project_id=FIXTURE_PID)
    assert out["status"] == "success", out
    names = [t.get("name") for t in (out.get("tool_calls") or [])]
    assert "construction_calc" in names, names
    answer = out.get("answer") or ""
    assert _MASTER_CORPUS_FALLBACK_NOTE.strip() not in answer
    sources = out.get("sources") or []
    assert all(s.get("layer") != "master_corpus" for s in sources)
    assert all(s.get("project_id") != MASTER_PID for s in sources)


def _install_hex_fixture_name(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_project",
        lambda pid, user_id=None, **kw: (
            {"id": pid, "name": HEX_FIXTURE_NAME}
            if pid == HEX_FIXTURE_PID
            else None
        ),
    )


def test_hex_pid_with_fixture_name_is_a_user_fixture(monkeypatch):
    """Live UI id is hex; FIXTURE- lives on the project name."""
    _install_hex_fixture_name(monkeypatch)
    assert project_is_user_fixture(HEX_FIXTURE_PID) is True
    assert project_is_user_fixture("master_corpus") is False
    assert project_is_user_fixture(MASTER_PID) is False


@pytest.mark.parametrize("q", SHORT_UI_ASKS)
def test_short_ui_ask_on_hex_fixture_suppresses_master_corpus(monkeypatch, q):
    _install_hex_fixture_name(monkeypatch)
    assert should_suppress_master_corpus_fallback(HEX_FIXTURE_PID, q) is True
    assert should_suppress_master_corpus_fallback("master_corpus", q) is False


@pytest.mark.parametrize("q", SHORT_UI_ASKS)
def test_banner_recovers_short_ask_from_audit_when_messages_lose_it(q):
    """Live pe_unit_convert: tool ran; postprocess still wore the MC banner
    because the last user bubble was predispatch / folded and the ask
    was only in the RAG audit preview."""
    out = _postprocess_answer(
        "calculator result line",
        None,
        [{"role": "user", "content": "PLATFORM PRE-DISPATCH: construction_calc"}],
        fallback_used=True,
        project_id=HEX_FIXTURE_PID,
        audit_rec={
            "project_id": HEX_FIXTURE_PID,
            "user_message_preview": q,
        },
    )
    assert _MASTER_CORPUS_FALLBACK_NOTE.strip() not in out
    assert "Master Corpus" not in out


@pytest.mark.parametrize("q", SHORT_UI_ASKS)
def test_model_written_mc_preamble_is_stripped_on_fixture_formula(q, monkeypatch):
    _install_hex_fixture_name(monkeypatch)
    model = (
        "_This project has no documents of its own for this question — "
        "answering from the Master Corpus._\n\n"
        "Typical lap is 40d from reference notes."
    )
    out = _postprocess_answer(
        model,
        None,
        [{"role": "user", "content": q}],
        fallback_used=True,
        project_id=HEX_FIXTURE_PID,
        audit_rec={"project_id": HEX_FIXTURE_PID, "user_message_preview": q},
    )
    assert "Master Corpus" not in out
    assert "Typical lap is 40d" in out


@pytest.mark.parametrize("q", SHORT_UI_ASKS)
def test_inject_strips_leaked_mc_on_hex_fixture_formula(monkeypatch, q):
    """Even if retrieve leaked a tagged fallback chunk, inject must drop
    it for a user FIXTURE calculator ask (original user_message, not the
    expanded retrieval_query)."""
    from app.core.rag import inject as inj

    leaked = [_chunk("m_bleed", MASTER_PID, 0.95, "generic formula notes")]
    leaked[0].layer = "master_corpus"

    def fake_retrieve(query, project_id, k=5, **kwargs):
        return leaked, 0

    monkeypatch.setattr(inj, "retrieve_with_filter", fake_retrieve)
    _install_hex_fixture_name(monkeypatch)
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", MASTER_PID)
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")

    sys_msg, audit = inj.rag_inject(
        user_message=q,
        project_id=HEX_FIXTURE_PID,
        conversation_id=None,
        user_id=None,
        agent_name="project-assistant",
    )
    assert audit.get("fallback_used") is not True
    chunks = audit.get("chunks") or []
    assert all(c.get("layer") != "master_corpus" for c in chunks)
    assert all(c.get("project_id") != MASTER_PID for c in chunks)
    if sys_msg and sys_msg.get("content"):
        assert "master_corpus" not in (sys_msg.get("content") or "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("q", SHORT_UI_ASKS)
async def test_short_ui_named_calculator_invokes_construction_calc(q):
    from app.agents.runtime import Agent

    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    msgs = [{"role": "user", "content": q}]
    rec = await _predispatch_formula_calc(
        agent, msgs, HEX_FIXTURE_PID, operator_text=q,
    )
    assert rec is not None, q
    assert rec["name"] == "construction_calc"
    assert rec.get("predispatched") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("q", SHORT_UI_ASKS)
async def test_hex_fixture_chat_calls_calc_without_mc_bleed(
    q, tmp_path, monkeypatch,
):
    from app.agents.runtime import Agent, _MASTER_CORPUS_FALLBACK_NOTE

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: False,
    )
    _install_empty_fixture_with_master(monkeypatch)
    _install_hex_fixture_name(monkeypatch)

    async def _llm_no_tool(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": (
                        "This project has no documents of its own for this "
                        "question — answering from the Master Corpus. "
                        "Typical value from reference notes."
                    ),
                },
            },
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", _llm_no_tool)
    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    out = await agent.chat(q, api_key=DEV_KEY, project_id=HEX_FIXTURE_PID)
    assert out["status"] == "success", out
    names = [t.get("name") for t in (out.get("tool_calls") or [])]
    assert "construction_calc" in names, names
    answer = out.get("answer") or ""
    assert _MASTER_CORPUS_FALLBACK_NOTE.strip() not in answer
    assert "answering from the Master Corpus" not in answer
    sources = out.get("sources") or []
    assert all(s.get("layer") != "master_corpus" for s in sources)
    assert all(s.get("project_id") != MASTER_PID for s in sources)


@pytest.mark.asyncio
@pytest.mark.parametrize("q", REPROBE_BLEED_ASKS)
async def test_reprobe_empty_fixture_chat_calls_calc_without_mc_bleed(
    q, tmp_path, monkeypatch,
):
    """Re-probe six: construction_calc must appear; MC banner must not."""
    from app.agents.runtime import Agent, _MASTER_CORPUS_FALLBACK_NOTE

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: False,
    )
    _install_empty_fixture_with_master(monkeypatch)

    async def _llm_no_tool(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": (
                        "_This project has no documents of its own for this "
                        "question — answering from the Master Corpus._ "
                        "I have to stop here and report the failure plainly."
                    ),
                },
            },
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", _llm_no_tool)
    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    out = await agent.chat(q, api_key=DEV_KEY, project_id=FIXTURE_PID)
    assert out["status"] == "success", out
    names = [t.get("name") for t in (out.get("tool_calls") or [])]
    assert "construction_calc" in names, names
    assert names != ["search_project_documents"], names
    answer = out.get("answer") or ""
    assert _MASTER_CORPUS_FALLBACK_NOTE.strip() not in answer
    assert "answering from the Master Corpus" not in answer
    sources = out.get("sources") or []
    assert all(s.get("layer") != "master_corpus" for s in sources)
    assert all(s.get("project_id") != MASTER_PID for s in sources)


def test_calc_already_fired_strips_mc_banner_even_without_ask():
    """Live pe_unit_convert: tool ran; postprocess lost the ask and
    still prepended the Master Corpus banner."""
    out = _postprocess_answer(
        (
            "_This project has no documents of its own for this question — "
            "answering from the Master Corpus._ "
            "The pe_unit_convert calculator failed."
        ),
        None,
        [{
            "role": "user",
            "content": "PLATFORM PRE-DISPATCH: construction_calc has ALREADY been run",
        }],
        fallback_used=True,
        project_id=FIXTURE_PID,
        audit_rec={"project_id": FIXTURE_PID, "user_message_preview": ""},
    )
    assert "Master Corpus" not in out
    # The failure sentence stays; the registry id is the display name.
    from app.lib.source_labels import formula_display_name
    shown = formula_display_name("pe_unit_convert")
    assert shown and shown in out
    assert "calculator failed" in out
    assert "pe_unit_convert" not in out
