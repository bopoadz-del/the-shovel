"""ACI one-way slab thickness is a registered calculator, not a corpus claim.

Live tip d708b5d, six fresh chats, deepseek-flash:

- T20 simply supported 4.8 m → 240 mm (L/20). 1/6. The pass is the only
  run that invoked construction_calc. The other five refused with no tool
  call ("I don't have that in the retrieved excerpts").
- U5 simply supported 8.4 m → 420 mm (L/20). 0/6. Named-standard gate
  ("ACI 318-19 is not in the retrieved excerpts … I cannot state what it
  requires").
- E13 one-end continuous 6.0 m → 250 mm (L/24). 0/6. Same gate ("no
  retrieved excerpt is that document").

In several runs the model named slab_thickness_min and offered to run it,
then ended the turn without a tool call.

The attribution guard applies to retrieved claims. A question a registered
calculator can answer is not refused for corpus absence. Provenance is the
formula plus the calculator's own standard field.

Fixture wording only. No live client names. No network LLM calls.
"""
from __future__ import annotations

import re

import pytest

from app.agents.runtime import (
    Agent,
    _message_is_formula_style_ask,
    _postprocess_answer,
    _predispatch_formula_calc,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk
from tests._dev_key import DEV_KEY


T20 = (
    "What is the ACI minimum thickness of a simply supported one-way "
    "solid slab spanning 4.8 m?"
)
U5 = (
    "To ACI 318-19, minimum thickness of a simply supported one-way "
    "solid slab spanning 8.4 m, fy 420, normal weight — answer in mm."
)
E13 = (
    "To ACI 318-19, what is the minimum thickness of a non-prestressed, "
    "normal-weight, one-way solid slab with one end continuous, spanning "
    "6.0 m, fy = 420 MPa, not supporting partitions liable to deflection "
    "damage?"
)

# Truth from ACI 318-19 Table 7.3.1.1 at fy = 420 (modifier 1.0).
CASES = (
    pytest.param(T20, 240.0, id="T20"),
    pytest.param(U5, 420.0, id="U5"),
    pytest.param(E13, 250.0, id="E13"),
)

CALC_STANDARD = "ACI 318-19 Table 7.3.1.1"

# The live refusal shape. Synthetic — no contract ids, no party names.
GUARD_REFUSAL = (
    "I don't have that in the retrieved excerpts for this question. "
    "ACI 318-19 is not in the retrieved excerpts, so I cannot state "
    "what it requires. I won't attribute a figure to it."
)

# Live failure: the model names the calculator and stops.
ANNOUNCE_NO_CALL = (
    "I don't have that in the retrieved excerpts for this question. "
    "I can run slab_thickness_min for this span. "
    "Would you like me to run the calculator?"
)

_GUARD_RE = re.compile(
    r"not in the retrieved excerpts|cannot state what|"
    r"won'?t attribute|will not attribute|"
    r"would you like me to run",
    re.IGNORECASE,
)

# A named code with no registered calculator. Same shape as P6b, but
# ACI 318 so a slab-thickness bypass cannot open every ACI 318 claim.
NO_CALC_ASK = (
    "Per ACI 318-19, what is the maximum temperature of fresh concrete "
    "at placing?"
)
NO_CALC_BAD = (
    "ACI 318-19 limits the maximum temperature of fresh concrete to 32°C."
)
NO_CALC_PROJECT = (
    "Project concrete specification. The maximum temperature of fresh "
    "concrete at the point of delivery is 32°C."
)

FIXTURE_PID = "proj_slab_calc_guard_fixture"
MASTER_PID = "master_src_slab_calc_guard"
GK_PID = "curated_kb_slab_calc_guard"

# Cites the code the way a project schedule does. The filename is not
# the code, and the layer is not knowledge-base, so the named-standard
# gate still treats ACI 318 as absent. The token 318-19 is present so
# the identifier RAG-miss short-circuit does not eat the turn first.
PROJECT_SCHEDULE = (
    "Project structural schedule. General concrete notes cite ACI 318-19 "
    "as a reference only. Concrete slab on grade not exceeding 150 mm. "
    "Raft slab 300 to 500 mm. Suspended slab 150 to 300 mm."
)


def _chunk(cid, doc_id, text, *, source_name="", layer="own", score=0.81):
    return Chunk(
        chunk_id=cid,
        project_id=FIXTURE_PID,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
        source_name=source_name,
        layer=layer,
    )


def _install_project_schedule(monkeypatch):
    """One project excerpt that is not ACI 318. No live embedder."""
    from app.core.rag import retriever as ret

    schedule = _chunk(
        "sched", "sched1", PROJECT_SCHEDULE,
        source_name="project-structural-schedule.txt",
    )

    def fake_search(self, project_id, qvec, k, query_text=None):
        if project_id == FIXTURE_PID:
            return [schedule]
        return []

    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.search", fake_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "schedule.txt", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK_PID)
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", MASTER_PID)
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    for var in (
        "RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
        "RAG_GK_LEXICAL_FOLD",
    ):
        monkeypatch.delenv(var, raising=False)
    from app.core.rag.embeddings import reset_embedder_cache
    reset_embedder_cache()
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True,
    )


def _presents_number_as_standard(text: str, standard: str, number: str) -> bool:
    std = re.compile(re.escape(standard), re.IGNORECASE)
    num = re.compile(rf"\b{re.escape(number)}\b")
    disclaimer = re.compile(
        r"project[-\s]?only|project requirement",
        re.IGNORECASE,
    )
    not_the_code = re.compile(
        rf"\bnot\b.{{0,80}}{re.escape(standard)}",
        re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        if not std.search(sentence) or not num.search(sentence):
            continue
        if disclaimer.search(sentence) or not_the_code.search(sentence):
            continue
        return True
    return False


def _calc_calls(out: dict) -> list[dict]:
    return [
        t for t in (out.get("tool_calls") or [])
        if t.get("name") == "construction_calc"
    ]


def _assert_slab_answer(out: dict, mm: float) -> None:
    calls = _calc_calls(out)
    assert calls, (
        "construction_calc was not invoked; tool_calls="
        f"{out.get('tool_calls')!r} answer={out.get('answer')!r}"
    )
    result = calls[0].get("result") or {}
    assert result.get("calculation") == "slab_thickness_min", result
    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    assert inner.get("min_thickness_mm") == pytest.approx(mm), result
    assert inner.get("standard") == CALC_STANDARD, result
    answer = out.get("answer") or ""
    shown = str(int(mm)) if float(mm) == int(mm) else str(mm)
    assert re.search(rf"\b{shown}\s*mm\b", answer), answer
    assert CALC_STANDARD in answer, answer
    assert not _GUARD_RE.search(answer), answer


async def _chat_with_model_text(monkeypatch, question: str, model_text: str) -> dict:
    _install_project_schedule(monkeypatch)

    async def _llm_no_tool(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": model_text}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", _llm_no_tool)
    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    return await agent.chat(
        question, api_key=DEV_KEY, project_id=FIXTURE_PID,
    )


@pytest.mark.parametrize("question,mm", CASES)
async def test_aci_slab_thickness_invokes_slab_thickness_min(
    monkeypatch, question, mm,
):
    """A guard refusal with no tool call must not ship.

    The mocked model returns the live refusal and makes no tool call.
    construction_calc (slab_thickness_min) still runs, and the answer
    states the calculator thickness with the calculator's standard.
    """
    out = await _chat_with_model_text(monkeypatch, question, GUARD_REFUSAL)
    assert out["status"] == "success", out
    _assert_slab_answer(out, mm)


@pytest.mark.parametrize("question,mm", CASES)
async def test_announced_slab_thickness_min_is_still_invoked(
    monkeypatch, question, mm,
):
    """Naming slab_thickness_min without calling it is not an answer."""
    out = await _chat_with_model_text(monkeypatch, question, ANNOUNCE_NO_CALL)
    assert out["status"] == "success", out
    _assert_slab_answer(out, mm)
    answer = (out.get("answer") or "").lower()
    assert "would you like me to run" not in answer


@pytest.mark.parametrize("question", (T20, U5, E13))
def test_slab_ask_is_a_formula_not_a_lookup(question):
    assert _message_is_formula_style_ask(question), question


@pytest.mark.parametrize("question,mm", CASES)
async def test_predispatch_runs_slab_thickness_min(question, mm):
    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    msgs = [{"role": "user", "content": question}]
    rec = await _predispatch_formula_calc(
        agent, msgs, FIXTURE_PID, operator_text=question,
    )
    assert rec is not None, question
    assert rec["name"] == "construction_calc"
    result = rec.get("result") or {}
    assert result.get("calculation") == "slab_thickness_min", result
    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    assert inner.get("min_thickness_mm") == pytest.approx(mm), result
    assert inner.get("standard") == CALC_STANDARD, result


@pytest.mark.parametrize("question", (U5, E13))
def test_numbered_code_on_a_calculator_ask_is_not_an_absence_refusal(question):
    """U5/E13 name ACI 318-19. The steering must not forbid the calculator."""
    chunks = [
        _chunk(
            "sched", "sched1", PROJECT_SCHEDULE,
            source_name="project-structural-schedule.txt",
        ),
    ]
    content = format_chunks_as_system_message(
        chunks, len(chunks), query=question,
    )["content"]
    assert "NAMED STANDARD ABSENT" not in content, content
    assert "do not state what it requires" not in content.lower(), content
    assert "slab_thickness_min" in content


def test_named_standard_without_a_calculator_is_still_gated():
    """ACI 318 fresh-concrete temperature has no calculator. P6b shape stays."""
    chunks = [
        _chunk(
            "conc", "conc1", NO_CALC_PROJECT,
            source_name="project-concrete-spec.txt",
        ),
    ]
    rag = format_chunks_as_system_message(chunks, len(chunks), query=NO_CALC_ASK)
    assert "NAMED STANDARD ABSENT" in rag["content"]
    assert "slab_thickness_min" not in rag["content"]
    out = _postprocess_answer(
        NO_CALC_BAD,
        rag,
        [{"role": "user", "content": NO_CALC_ASK}],
    )
    assert not _presents_number_as_standard(out, "ACI 318", "32"), out
    assert re.search(r"not in the retrieved excerpts", out, re.IGNORECASE), out
    assert re.search(r"project-only", out, re.IGNORECASE), out
    assert "32" in out
    assert not _message_is_formula_style_ask(NO_CALC_ASK)
