"""S-R10 b: the evidence loop's gate is the first pass's coverage of the
question, not the question's wording.

Each test builds its own question and first-pass text. Nothing here is a
document, a project or a tester row.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.agents import runtime
from app.core.rag import route
from app.core.rag.inject import _content_terms


def _excerpt(doc_id: str, chunk: int, body: str) -> str:
    return f"[doc_id={doc_id} chunk={chunk} score=0.800 class=project_corpus] {body}"


def _context(*excerpts: str, header: str = "REFERENCE CONTEXT — answer from below.") -> str:
    return header + "\n" + "\n\n".join(excerpts)


# --- the pure coverage check -------------------------------------------------

@pytest.mark.parametrize("question,body", [
    ("Which grade of concrete is used for the raft?",
     "The raft shall be cast in concrete grade C40; sulphate resisting cement is used."),
    ("How long is the curing period for slabs",
     "Slabs: the curing period is seven days, long enough for the membrane."),
])
def test_excerpts_holding_every_term_cover_the_question(question, body):
    terms, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, body)), content_terms=_content_terms,
    )
    assert terms
    assert missing == ()


@pytest.mark.parametrize("question,body,absent", [
    ("Which grade of concrete is used for the raft?",
     "The raft shall be cast in sulphate resisting cement.", {"grade", "concrete", "used"}),
    ("How long is the curing period for slabs",
     "Slabs shall be power floated.", {"long", "curing", "period"}),
])
def test_a_term_absent_from_the_excerpts_is_missing(question, body, absent):
    _, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, body)), content_terms=_content_terms,
    )
    assert set(missing) == absent


@pytest.mark.parametrize("question", [
    "Which grade of concrete is used for the raft?",
    "How long is the curing period for slabs",
])
def test_header_prose_and_markers_never_cover(question):
    # The question's own words sit in the platform header and in a marker's
    # file name; the excerpt body holds none of them.
    header = "REFERENCE CONTEXT for: " + question
    marker_body = "[doc_id=d1 chunk=0 score=0.800 src=" + question.replace(" ", "_") + "] unrelated"
    _, missing = route.question_coverage(
        question, header + "\n" + marker_body, content_terms=_content_terms,
    )
    terms, _ = route.question_coverage(question, "", content_terms=_content_terms)
    assert missing == terms


@pytest.mark.parametrize("question", [
    "Which grade of concrete is used for the raft?",
    "How long is the curing period for slabs",
])
def test_a_platform_note_echoing_the_question_never_covers(question):
    note = "[bill item rows: " + question + "]"
    _, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, "unrelated text\n" + note)),
        content_terms=_content_terms,
    )
    terms, _ = route.question_coverage(question, "", content_terms=_content_terms)
    assert missing == terms


@pytest.mark.parametrize("question,body", [
    ("Where are the piles shown", "Pile caps are detailed on the layout."),
    ("What is the cover to rebar", "Coverage of the rebars is checked."),
])
def test_whole_words_not_substrings(question, body):
    _, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, body)), content_terms=_content_terms,
    )
    assert missing


@pytest.mark.parametrize("question", ["what is it?", "what about it?"])
def test_a_question_with_no_content_terms_is_never_uncovered(question):
    terms, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, "anything")), content_terms=_content_terms,
    )
    assert terms == () and missing == ()


# --- the runtime gate --------------------------------------------------------

def _run_gate(monkeypatch, question: str, first_pass: str, *, excluded: bool = False):
    calls: list[str] = []

    async def fake_augment(**kwargs):
        calls.append(kwargs["question"])
        return kwargs["first_pass_evidence"] + "\n" + _excerpt("d9", 1, "more")

    monkeypatch.setattr(route, "augment_figure_retrieval", fake_augment)
    monkeypatch.setattr(runtime, "_first_pass_composes_figure", lambda *a: False)
    monkeypatch.setattr(runtime, "_agentic_figure_retrieval_excluded", lambda t: excluded)
    monkeypatch.setattr(runtime, "_bill_led_rag_msg", lambda q, p, m: m)
    monkeypatch.setattr(
        "app.core.rag.vector_store.get_store",
        lambda: SimpleNamespace(bm25_search=lambda *a, **k: []),
    )
    out = asyncio.run(runtime._agentic_figure_retrieval_augment(
        agent=SimpleNamespace(), user_message=question, project_id="p",
        api_key="k", user_id="u", rag_sys_msg={"role": "system", "content": first_pass},
    ))
    return calls, out


# Interrogative shapes the old noun whitelist never let through: "which",
# "how long", no figure noun.
_UNGATED_SHAPES = [
    ("Which grade of concrete is used for the raft?",
     "Concrete grade C40 is used for the raft."),
    ("How long is the curing period for slabs",
     "Slabs: the curing period is seven days, long enough."),
]


@pytest.mark.parametrize("question,covering_body", _UNGATED_SHAPES)
def test_an_uncovered_first_pass_enters_the_loop_whatever_the_wording(
    question, covering_body, monkeypatch,
):
    first_pass = _context(_excerpt("d1", 0, "Formwork shall be plywood."))
    calls, out = _run_gate(monkeypatch, question, first_pass)
    assert calls == [question]
    assert "doc_id=d9" in out["content"]


@pytest.mark.parametrize("question,covering_body", _UNGATED_SHAPES)
def test_a_covering_first_pass_skips_the_plan_call(question, covering_body, monkeypatch):
    first_pass = _context(_excerpt("d1", 0, covering_body))
    calls, out = _run_gate(monkeypatch, question, first_pass)
    assert calls == []
    assert out["content"] == first_pass


@pytest.mark.parametrize("question,covering_body", _UNGATED_SHAPES)
def test_exclusions_still_keep_the_loop_out(question, covering_body, monkeypatch):
    first_pass = _context(_excerpt("d1", 0, "Formwork shall be plywood."))
    calls, out = _run_gate(monkeypatch, question, first_pass, excluded=True)
    assert calls == []
    assert out["content"] == first_pass
