"""Tests for the agentic figure-retrieval route hook (item 2)."""
from __future__ import annotations

import pytest

from app.core.rag.evidence_loop import run_project_figure_retrieval
from app.core.rag.route import augment_figure_retrieval, wants_agentic_figure_retrieval


def _asks_measured_true(_text: str) -> bool:
    return True


def _asks_measured_false(_text: str) -> bool:
    return False


def _never_exclude(_text: str) -> bool:
    return False


def _exclude_calculate(text: str) -> bool:
    return "calculate" in text.lower()


@pytest.mark.parametrize("message", [
    "what is the minimum cover to reinforcement for the raft",
    "what compaction density does the sub-grade require",
])
def test_measured_quantity_ask_triggers(message):
    assert wants_agentic_figure_retrieval(
        message, asks_measured_quantity=_asks_measured_true, exclude=_never_exclude,
    )


@pytest.mark.parametrize("message", [
    "what is the standard rate for blockwork walls",
    "how much productivity should I allow for wall plaster",
])
def test_value_or_rate_ask_triggers(message):
    assert wants_agentic_figure_retrieval(
        message, asks_measured_quantity=_asks_measured_false, exclude=_never_exclude,
    )


@pytest.mark.parametrize("message", [
    "generate the payment certificate for this month",
    "summarise the contract for me",
])
def test_non_figure_ask_does_not_trigger(message):
    assert not wants_agentic_figure_retrieval(
        message, asks_measured_quantity=_asks_measured_false, exclude=_never_exclude,
    )


@pytest.mark.parametrize("message", [
    "calculate the volume of a 3m x 2m x 1m trench",
    "please calculate 5 plus 7",
])
def test_excluded_ask_does_not_trigger(message):
    assert not wants_agentic_figure_retrieval(
        message, asks_measured_quantity=_asks_measured_false, exclude=_exclude_calculate,
    )


@pytest.mark.parametrize("message", [
    "what is the standard rate for blockwork walls",
    "how much productivity should I allow for wall plaster",
])
def test_excluded_figure_ask_does_not_trigger(message):
    assert not wants_agentic_figure_retrieval(
        message,
        asks_measured_quantity=_asks_measured_false,
        exclude=lambda t: "rate" in t.lower() or "productivity" in t.lower(),
    )


def _chunk(text):
    from types import SimpleNamespace
    return SimpleNamespace(text=text)


def _fake_retrieve(chunks_by_item):
    def _retrieve(query, project_id, k=5):
        return [_chunk(chunks_by_item.get(query, ""))], 0
    return _retrieve


def _fake_bm25(chunks_by_item):
    def _bm25(project_id, query, k=50):
        return [_chunk(chunks_by_item.get(query, ""))]
    return _bm25


async def _plan_caller(plan_text):
    return {"choice": {"message": {"content": plan_text}}}


@pytest.mark.parametrize("plan_text,initial", [
    ("bulk excavation, blinding", "bulk excavation 75 mm blinding to slab"),
    ("formwork, painting", "formwork and painting quantities"),
])
async def test_augment_returns_first_pass_when_covered(plan_text, initial):
    out = await augment_figure_retrieval(
        question="what are the quantities",
        project_id="p",
        first_pass_evidence=initial,
        plan_caller=lambda q: _plan_caller(plan_text),
        retrieve_fn=_fake_retrieve({}),
        bm25_fn=_fake_bm25({}),
    )
    assert out == initial


@pytest.mark.parametrize("plan_text,initial,item_fill", [
    ("bulk excavation, blinding", "bulk excavation only", {"blinding": "75 mm blinding to concrete"}),
    ("formwork, painting", "formwork only", {"painting": "painting to all walls"}),
])
async def test_augment_folds_in_recovered_evidence(plan_text, initial, item_fill):
    out = await augment_figure_retrieval(
        question="what are the quantities",
        project_id="p",
        first_pass_evidence=initial,
        plan_caller=lambda q: _plan_caller(plan_text),
        retrieve_fn=_fake_retrieve({}),
        bm25_fn=_fake_bm25(item_fill),
    )
    assert out != initial
    assert "blinding" in out or "painting" in out


@pytest.mark.parametrize("plan_text,initial,item_fill", [
    ("bulk excavation, blinding", "bulk excavation only", {"blinding": "75 mm blinding to concrete"}),
    ("formwork, painting", "formwork only", {"painting": "painting to all walls"}),
])
async def test_run_sync_executor_is_used(plan_text, initial, item_fill):
    calls = []

    async def _run_sync(fn, *args, **kwargs):
        calls.append(fn.__name__)
        return fn(*args, **kwargs)

    outcome = await run_project_figure_retrieval(
        question="what are the quantities",
        plan_caller=lambda q: _plan_caller(plan_text),
        retrieve_fn=_fake_retrieve({}),
        bm25_fn=_fake_bm25(item_fill),
        project_id="p",
        first_pass_evidence=initial,
        run_sync=_run_sync,
    )
    assert calls == ["run_loop"]
    assert outcome.missing == ()
