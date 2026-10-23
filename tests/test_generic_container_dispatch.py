"""Scoped Dispatch v2. The reasoner may pick a deliverable-producing container
action (risk register, cash-flow, commissioning checklist, ...): run_workflow
dispatches ONLY those on the explicit SCOPED_DISPATCH_ALLOWLIST via the
container, and LLM-synthesizes the result. Q&A / lookup / document actions
(chat, process_document, ...) are NEVER intercepted even though the container
implements them - they fall through to the RAG-grounded agent path. Raw render
is the emergency fallback when synthesis fails, never a silent stub.
"""
from __future__ import annotations

import asyncio

import app.core.predefined_reasoning as pr
from app.schemas.project_session import ProjectSession


class _FakeContainer:
    def get_actions(self):
        # A deliverable action (allowlisted) + Q&A/lookup actions (never dispatch)
        return {
            "risk_register_auto_populate": object(),
            "process_document": object(),
            "chat": object(),
            "benchmark_lookup": object(),
        }

    async def execute(self, envelope):
        assert envelope.get("action") == "risk_register_auto_populate"
        return {
            "status": "success",
            "action": "risk_register",
            "total_risks": 3,
            "risks": [
                {"id": "R1", "hazard": "excavation collapse", "severity": "high"},
                {"id": "R2", "hazard": "utility strike", "severity": "medium"},
                {"id": "R3", "hazard": "dewatering", "severity": "low"},
            ],
        }


def _patch_container(monkeypatch):
    monkeypatch.setattr("app.dependencies.get_block_instance",
                        lambda name: _FakeContainer())


def _fake_llm(monkeypatch, text=None, boom=False):
    async def _complete(messages, **kw):
        if boom:
            raise RuntimeError("no LLM provider configured")
        return text
    import app.core.llm_client as lc
    monkeypatch.setattr(lc, "complete", _complete)


def test_allowlisted_action_dispatches_and_synthesizes(monkeypatch):
    _patch_container(monkeypatch)
    _fake_llm(monkeypatch, text="## Risk Register\nThree risks were identified: "
              "excavation collapse (high), utility strike (medium), dewatering (low).")
    ctx = {"message": "populate a risk register", "project_id": "p",
           "document_ids": [], "params": {}}
    out = asyncio.run(pr.run_workflow("risk_register_auto_populate", ctx,
                                      ProjectSession.new("t", "system")))
    assert out["handled"] is True and out["status"] == "success"
    # The delivered answer is the LLM synthesis, not the raw bullet dump.
    assert "Risk Register" in out["answer"]
    assert out["plan_steps"] == ["risk_register_auto_populate"]


def test_synthesis_failure_falls_back_to_deterministic_render(monkeypatch):
    _patch_container(monkeypatch)
    _fake_llm(monkeypatch, boom=True)          # LLM unavailable
    ctx = {"message": "populate a risk register", "project_id": "p",
           "document_ids": [], "params": {}}
    out = asyncio.run(pr.run_workflow("risk_register_auto_populate", ctx,
                                      ProjectSession.new("t", "system")))
    assert out["handled"] is True
    # never a stub: the deterministic render still carries the real payload
    ans = out["answer"]
    assert "excavation collapse" in ans and "utility strike" in ans


def test_qa_and_lookup_actions_are_never_intercepted(monkeypatch):
    """Anti-hijack guard: even though the container implements process_document /
    chat / benchmark_lookup, they are NOT on the allowlist, so dispatch returns
    handled=False and the caller keeps the RAG-grounded agent path."""
    _patch_container(monkeypatch)
    _fake_llm(monkeypatch, text="should not be called")
    for qa_action in ("process_document", "chat", "benchmark_lookup"):
        out = asyncio.run(pr.run_workflow(qa_action, {"message": "what does X say?"},
                                          ProjectSession.new("t", "system")))
        assert out == {"handled": False}, f"{qa_action} must not be dispatched"


def test_unknown_action_falls_through(monkeypatch):
    _patch_container(monkeypatch)
    out = asyncio.run(pr.run_workflow("chit_chat_question", {"message": "hi"},
                                      ProjectSession.new("t", "system")))
    assert out == {"handled": False}


def test_container_action_names_is_allowlist_intersection(monkeypatch):
    _patch_container(monkeypatch)
    names = pr.container_action_names()
    assert names == {"risk_register_auto_populate"}   # only the allowlisted impl
    assert "process_document" not in names and "chat" not in names


def test_container_action_names_empty_without_kit(monkeypatch):
    def _boom(name):
        raise RuntimeError("construction kit not loaded")
    monkeypatch.setattr("app.dependencies.get_block_instance", _boom)
    assert pr.container_action_names() == set()
