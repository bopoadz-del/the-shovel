"""The F-SPEED regression fixes: workers at normal priority, the corpus
readiness check off the worker queue, the rate limiter off the event loop,
keyword patterns compiled at import."""
from __future__ import annotations

import asyncio
import threading
from tests._dev_key import DEV_KEY


def test_retrieval_workers_run_at_normal_priority(monkeypatch):
    from app.core.rag import retrieval_worker

    calls = []
    monkeypatch.delenv("RAG_RETRIEVAL_WORKER_NICE", raising=False)
    monkeypatch.setattr(retrieval_worker.os, "nice", lambda n: calls.append(n), raising=False)
    retrieval_worker._worker_start(None)
    assert calls == []  # default 0: no change of priority


def test_the_readiness_check_does_not_queue_behind_retrieval(monkeypatch):
    import app.agents.runtime as rt
    from app.core.rag import retrieval_worker

    def boom(*a, **k):
        raise AssertionError("readiness check went to the retrieval worker queue")

    monkeypatch.setattr(retrieval_worker, "call", boom)
    monkeypatch.setattr(retrieval_worker, "enabled", lambda: True)
    monkeypatch.setattr(rt, "project_is_rag_ready", lambda pid: pid == "p-ready")
    assert asyncio.run(rt._project_is_rag_ready_off_process("p-ready")) is True


def test_the_rate_limit_check_runs_off_the_event_loop(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    from app.core import rate_limit

    seen = {}
    real = rate_limit.check_and_record

    def spy(identity):
        seen["thread"] = threading.current_thread()
        return real(identity)

    monkeypatch.setattr(main._rate_limit, "check_and_record", spy)
    with TestClient(main.app) as client:
        client.get("/v1/projects", headers={"Authorization": "Bearer " + DEV_KEY})
    assert seen["thread"] is not threading.main_thread()


def test_keyword_patterns_are_compiled_at_import():
    from app.blocks import smart_orchestrator as so

    every = {kw for _a, kws in so.ACTION_PATTERNS for kw in kws}
    assert every and every <= set(so._KW_REGEX_CACHE)


def test_no_usage_record_runs_on_the_event_loop():
    """Every usage_tracker.record call in the model-call code is awaited
    through off_loop (a database write per model call)."""
    import ast
    from pathlib import Path

    src = Path("app/agents/core/llm.py").read_text(encoding="utf-8")
    direct = [n.lineno for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "record" and getattr(n.func.value, "id", "") == "usage_tracker"]
    assert direct == [], f"usage_tracker.record called directly at lines {direct}"
    assert src.count("usage_tracker.record,") >= 2
