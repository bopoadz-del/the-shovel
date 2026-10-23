"""Retrieval workers are sized from the task's vCPUs and share one model
server; turns stream progress before the first token. Synthetic data."""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import pytest

from app.core import turn_progress
from app.core.rag import model_server, retrieval_worker


@pytest.mark.parametrize("explicit,vcpus,per,expected", [
    ("3", 1.0, "2", 3),      # an explicit count wins
    ("", 1.0, "2", 2),       # 1 vCPU x 2
    ("", 4.0, "1.5", 6),
    ("", 0.5, "1", 1),       # never below one
    ("", 2.0, "nonsense", 4),
])
def test_worker_count_comes_from_config_and_vcpus(monkeypatch, explicit, vcpus, per, expected):
    monkeypatch.setenv("RAG_RETRIEVAL_WORKERS", explicit)
    monkeypatch.setenv("RAG_RETRIEVAL_WORKERS_PER_VCPU", per)
    monkeypatch.setattr(retrieval_worker, "available_vcpus", lambda: vcpus)
    assert retrieval_worker._workers() == expected


def test_available_vcpus_is_positive():
    assert retrieval_worker.available_vcpus() > 0


def test_workers_share_one_model_server(monkeypatch):
    """Two workers encode through the same server process; neither loads a model."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_RETRIEVAL_WORKERS", "2")
    monkeypatch.setenv("RAG_RETRIEVAL_WORKER_NICE", "0")
    monkeypatch.setattr(retrieval_worker, "_POOL", None)
    # A server an earlier test started has that test's environment; start ours.
    model_server.stop()
    try:
        assert retrieval_worker.warm_all() == 2
        from app.core.rag import embeddings

        emb = embeddings.get_embedder()
        assert type(emb).__name__ == "RemoteEmbedder"  # this process too
        here = emb.encode(["notice of suspension", "retention release"])
        out = retrieval_worker.call_sync(_encode_in_worker, ["notice of suspension", "retention release"])
        assert out["kind"] == "RemoteEmbedder"
        assert out["local_model"] is None
        assert np.allclose(np.asarray(out["vectors"]), here)
    finally:
        pool = retrieval_worker._POOL
        if pool is not None:
            pool.shutdown()
        monkeypatch.setattr(retrieval_worker, "_POOL", None)
        model_server.stop()
        from app.core.rag import embeddings, reranker

        embeddings.reset_embedder_cache()
        reranker._model = None
        reranker._model_name = None


def _encode_in_worker(texts):
    from app.core.rag import embeddings

    e = embeddings.get_embedder()
    return {"kind": type(e).__name__, "local_model": e._model, "vectors": e.encode(texts).tolist()}


def test_every_stream_opens_with_an_accepted_status():
    async def body():
        yield "data: {\"type\": \"token\", \"content\": \"hi\"}\n\n"

    async def run():
        return [x async for x in turn_progress.opened(body())]

    out = asyncio.run(run())
    first = json.loads(out[0][len("data: "):])
    assert first == {"type": "status", "stage": "accepted", "label": turn_progress.LABELS["accepted"]}
    assert "token" in out[1]


def test_every_stage_has_a_label():
    for stage in ("accepted", "searching", "calculating", "writing"):
        assert turn_progress.event(stage)["label"]


@pytest.mark.parametrize("limits,expected", [({"CPU": 1.0}, 1.0), ({"CPU": 1024}, 1.0),
                                              ({"CPU": 0.5}, 0.5), ({}, None)])
def test_the_task_cpu_limit_comes_from_ecs_metadata(monkeypatch, limits, expected):
    import http.server
    import threading

    body = json.dumps({"Limits": limits}).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", f"http://127.0.0.1:{srv.server_port}/v4/x")
        assert retrieval_worker._ecs_task_cpu_limit() == expected
        if expected:
            assert retrieval_worker.available_vcpus() == expected
    finally:
        srv.shutdown()


def test_no_ecs_metadata_means_no_task_limit(monkeypatch):
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4", raising=False)
    assert retrieval_worker._ecs_task_cpu_limit() is None
