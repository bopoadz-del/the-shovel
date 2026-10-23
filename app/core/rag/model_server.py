"""One process holds the retrieval models; every retrieval worker shares it.

With retrieval in N worker processes (app.core.rag.retrieval_worker), each
worker loading its own embedder would multiply the model's memory by N. The
model server is a separate process that loads the embedder (and the
cross-encoder, when reranking is on) once; workers get thin proxies that
send texts and receive vectors or scores. Encoding calls are served in
threads, and the model libraries release the GIL while they compute.

The server starts once, from the web process, before the worker pool. Its
address and a random key are handed to each worker at start.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from multiprocessing.managers import BaseManager
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)


class _Models:
    """Lives in the server process; loads each model on first use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embedder = None

    def _emb(self):
        with self._lock:
            if self._embedder is None:
                from app.core.rag.embeddings import get_embedder

                self._embedder = get_embedder()
            return self._embedder

    def info(self) -> dict:
        e = self._emb()
        return {"model_name": e.model_name, "dim": e.dim, "backend": e.backend, "fake": bool(e._fake)}

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._emb().encode(texts)

    def rerank_scores(self, pairs: List[Tuple[str, str]]) -> Optional[List[float]]:
        from app.core.rag import reranker

        model = reranker._get_model()
        if model is None:
            return None
        return [float(s) for s in model.predict(pairs, show_progress_bar=False)]


_MODELS: Optional[_Models] = None


def _models() -> _Models:
    global _MODELS
    if _MODELS is None:
        _MODELS = _Models()
    return _MODELS


class _Manager(BaseManager):
    pass


_Manager.register("models", callable=_models)

_SERVER: Optional[_Manager] = None
_SERVER_LOCK = threading.Lock()


def start() -> Tuple[Any, bytes]:
    """Start the model server once; return (address, authkey) for workers."""
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            mgr = _Manager(authkey=os.urandom(32), ctx=multiprocessing.get_context("spawn"))
            mgr.start()
            _SERVER = mgr
            _LOG.info("retrieval model server started (pid %s)", mgr._process.pid)
        return _SERVER.address, bytes(_SERVER._authkey)


def stop() -> None:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            _SERVER.shutdown()
            _SERVER = None


def connect(address: Any, authkey: bytes) -> Any:
    mgr = _Manager(address=address, authkey=authkey)
    mgr.connect()
    return mgr.models()


# ── in each worker: proxies with the interfaces retrieval already uses ───────

def remote_embedder(models: Any):
    """An ``Embedder`` whose encoding is done by the model server."""
    from app.core.rag.embeddings import Embedder

    info = models.info()

    class RemoteEmbedder(Embedder):
        def __init__(self) -> None:  # no local model
            self.model_name = info["model_name"]
            self._dim = info["dim"]
            self._backend = info["backend"]
            self._fake = info["fake"]
            self._model = None

        def encode(self, texts: List[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, self.dim), dtype=np.float32)
            return models.encode(list(texts))

    return RemoteEmbedder()


class RemoteCrossEncoder:
    """``CrossEncoder.predict`` answered by the model server."""

    def __init__(self, models: Any) -> None:
        self._models = models

    def predict(self, pairs: Sequence, show_progress_bar: bool = False) -> List[float]:
        scores = self._models.rerank_scores([tuple(p) for p in pairs])
        if scores is None:
            raise RuntimeError("the model server has no cross-encoder")
        return scores


def install_in_this_process(address: Any, authkey: bytes) -> None:
    """Route this process's embedder (and reranker) to the model server."""
    from app.core.rag import embeddings, reranker

    models = connect(address, authkey)
    with embeddings._CACHE_LOCK:
        embeddings._EMBEDDER_CACHE = remote_embedder(models)
    with reranker._lock:
        reranker._model = RemoteCrossEncoder(models)
        reranker._model_name = os.getenv("RAG_RERANKER_MODEL", "").strip() or reranker.DEFAULT_MODEL
        reranker._load_failed = False
