"""Embedding model wrapper — loads a sentence-transformers or model2vec
model and reports its actual dimension.

The previous implementation hardcoded dimensions per backend and could load
a 256-dim model2vec checkpoint through the sentence-transformers path,
reporting 384 dims while emitting 256-dim vectors. That corrupted the
vector store (query vectors and stored vectors no longer matched). This
version:

1. Loads the model at construction time.
2. Detects ``dim`` from the model's actual output shape, never from a
   hardcoded constant.
3. L2-normalizes every emitted vector in code, regardless of backend, so
   cosine similarity is always a pure dot product.
4. Exposes a startup probe so callers can assert embedder.dim matches the
   store's declared dimension and fail loud instead of silently corrupting
   retrieval.
"""
from __future__ import annotations

import hashlib
import logging
import os
from threading import Lock
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Defaults. The actual dimension is read from the loaded model in Embedder.__init__.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL2VEC = "minishlab/potion-base-8M"

# Legacy constants kept only for reference / old tests that import them.
EMBEDDING_DIM = 384
EMBEDDING_DIM_MODEL2VEC = 256


# ── Module-level cache ───────────────────────────────────────────────────────
# One embedder per process. Loading the model is slow (~1-2 s) and the
# weights are large enough that we don't want multiple copies in RAM.

_EMBEDDER_CACHE: Optional["Embedder"] = None
_CACHE_LOCK = Lock()


def get_embedder(model_name: Optional[str] = None) -> "Embedder":
    """Return the process-cached embedder, creating it on first call.

    ``model_name=None`` honors the ``RAG_EMBEDDING_MODEL`` env var, then
    falls back to ``DEFAULT_MODEL2VEC``. Pass ``"fake"`` for tests.
    """
    global _EMBEDDER_CACHE
    name = model_name or os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_MODEL2VEC
    with _CACHE_LOCK:
        if _EMBEDDER_CACHE is None or _EMBEDDER_CACHE.model_name != name:
            _EMBEDDER_CACHE = Embedder(model_name=name)
    return _EMBEDDER_CACHE


def get_embedding_identity(model_name: Optional[str] = None) -> dict:
    """Return the identity dict of the configured embedder without caching."""
    return get_embedder(model_name=model_name).identity


# Upper bound on texts per backend call. 64 short passages keeps the
# tokenizer + forward-pass peak far below instance memory while staying large
# enough that throughput is unaffected. Overridable for benchmarks.
ENCODE_SLICE = int(os.getenv("EMBED_ENCODE_SLICE", "64"))


def embedder_is_loaded() -> bool:
    """True if the process-cached embedder is already warm-loaded.

    A NON-loading probe for health checks — reading the cache never triggers
    the (slow, ~1-2s cold) model load, so /health stays cheap. ``False`` means
    the embedder will lazy-load on first real use, not that it is broken.
    """
    return _EMBEDDER_CACHE is not None


def loaded_embedder_identity() -> Optional[dict]:
    """Identity of the already-loaded embedder, or None if not yet loaded.
    Non-loading (health-check safe)."""
    emb = _EMBEDDER_CACHE
    return emb.identity if emb is not None else None


def reset_embedder_cache() -> None:
    """Drop the cached embedder. Used by tests to swap fake/real cleanly."""
    global _EMBEDDER_CACHE, _LOAD_FAILURE
    with _CACHE_LOCK:
        _EMBEDDER_CACHE = None
        _LOAD_FAILURE = None


# Remembers WHY the configured embedder could not be constructed, so the
# condition can be reported instead of being rediscovered on every call.
_LOAD_FAILURE: Optional[str] = None


def embedder_health() -> dict:
    """Whether the configured embedder can actually be CONSTRUCTED.

    ``Embedder.available()`` answers a much weaker question — whether the
    backend LIBRARIES import — and callers have been reading it as "retrieval
    works". Those are different claims, and the gap between them was a silent
    production outage: the libraries ship in the image, but the model WEIGHTS
    were fetched from the model host at runtime, so ``available()`` returned
    True while every embed call failed. Documents were then stored, listed, and
    indexed with zero chunks, with the failure logged as a warning and dropped.

    LOADING, not just importing, is what this checks. It is deliberately NOT
    wired into ``available()``: that is called on hot paths and by readiness
    probes, and a cold construction costs seconds and hundreds of MB — the
    trap ``embedder_is_loaded`` already documents. Use this from health/admin
    surfaces and from boot warm-up, where paying for the truth is the point.

    Returns ``{"ok", "model", "backend", "dim", "loaded", "error"}``. Never
    raises: a health probe that dies tells you nothing.
    """
    global _LOAD_FAILURE
    name = os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_MODEL2VEC
    report = {
        "ok": False, "model": name, "backend": None,
        "dim": None, "loaded": embedder_is_loaded(), "error": None,
    }
    if _LOAD_FAILURE is not None:
        report["error"] = _LOAD_FAILURE
        return report
    try:
        emb = get_embedder()
    except Exception as exc:  # noqa: BLE001 — probes report, they don't raise
        _LOAD_FAILURE = f"{type(exc).__name__}: {exc}"
        report["error"] = _LOAD_FAILURE
        logger.error(
            "EMBEDDER_UNAVAILABLE model=%s — %s; semantic retrieval and "
            "document indexing cannot work in this process",
            name, _LOAD_FAILURE,
        )
        return report
    report.update(
        ok=True, backend=emb.backend, dim=emb.dim,
        loaded=True, model=emb.model_name,
    )
    return report


class Embedder:
    """Wraps a real semantic embedding model (or fake mode for tests).

    Construction loads the model and probes its output dimension so the
    vector store can be sized exactly. ``encode`` always returns L2-normalized
    float32 vectors of shape ``(len(texts), dim)``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._fake = model_name == "fake"
        self._model: Optional[object] = None
        self._backend: Optional[str] = None
        self._dim: int = 0

        if self._fake:
            self._backend = "fake"
            self._dim = EMBEDDING_DIM_MODEL2VEC
            return

        if _has_sentence_transformers():
            # Try sentence-transformers first for full-featured models.
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model: %s", model_name)
            self._model = SentenceTransformer(model_name)
            self._backend = "sentence_transformers"
        elif _has_model2vec():
            from model2vec import StaticModel
            logger.info("Loading model2vec model: %s", model_name)
            self._model = StaticModel.from_pretrained(model_name)
            self._backend = "model2vec"
        else:
            raise RuntimeError(
                "No semantic embedding backend available. Install "
                "sentence-transformers (requirements-rag.txt) OR model2vec "
                "(baseline requirements.txt) -- or pass model_name='fake'."
            )

        # Probe actual dimension from the model, fail loud if it can't be read.
        probe = self._encode_raw(["dimension probe"])
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise RuntimeError(
                f"Embedder {model_name} returned unexpected probe shape {probe.shape}"
            )
        self._dim = int(probe.shape[1])
        if self._dim <= 0:
            raise RuntimeError(f"Embedder {model_name} reported non-positive dim {self._dim}")
        logger.info("Embedder %s dim=%d backend=%s", model_name, self._dim, self._backend)

    @staticmethod
    def available() -> bool:
        """True when a backend LIBRARY is importable.

        NOT a statement that embedding works. The libraries are installed in
        the image unconditionally, so this returns True even when the model
        weights cannot be loaded — which is precisely how a broken embedder
        presented as a healthy one while documents indexed to zero chunks.
        Kept cheap on purpose (hot paths and readiness probes call it); use
        :func:`embedder_health` when the question is "does embedding actually
        work".
        """
        return _has_sentence_transformers() or _has_model2vec()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def query_instruction(self) -> Optional[str]:
        """Instruction prefix recommended for retrieval queries.

        Some sentence-transformers models (notably BGE) are trained with an
        asymmetric instruction: queries are prefixed with a retrieval task
        description, while passage/documents are encoded bare. This property
        returns the correct prefix for the loaded model, or None when the
        model does not specify one.

        Production retrieval and the embedder benchmark MUST use the same
        setting so benchmark numbers predict live behaviour.
        """
        if self._fake:
            return None
        # BGE models are trained with a query instruction for retrieval.
        # The exact string is the one documented on the BGE model cards.
        if "bge-" in self.model_name.lower():
            return "Represent this sentence for searching relevant passages: "
        return None

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Encode query texts, applying the model's retrieval instruction."""
        instr = self.query_instruction
        if instr:
            texts = [instr + t for t in texts]
        return self.encode(texts)

    @property
    def identity(self) -> dict:
        """Embedding-identity metadata for namespace compatibility checks.

        Stores the model name, dimension, and normalization flag alongside
        every vector written to a namespace. A store opened with a different
        identity fails loud, making mixed-model contamination structurally
        impossible.
        """
        return {
            "model": self.model_name,
            "dim": self._dim,
            "normalized": True,
        }

    def matches_identity(self, stored: dict) -> bool:
        """True when ``stored`` identity matches this embedder exactly."""
        if not isinstance(stored, dict):
            return False
        return (
            stored.get("model") == self.model_name
            and stored.get("dim") == self._dim
            and stored.get("normalized") is True
        )

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        """Raw model output, not normalized. Internal use only."""
        if self._backend == "sentence_transformers":
            # Do not rely on the model's normalize_embeddings flag; we normalize
            # ourselves so every backend behaves identically.
            return self._model.encode(
                texts,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
        if self._backend == "model2vec":
            return np.asarray(self._model.encode(texts), dtype=np.float32)
        raise RuntimeError(f"Unknown backend {self._backend}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors of shape (len(texts), dim).

        Encodes in bounded slices of ENCODE_SLICE texts. Callers hand this
        whole DOCUMENTS -- indexing a 203-page contract (~600 chunks in one
        call) spiked a 2GB instance to 1.81GB and OOM-killed it mid-index,
        taking the request (and, pre-disk, the uploaded file) with it. Peak
        memory is now bounded by the slice size, not the document size.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._fake:
            return np.array([_fake_embedding(t) for t in texts], dtype=np.float32)
        if len(texts) <= ENCODE_SLICE:
            return _l2_normalize(self._encode_raw(texts))
        parts = [
            self._encode_raw(texts[i:i + ENCODE_SLICE])
            for i in range(0, len(texts), ENCODE_SLICE)
        ]
        return _l2_normalize(np.concatenate(parts, axis=0))


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid divide-by-zero; zero vectors remain zero.
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        logger.warning("sentence-transformers import unavailable", exc_info=True)
        return False


def _has_model2vec() -> bool:
    try:
        import model2vec  # noqa: F401
        return True
    except ImportError:
        return False


def _fake_embedding(text: str) -> np.ndarray:
    """Deterministic 256-dim L2-normalized vector derived from text hash."""
    dim = EMBEDDING_DIM_MODEL2VEC
    h = hashlib.sha256(text.encode("utf-8")).digest()
    buf = bytearray()
    seed = h
    while len(buf) < dim * 4:
        seed = hashlib.sha256(seed).digest()
        buf.extend(seed)
    arr = np.frombuffer(bytes(buf[: dim * 4]), dtype=np.uint32).astype(np.float32)
    arr = (arr / (2**32 - 1)) * 2.0 - 1.0
    return _l2_normalize(arr.reshape(1, -1))[0]
