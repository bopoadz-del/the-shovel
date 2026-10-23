"""Embedding must run in bounded batches, whatever the document size.

Live-fire phase 3: indexing a 203-page contract (530K chars -> ~600 chunks)
spiked instance memory from ~550MB to 1.81GB against a 2GB limit -- OOM, 502,
instance replaced, and the file being indexed was lost with it (before the
persistent-disk fix). Mechanism: retriever.index_chunks hands the ENTIRE chunk
list to embedder.encode in one call.

The fence asserts the property, not the symptom (an OOM cannot be red-tested
politely): encode() must feed the backend in slices no larger than the bound,
and the batched output must be numerically identical to the unbatched one.
"""
from __future__ import annotations

import numpy as np

import app.core.rag.embeddings as emb


class _SpyBackend:
    """Stands in for the sentence-transformers model; records batch sizes."""

    def __init__(self, dim=8):
        self.dim = dim
        self.batch_sizes = []

    def encode(self, texts, **kw):
        self.batch_sizes.append(len(texts))
        # deterministic per-text vector so batched == unbatched is checkable
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % self.dim] = 1.0
        return out


def _spied_embedder(monkeypatch, n_texts):
    e = object.__new__(emb.Embedder)
    e._fake = False
    e._backend = "sentence_transformers"
    e._model = _SpyBackend()
    e._dim = 8
    e.model_name = "spy"
    return e


def test_a_large_corpus_is_encoded_in_bounded_slices(monkeypatch):
    e = _spied_embedder(monkeypatch, 1000)
    texts = [f"chunk {i}" for i in range(1000)]
    out = e.encode(texts)
    assert out.shape == (1000, 8)
    sizes = e._model.batch_sizes
    assert len(sizes) > 1, "everything went to the backend in ONE call"
    assert max(sizes) <= emb.ENCODE_SLICE, f"slice bound exceeded: {max(sizes)}"
    assert sum(sizes) == 1000, "texts lost or duplicated across slices"


def test_batched_output_equals_unbatched(monkeypatch):
    e1 = _spied_embedder(monkeypatch, 0)
    texts = [f"t{i}" for i in range(emb.ENCODE_SLICE + 7)]
    batched = e1.encode(texts)
    e2 = _spied_embedder(monkeypatch, 0)
    single = np.concatenate([e2.encode([t]) for t in texts])
    assert np.allclose(batched, single), "batching changed the vectors"


def test_small_inputs_are_a_single_call(monkeypatch):
    e = _spied_embedder(monkeypatch, 3)
    e.encode(["a", "b", "c"])
    assert e._model.batch_sizes == [3]
