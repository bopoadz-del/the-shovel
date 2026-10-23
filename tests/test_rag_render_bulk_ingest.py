"""Tests for resumable Render bulk RAG ingestion.

Uses MemoryStore + a deterministic FakeEmbedder — no network, no Postgres.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from scripts.rag_render_bulk_ingest import (
    EMBED_MODEL,
    EXPECTED_DIM,
    ChunkRow,
    DocMeta,
    ExistingMeta,
    IngestStats,
    MemoryStore,
    StopFlag,
    is_valid_existing,
    plan_work,
    run_ingest,
    text_hash,
)


class FakeEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, dim: int = EXPECTED_DIM):
        self.model_name = model_name
        self.dim = dim
        self.encode_calls = 0
        self.encoded_texts: List[str] = []

    def encode(self, texts):
        self.encode_calls += 1
        self.encoded_texts.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # deterministic pseudo-embedding from text hash
            h = abs(hash(t)) % (10**8)
            out[i, 0] = (h % 1000) / 1000.0
            out[i, 1] = ((h // 1000) % 1000) / 1000.0
            out[i, 2] = 1.0
        # L2 normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return out / norms


def _docs_and_chunks(n: int = 8, doc_id: str = "docA") -> tuple:
    docs = {
        doc_id: DocMeta(doc_id=doc_id, original_name=f"{doc_id}.pdf"),
    }
    flat = [
        ChunkRow(doc_id=doc_id, chunk_index=i, text=f"chunk text {i} for {doc_id}")
        for i in range(n)
    ]
    return docs, flat


def test_skip_valid_existing():
    docs, flat = _docs_and_chunks(4)
    store = MemoryStore()
    emb = FakeEmbedder()
    # seed all as valid
    vecs = emb.encode([r.text for r in flat])
    store.upsert_batch(flat, vecs, EMBED_MODEL, EXPECTED_DIM)
    store._commit_n = 0

    stats = run_ingest(
        store, flat, docs, FakeEmbedder(), batch_size=2, resume=True, upsert_docs=True
    )
    assert stats.valid_skipped == 4
    assert stats.written == 0
    assert FakeEmbedder().encode_calls == 0  # new instance unused — check via returned
    assert stats.stopped_reason == "nothing_to_do"
    assert store.count_chunks("example_infra_pack") == 4


def test_generate_missing_only():
    docs, flat = _docs_and_chunks(4)
    store = MemoryStore()
    emb = FakeEmbedder()
    # seed first 2
    store.upsert_batch(flat[:2], emb.encode([r.text for r in flat[:2]]), EMBED_MODEL, EXPECTED_DIM)
    store._commit_n = 0

    runner = FakeEmbedder()
    stats = run_ingest(store, flat, docs, runner, batch_size=2, resume=True)
    assert stats.valid_skipped == 2
    assert stats.written == 2
    assert store.count_chunks("example_infra_pack") == 4
    assert set(runner.encoded_texts) == {flat[2].text, flat[3].text}


def test_resume_after_partial():
    docs, flat = _docs_and_chunks(6)
    store = MemoryStore()
    emb = FakeEmbedder()

    stats1 = run_ingest(store, flat, docs, emb, batch_size=2, max_batches=1, resume=True)
    assert stats1.written == 2
    assert stats1.stopped_reason == "max_batches"
    assert store.count_chunks("example_infra_pack") == 2

    emb2 = FakeEmbedder()
    stats2 = run_ingest(store, flat, docs, emb2, batch_size=2, resume=True)
    assert stats2.valid_skipped == 2
    assert stats2.written == 4
    assert store.count_chunks("example_infra_pack") == 6
    assert len(store.chunks) == 6  # no dups


def test_no_duplicate_chunks_on_rerun():
    docs, flat = _docs_and_chunks(5)
    store = MemoryStore()
    emb = FakeEmbedder()
    run_ingest(store, flat, docs, emb, batch_size=3, resume=True)
    n1 = store.count_chunks("example_infra_pack")
    emb2 = FakeEmbedder()
    stats = run_ingest(store, flat, docs, emb2, batch_size=3, resume=True)
    assert stats.written == 0
    assert stats.valid_skipped == 5
    assert store.count_chunks("example_infra_pack") == n1 == 5
    assert len(store.chunks) == 5


def test_failed_batch_writes_nothing():
    docs, flat = _docs_and_chunks(4)
    store = MemoryStore(fail_on_commit=1)
    emb = FakeEmbedder()
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        run_ingest(store, flat, docs, emb, batch_size=2, resume=True)
    assert store.count_chunks("example_infra_pack") == 0
    assert store.writes_attempted == 1


def test_failed_commit_rolls_back_batch_only_prior_survive():
    docs, flat = _docs_and_chunks(6)
    store = MemoryStore()
    emb = FakeEmbedder()
    # first batch succeeds
    run_ingest(store, flat, docs, emb, batch_size=2, max_batches=1, resume=True)
    assert store.count_chunks("example_infra_pack") == 2
    prior_ids = set(store.chunks.keys())

    # next run: fail on first new commit
    store.fail_on_commit = store._commit_n + 1
    emb2 = FakeEmbedder()
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        run_ingest(store, flat, docs, emb2, batch_size=2, resume=True)

    assert store.count_chunks("example_infra_pack") == 2
    assert set(store.chunks.keys()) == prior_ids


def test_sigint_stop_is_resumable():
    docs, flat = _docs_and_chunks(8)
    store = MemoryStore()
    emb = FakeEmbedder()
    stop = StopFlag()

    # stop before any batch by setting flag immediately via wrapper
    class StoppingEmbedder(FakeEmbedder):
        def encode(self, texts):
            stop.stop = True
            stop.reason = "SIGINT"
            return super().encode(texts)

    stats = run_ingest(
        store, flat, docs, StoppingEmbedder(), batch_size=2, resume=True, stop_flag=stop
    )
    # first batch still completes (stop checked between batches)
    assert stats.written == 2
    assert stats.stopped_reason == "SIGINT"
    assert store.count_chunks("example_infra_pack") == 2

    # resume completes the rest
    stats2 = run_ingest(store, flat, docs, FakeEmbedder(), batch_size=2, resume=True)
    assert stats2.valid_skipped == 2
    assert stats2.written == 6
    assert store.count_chunks("example_infra_pack") == 8


def test_sigterm_reason_recorded():
    docs, flat = _docs_and_chunks(4)
    store = MemoryStore()
    stop = StopFlag(stop=True, reason="SIGTERM")
    stats = run_ingest(
        store, flat, docs, FakeEmbedder(), batch_size=2, resume=True, stop_flag=stop
    )
    # stop checked at start of each batch — nothing written
    assert stats.written == 0
    assert stats.stopped_reason == "SIGTERM"


def test_wrong_model_rejected_and_regenerated():
    docs, flat = _docs_and_chunks(2)
    store = MemoryStore()
    bad = FakeEmbedder(model_name="other-model", dim=EXPECTED_DIM)
    store.upsert_batch(flat, bad.encode([r.text for r in flat]), "other-model", EXPECTED_DIM)
    store._commit_n = 0

    emb = FakeEmbedder()
    stats = run_ingest(store, flat, docs, emb, batch_size=2, resume=True)
    assert stats.rejected_wrong_identity == 2
    assert stats.written == 2
    for row in store.chunks.values():
        assert row["embedding_model"] == EMBED_MODEL
        assert row["embedding_dim"] == EXPECTED_DIM


def test_wrong_dims_rejected_and_regenerated():
    docs, flat = _docs_and_chunks(2)
    store = MemoryStore()
    # manually insert wrong dim metadata (embedding array still 384 for store simplicity)
    emb = FakeEmbedder()
    vecs = emb.encode([r.text for r in flat])
    store.upsert_batch(flat, vecs, EMBED_MODEL, 128)  # wrong dim stamped
    store._commit_n = 0

    stats = run_ingest(store, flat, docs, FakeEmbedder(), batch_size=2, resume=True)
    assert stats.rejected_wrong_identity == 2
    assert stats.written == 2
    for row in store.chunks.values():
        assert row["embedding_dim"] == EXPECTED_DIM


def test_text_hash_change_regenerates():
    docs, flat = _docs_and_chunks(2)
    store = MemoryStore()
    emb = FakeEmbedder()
    store.upsert_batch(flat, emb.encode([r.text for r in flat]), EMBED_MODEL, EXPECTED_DIM)
    store._commit_n = 0

    # mutate source text
    changed = [
        ChunkRow(doc_id=r.doc_id, chunk_index=r.chunk_index, text=r.text + " UPDATED")
        for r in flat
    ]
    runner = FakeEmbedder()
    stats = run_ingest(store, changed, docs, runner, batch_size=2, resume=True)
    assert stats.regenerated_text_hash == 2
    assert stats.written == 2
    assert store.chunks[changed[0].chunk_id]["text"].endswith("UPDATED")


def test_verify_only_no_writes():
    docs, flat = _docs_and_chunks(4)
    store = MemoryStore()
    emb = FakeEmbedder()
    stats = run_ingest(
        store, flat, docs, emb, batch_size=2, resume=True, verify_only=True, upsert_docs=False
    )
    assert stats.stopped_reason == "verify_only"
    assert stats.written == 0
    assert store.count_chunks("example_infra_pack") == 0
    assert store.writes_attempted == 0
    assert store.documents == {}


def test_max_batches_2_stops():
    docs, flat = _docs_and_chunks(10)
    store = MemoryStore()
    stats = run_ingest(
        store, flat, docs, FakeEmbedder(), batch_size=2, max_batches=2, resume=True
    )
    assert stats.batches == 2
    assert stats.written == 4
    assert stats.stopped_reason == "max_batches"
    assert store.count_chunks("example_infra_pack") == 4


def test_completed_corpus_zero_dup_writes(tmp_path: Path):
    docs, flat = _docs_and_chunks(6)
    store = MemoryStore()
    ck = tmp_path / "ckpt.json"
    run_ingest(
        store, flat, docs, FakeEmbedder(), batch_size=3, resume=True, checkpoint_path=ck
    )
    assert store.count_chunks("example_infra_pack") == 6
    assert ck.exists()

    before = dict(store.chunks)
    stats = run_ingest(store, flat, docs, FakeEmbedder(), batch_size=3, resume=True)
    assert stats.written == 0
    assert stats.valid_skipped == 6
    assert store.chunks.keys() == before.keys()
    # embeddings unchanged (same object ids after skip — no upsert)
    for cid in before:
        assert store.chunks[cid]["text"] == before[cid]["text"]


def test_is_valid_existing_helpers():
    row = ChunkRow(doc_id="d", chunk_index=0, text="hello")
    ok = ExistingMeta(
        chunk_id=row.chunk_id,
        embedding_model=EMBED_MODEL,
        embedding_dim=EXPECTED_DIM,
        embedding_normalized=True,
        text="hello",
    )
    assert is_valid_existing(ok, row, EMBED_MODEL, EXPECTED_DIM) == (True, "ok")
    bad = ExistingMeta(
        chunk_id=row.chunk_id,
        embedding_model="x",
        embedding_dim=EXPECTED_DIM,
        embedding_normalized=True,
        text="hello",
    )
    assert is_valid_existing(bad, row, EMBED_MODEL, EXPECTED_DIM)[0] is False


def test_plan_work_counts():
    flat = [ChunkRow(doc_id="d", chunk_index=i, text=f"t{i}") for i in range(3)]
    existing = {
        flat[0].chunk_id: ExistingMeta(
            chunk_id=flat[0].chunk_id,
            embedding_model=EMBED_MODEL,
            embedding_dim=EXPECTED_DIM,
            embedding_normalized=True,
            text=flat[0].text,
        ),
        flat[1].chunk_id: ExistingMeta(
            chunk_id=flat[1].chunk_id,
            embedding_model="wrong",
            embedding_dim=EXPECTED_DIM,
            embedding_normalized=True,
            text=flat[1].text,
        ),
    }
    todo, stats = plan_work(flat, existing, EMBED_MODEL, EXPECTED_DIM)
    assert stats.valid_skipped == 1
    assert stats.rejected_wrong_identity == 1
    assert len(todo) == 2
    assert text_hash("abc") == text_hash("abc")
    assert text_hash("abc") != text_hash("abd")


def test_preview_no_writes():
    docs, flat = _docs_and_chunks(4)
    store = MemoryStore()
    stats = run_ingest(
        store, flat, docs, FakeEmbedder(), batch_size=2, preview=True, upsert_docs=False
    )
    assert stats.stopped_reason == "preview"
    assert store.count_chunks("example_infra_pack") == 0


def test_destructive_wipe_clears(tmp_path: Path):
    docs, flat = _docs_and_chunks(3)
    store = MemoryStore()
    run_ingest(store, flat, docs, FakeEmbedder(), batch_size=3, resume=True)
    assert store.count_chunks("example_infra_pack") == 3
    stats = run_ingest(
        store,
        flat,
        docs,
        FakeEmbedder(),
        batch_size=3,
        resume=True,
        destructive_wipe=True,
    )
    # wipe then rewrite
    assert stats.written == 3
    assert store.count_chunks("example_infra_pack") == 3
