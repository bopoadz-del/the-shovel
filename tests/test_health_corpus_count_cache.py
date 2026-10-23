"""Corpus chunk count on /health is cached. No database."""
from __future__ import annotations

import threading
import time

import pytest

import app.core.health_probes as health_probes
from app.core.health_probes import probe_corpus_chunks, reset_corpus_chunk_cache


def _ok(chunks: int, source: str = "estimate") -> dict:
    return {
        "chunks": chunks,
        "source": source,
        "table": "chunks_v2",
        "latency_ms": 4,
        "error": None,
    }


def setup_function():
    reset_corpus_chunk_cache()


def teardown_function():
    thread = health_probes._corpus_refresh_thread
    if thread is not None and thread.is_alive():
        thread.join(2)
    reset_corpus_chunk_cache()


def test_two_calls_inside_ttl_refresh_once(monkeypatch):
    calls = []

    def refresher(timeout_s: float = 1.5):
        calls.append(timeout_s)
        return _ok(11)

    monkeypatch.setattr(health_probes, "_refresh_corpus_count", refresher)
    first = probe_corpus_chunks()
    second = probe_corpus_chunks()
    assert calls == [1.5]
    assert first["source"] == "estimate"
    assert first["chunks"] == 11
    assert second["source"] == "cached"
    assert second["chunks"] == 11
    assert second["age_s"] >= 0
    assert second.get("stale") is not True


def test_stale_call_returns_immediately_and_refreshes_once(monkeypatch):
    calls = []
    started = threading.Event()
    release = threading.Event()

    def refresher(timeout_s: float = 1.5):
        calls.append(timeout_s)
        if len(calls) == 1:
            return _ok(4, "estimate")
        started.set()
        release.wait(2)
        return _ok(8, "exact")

    monkeypatch.setattr(health_probes, "_refresh_corpus_count", refresher)
    primed = probe_corpus_chunks()
    assert primed["chunks"] == 4
    with health_probes._corpus_cache_lock:
        health_probes._corpus_cache_at = time.monotonic() - (health_probes._corpus_ttl_s() + 5)

    began = time.monotonic()
    stale = probe_corpus_chunks()
    elapsed = time.monotonic() - began
    assert elapsed < 0.3
    assert stale["stale"] is True
    assert stale["chunks"] == 4
    assert stale["source"] == "estimate"
    assert stale["age_s"] >= health_probes._corpus_ttl_s()
    assert started.wait(2)

    again = probe_corpus_chunks()
    assert again["stale"] is True
    assert again["chunks"] == 4
    release.set()
    thread = health_probes._corpus_refresh_thread
    assert thread is not None
    thread.join(2)
    assert calls == [1.5, 1.5]


def test_failed_refresh_is_not_requeried_inside_negative_ttl(monkeypatch):
    calls = []

    def refresher(timeout_s: float = 1.5):
        calls.append(timeout_s)
        raise RuntimeError("db down")

    monkeypatch.setattr(health_probes, "_refresh_corpus_count", refresher)
    first = probe_corpus_chunks()
    second = probe_corpus_chunks()
    assert calls == [1.5]
    assert first["error"] and "db down" in first["error"]
    assert second["error"] and "db down" in second["error"]
    assert second["source"] == "cached"
    assert second["age_s"] < health_probes._corpus_negative_ttl_s()

    with health_probes._corpus_cache_lock:
        health_probes._corpus_cache_at = time.monotonic() - (
            health_probes._corpus_negative_ttl_s() + 1
        )
    third = probe_corpus_chunks()
    assert calls == [1.5, 1.5]
    assert third["error"] and "db down" in third["error"]


def test_parallel_cold_calls_refresh_once(monkeypatch):
    calls = []
    call_lock = threading.Lock()
    gate = threading.Event()

    def refresher(timeout_s: float = 1.5):
        with call_lock:
            calls.append(timeout_s)
        gate.wait(2)
        return _ok(5)

    monkeypatch.setattr(health_probes, "_refresh_corpus_count", refresher)
    workers = 6
    barrier = threading.Barrier(workers)
    results = []
    result_lock = threading.Lock()

    def worker():
        barrier.wait()
        payload = probe_corpus_chunks()
        with result_lock:
            results.append(payload)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    deadline = time.time() + 2
    while time.time() < deadline and len(results) < workers - 1:
        time.sleep(0.01)
    try:
        assert calls == [1.5]
        assert len(results) == workers - 1
        assert all(row.get("source") == "warming" and row.get("warming") is True for row in results)
    finally:
        gate.set()
        for thread in threads:
            thread.join(2)
    assert calls == [1.5]
    assert sum(1 for row in results if row.get("chunks") == 5) == 1
    assert sum(1 for row in results if row.get("source") == "warming") == workers - 1


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "abc"])
def test_unusable_ttl_env_uses_default(monkeypatch, raw):
    """0, negative, nan, inf, and non-numeric values do not change the TTLs."""
    monkeypatch.setenv("CORPUS_CHUNK_CACHE_TTL_S", raw)
    monkeypatch.setenv("CORPUS_CHUNK_CACHE_NEGATIVE_TTL_S", raw)
    assert health_probes._corpus_ttl_s() == 300
    assert health_probes._corpus_negative_ttl_s() == 30


def test_env_ttl_overrides_take_effect(monkeypatch):
    """Both TTLs come from the environment. Defaults stay 300s and 30s."""
    assert health_probes._corpus_ttl_s() == 300
    assert health_probes._corpus_negative_ttl_s() == 30

    monkeypatch.setenv("CORPUS_CHUNK_CACHE_TTL_S", "12")
    monkeypatch.setenv("CORPUS_CHUNK_CACHE_NEGATIVE_TTL_S", "3")
    assert health_probes._corpus_ttl_s() == 12
    assert health_probes._corpus_negative_ttl_s() == 3

    calls = []

    def refresher(timeout_s: float = 1.5):
        calls.append(timeout_s)
        if len(calls) == 1:
            raise RuntimeError("db down")
        return _ok(9)

    monkeypatch.setattr(health_probes, "_refresh_corpus_count", refresher)
    probe_corpus_chunks()
    cached_failure = probe_corpus_chunks()
    assert calls == [1.5]
    assert cached_failure["source"] == "cached"
    assert "db down" in cached_failure["error"]

    # 4s is past the overridden negative window and inside the 30s default.
    with health_probes._corpus_cache_lock:
        health_probes._corpus_cache_at = time.monotonic() - 4
    refreshed = probe_corpus_chunks()
    assert calls == [1.5, 1.5]
    assert refreshed["chunks"] == 9
    assert refreshed["source"] == "estimate"

    still_fresh = probe_corpus_chunks()
    assert calls == [1.5, 1.5]
    assert still_fresh["source"] == "cached"
    assert still_fresh["chunks"] == 9

    # 13s is past the overridden success TTL and inside the 300s default.
    started = threading.Event()
    release = threading.Event()

    def slow(timeout_s: float = 1.5):
        calls.append(timeout_s)
        started.set()
        release.wait(2)
        return _ok(10, "exact")

    monkeypatch.setattr(health_probes, "_refresh_corpus_count", slow)
    with health_probes._corpus_cache_lock:
        health_probes._corpus_cache_at = time.monotonic() - 13
    try:
        stale = probe_corpus_chunks()
        assert stale["stale"] is True
        assert stale["chunks"] == 9
        assert stale["source"] == "estimate"
        assert started.wait(2)
    finally:
        release.set()
        thread = health_probes._corpus_refresh_thread
        if thread is not None:
            thread.join(2)
    assert calls == [1.5, 1.5, 1.5]


def test_refresher_error_is_payload_and_does_not_raise(monkeypatch):
    def refresher(timeout_s: float = 1.5):
        raise RuntimeError("db down")

    monkeypatch.setattr(health_probes, "_refresh_corpus_count", refresher)
    payload = probe_corpus_chunks()
    assert payload["chunks"] is None
    assert payload["error"]
    assert "RuntimeError" in payload["error"]
    assert "db down" in payload["error"]
