"""prefetch_embedder must retry Hub 429s instead of failing the image build.

Compose-health on PR #451 died at Dockerfile:102: huggingface.co returned
429 for minishlab/potion-base-8M HEAD requests. The Hub client's own 5×
1–8s retry exhausted and the build failed; health never ran. The outer
loop is what this pins.
"""
from __future__ import annotations

import scripts.prefetch_embedder as prefetch


def test_429_is_retryable():
    assert prefetch._retryable_prefetch_error(
        OSError("HTTP Error 429 thrown while requesting HEAD https://huggingface.co/")
    )
    assert prefetch._retryable_prefetch_error(
        OSError(
            "We couldn't connect to 'https://huggingface.co' to load the files"
        )
    )


def test_missing_model_is_not_retryable():
    assert not prefetch._retryable_prefetch_error(
        OSError("minishlab/does-not-exist is not a valid model identifier")
    )


def test_load_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def _flaky(_name):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("HTTP Error 429 thrown while requesting HEAD config.json")
        return object(), "sentence_transformers"

    slept = []
    monkeypatch.setattr(prefetch.time, "sleep", slept.append)
    model, backend = prefetch._load_with_retry("minishlab/potion-base-8M", load=_flaky)
    assert backend == "sentence_transformers"
    assert calls["n"] == 3
    assert slept == [20, 40]


def test_load_does_not_retry_a_real_failure():
    def _boom(_name):
        raise RuntimeError("model has no encode()")

    try:
        prefetch._load_with_retry("x", load=_boom)
    except RuntimeError as exc:
        assert "encode" in str(exc)
    else:
        raise AssertionError("non-retryable error must propagate")
