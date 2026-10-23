"""Extraction must be survivable: a runaway document dies alone.

Measured on the live box 2026-08-20: one 4.4 MB PDF producing 376 chunks took
the instance from 790 MB to 3.6 GB, peaked at 3,892 MB of 4,096, and was
SIGKILLed. With ``UVICORN_WORKERS`` unset there is ONE uvicorn worker, so the
kill took every concurrent request with it -- surfacing as a 502 on an
unrelated chat turn, with no traceback anywhere because SIGKILL raises nothing.

Bounding the input cannot fix that: the 4.4 MB file blew up while a 200 MB
scan would not. These tests pin the bound that can hold -- extraction in a
child under RLIMIT_AS -- and, critically, that a failed child returns a REASON
rather than a bare empty string, so a 0-chunk document is distinguishable from
a genuinely empty file.
"""
from __future__ import annotations

import os

import pytest

from app.core import extract_isolated


# Module-level helpers: a forked child re-uses the parent's memory image, but
# these must still be importable/picklable in any start method.
def _ok(a, b):
    return (f"{a}-{b}", {"meta": True})


def _hog(_a, _b):
    """Allocate far past any sane child ceiling."""
    blocks = []
    for _ in range(4096):
        blocks.append(bytearray(8 * 1024 * 1024))  # 8 MB a time, up to 32 GB
    return ("never", {})


def _boom(_a, _b):
    raise ValueError("extractor exploded")


def _sleeper(_a, _b):
    import time
    time.sleep(60)
    return ("never", {})


def test_returns_the_real_result_when_extraction_succeeds():
    out, diag = extract_isolated.run_isolated(
        _ok, ("x", "y"), fallback=("", {}), label="test"
    )
    assert out == ("x-y", {"meta": True})
    assert diag == {}


def test_falls_back_in_process_when_isolation_is_unavailable(monkeypatch):
    """Windows dev boxes have no fork. Behaviour there must be unchanged."""
    monkeypatch.setattr(extract_isolated, "_ENABLED", False)
    assert extract_isolated.isolation_available() is False
    out, diag = extract_isolated.run_isolated(
        _ok, ("a", "b"), fallback=("", {}), label="test"
    )
    assert out == ("a-b", {"meta": True})
    assert diag == {}, "in-process fallback must not report a failure"


def test_isolation_is_not_nested(monkeypatch):
    """Archive members recurse through _extract_with_meta. Forking per member
    would multiply processes without bounding anything further."""
    monkeypatch.setenv(extract_isolated._IN_CHILD_ENV, "1")
    assert extract_isolated.isolation_available() is False


@pytest.mark.skipif(
    not extract_isolated.isolation_available(),
    reason="requires POSIX fork (Render/Linux); skipped on the Windows dev box",
)
def test_a_memory_hog_is_contained_and_reported(monkeypatch):
    """THE case this module exists for: the allocation dies inside the child,
    the caller gets a reason, and this test process is still alive to assert
    it -- which is the whole point."""
    monkeypatch.setattr(extract_isolated, "_MEM_MB", 256)
    out, diag = extract_isolated.run_isolated(
        _hog, ("a", "b"), fallback=("", {}), label="extraction"
    )
    assert out == ("", {}), "must fall back to empty, not propagate the hog"
    assert diag.get("extract_failed") in ("memory", "crash"), diag
    assert "extract_failed_detail" in diag
    # A 0-chunk document must say WHY it is empty.
    assert diag["extract_failed_detail"], diag


@pytest.mark.skipif(
    not extract_isolated.isolation_available(),
    reason="requires POSIX fork (Render/Linux); skipped on the Windows dev box",
)
def test_a_crashing_extractor_is_reported_not_raised():
    out, diag = extract_isolated.run_isolated(
        _boom, ("a", "b"), fallback=("", {}), label="extraction"
    )
    assert out == ("", {})
    assert diag.get("extract_failed") == "error", diag
    assert "extractor exploded" in diag["extract_failed_detail"]


@pytest.mark.skipif(
    not extract_isolated.isolation_available(),
    reason="requires POSIX fork (Render/Linux); skipped on the Windows dev box",
)
def test_a_wedged_extractor_times_out_instead_of_pinning_the_worker(monkeypatch):
    monkeypatch.setattr(extract_isolated, "_TIMEOUT_S", 1.0)
    out, diag = extract_isolated.run_isolated(
        _sleeper, ("a", "b"), fallback=("", {}), label="extraction"
    )
    assert out == ("", {})
    assert diag.get("extract_failed") == "timeout", diag


def test_env_knobs_have_headroom_defaults():
    """The ceiling must leave room for the measured baseline + a chat turn on
    the 4 GB instance: ~670 MB idle + ~190 MB chat + child <= 4096 MB."""
    assert 256 <= extract_isolated._MEM_MB <= 2560, extract_isolated._MEM_MB
    assert extract_isolated._TIMEOUT_S >= 60


def test_doc_index_reports_the_failure_reason_in_meta(monkeypatch, tmp_path):
    """The wiring, not just the helper: a failed extraction must surface its
    reason through _extract_with_meta so the document carries a diagnosis.

    The file must be big enough to clear the isolation threshold -- a small or
    missing one takes the in-process path by design and never reaches
    run_isolated at all.
    """
    from app.core import doc_index

    big = tmp_path / "report.pdf"
    big.write_bytes(b"%PDF-1.4 " + b"0" * (2 * 1024 * 1024))

    def fake_run_isolated(fn, args, *, fallback, label):
        return fallback, {"extract_failed": "memory",
                          "extract_failed_detail": f"{label} hit the cap"}

    monkeypatch.setattr(extract_isolated, "worth_isolating", lambda _p: True)
    monkeypatch.setattr(extract_isolated, "run_isolated", fake_run_isolated)
    text, meta = doc_index._extract_with_meta(str(big), "report.pdf")
    assert text == ""
    assert meta["extract_failed"] == "memory"
    assert "report.pdf" in meta["extract_failed_detail"]


def test_small_documents_are_not_isolated(tmp_path):
    """Side effects during extraction are only observable in-process, so a
    document too small to threaten the box must NOT be forked.

    This is what CI caught: test_extract_pdf_ocr_page_cap_bounds_memory counts
    _ocr_pdf_page calls in a closure, and saw 0 because the counting happened
    inside the child.
    """
    small = tmp_path / "tiny.pdf"
    small.write_bytes(b"%PDF-1.4 dummy")
    assert extract_isolated.worth_isolating(str(small)) is False


def test_missing_files_are_not_isolated(tmp_path):
    """A path that cannot be stat'd fails in-process, where the real error is
    visible, rather than being reported as an opaque child crash."""
    assert extract_isolated.worth_isolating(str(tmp_path / "gone.pdf")) is False
