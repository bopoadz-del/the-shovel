"""The child's memory ceiling must be a BUDGET, not an absolute number.

This is the regression test for a bug that reached production and silently
emptied every document >= 1 MB.

``RLIMIT_AS`` caps VIRTUAL address space, and ``fork()`` gives the child a copy
of the parent's whole mapping. With torch and the embedding model resident the
parent's VmSize is already multiple GB, so setting an ABSOLUTE ceiling of
1536 MB was already breached at the moment the child started. The first
allocation raised MemoryError, ``_extract_pdf`` caught it via ``except
Exception`` and returned ``("", {})`` -- a *successful* empty extraction -- and
the document indexed as ZERO_CHUNK with nothing logged anywhere.

Observed live on 2026-08-21 after deploying the isolation change::

    doc-reindex d0c0000c (7.9 MB, 738 pages) -> ZERO_CHUNK in 0.42s
    fresh_extraction = {"total_chars": 0, "meta": {}}
    pdf_first_pages_chars = [357, 791, 540, 847, ... 6610, 6725]

The pages plainly had text. Two independent defects had to line up: a limit
that could never be satisfied, and a handler that turned the resulting error
into an empty file. Both are covered here.
"""
from __future__ import annotations

import os

import pytest

from app.core import extract_isolated


def test_limit_is_relative_to_the_parent_not_absolute(monkeypatch):
    """The whole bug in one assertion: the ceiling must exceed what the child
    already inherits, or it is unsatisfiable before a single byte is read.
    The parent's size is read ONCE and the limit is computed from that same
    reading: read twice, a busy process grows in between (CI saw the two
    reads differ by 17 MB) and the exact assertion compared two moments."""
    parent = extract_isolated._parent_virtual_bytes()
    monkeypatch.setattr(extract_isolated, "_parent_virtual_bytes", lambda: parent)
    if parent is None:
        pytest.skip("/proc/self/status unavailable (non-Linux)")

    budget = 1536 * 1024 * 1024
    limit = extract_isolated.child_address_space_limit(budget)

    assert limit is not None
    assert limit > parent, (
        "limit must sit ABOVE the parent's inherited virtual size; the "
        "absolute 1536MB ceiling was below it, so the child died instantly"
    )
    assert limit == parent + budget


def test_budget_maths_holds_without_proc(monkeypatch):
    """Platform-independent version of the assertion above.

    The /proc test SKIPS on Windows, which is where this bug was written and
    where it passed review -- the fork path never runs here either. A test that
    only executes on CI is exactly how the outage slipped through, so pin the
    arithmetic everywhere by injecting a realistic parent size.

    3.5 GB is representative: torch plus the embedding model resident.
    """
    parent = 3500 * 1024 * 1024
    budget = 1536 * 1024 * 1024
    monkeypatch.setattr(extract_isolated, "_parent_virtual_bytes", lambda: parent)

    limit = extract_isolated.child_address_space_limit(budget)

    assert limit == parent + budget
    assert limit > parent, "the shipped bug: ceiling below what the child inherits"
    # The old behaviour, for contrast: a bare 1536MB ceiling sits far UNDER the
    # inherited 3.5GB, so the child could never allocate anything at all.
    assert budget < parent


def test_no_limit_rather_than_an_impossible_one():
    """When the parent's size is unknown, set no limit at all.

    Guessing an absolute number is what caused the outage. No limit still
    isolates: the child is the largest process, so the OOM killer takes it and
    the web worker survives.
    """
    real = extract_isolated._parent_virtual_bytes
    extract_isolated._parent_virtual_bytes = lambda: None
    try:
        assert extract_isolated.child_address_space_limit(123) is None
    finally:
        extract_isolated._parent_virtual_bytes = real


@pytest.mark.skipif(os.name != "posix", reason="reads /proc")
def test_parent_virtual_size_is_plausible():
    """Guard the premise: if VmSize ever reads as tiny, the budget maths is
    meaningless and this whole approach needs revisiting."""
    parent = extract_isolated._parent_virtual_bytes()
    if parent is None:
        pytest.skip("/proc/self/status unavailable")
    assert parent > 16 * 1024 * 1024, f"implausible VmSize: {parent}"


def test_memory_error_is_not_reported_as_an_empty_document(monkeypatch, tmp_path):
    """The second half of the outage: a memory failure must PROPAGATE, not be
    absorbed into ("", {}) and indexed as an empty file."""
    from app.core import doc_index

    pdf = tmp_path / "big.pdf"
    pdf.write_bytes(b"%PDF-1.4 " + b"0" * 1024)

    class _Boom:
        def __init__(self, *a, **k):
            raise MemoryError("cannot allocate (injected)")

    import fitz
    monkeypatch.setattr(fitz, "open", _Boom)

    with pytest.raises(MemoryError):
        doc_index._extract_pdf(str(pdf))


def test_generic_failures_are_still_absorbed(monkeypatch, tmp_path):
    """Only MemoryError changes. A corrupt PDF must still degrade to empty
    rather than failing the whole ingest run."""
    from app.core import doc_index

    pdf = tmp_path / "corrupt.pdf"
    # Header present so the quarantine gate keeps the file and the mocked
    # opener still absorbs a generic parser failure.
    pdf.write_bytes(b"%PDF-1.4\nnot a pdf at all")

    import fitz

    def _bad(*a, **k):
        raise ValueError("cannot open broken document")

    monkeypatch.setattr(fitz, "open", _bad)

    text, meta = doc_index._extract_pdf(str(pdf))
    assert text == ""
    assert meta.get("extract_failed") == "ValueError"
    assert "cannot open" in (meta.get("extract_failed_detail") or "")
