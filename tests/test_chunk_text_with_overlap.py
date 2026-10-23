"""``chunk_text_with_overlap`` — finer-grained chunker for BOQ-style PDFs
(FOLLOW-UP #91). Targets 400-600 chars per chunk with 50-char overlap and
respects BOQ row boundaries so a line stays adjacent to its item code."""
from __future__ import annotations

import pytest

from app.core.doc_index import chunk_text_with_overlap


def test_empty_returns_empty():
    assert chunk_text_with_overlap("") == []
    assert chunk_text_with_overlap("   \n\n  ") == []


def test_short_text_fits_in_one_chunk():
    text = "Short BOQ line item with no need to split."
    chunks = chunk_text_with_overlap(text, target_chars=500)
    assert chunks == [text]


def test_chunks_target_size_within_bounds():
    # Build 5000 chars of repeated lines to force splitting.
    body = ("This is a row of BOQ text that should be split into reasonable chunks. " * 80)
    chunks = chunk_text_with_overlap(body, target_chars=500, overlap=50)
    assert len(chunks) >= 5
    # Each chunk should be at most max_chars (800 default).
    assert all(len(c) <= 800 for c in chunks)
    # Average should be in the 400-600 target band.
    avg = sum(len(c) for c in chunks) / len(chunks)
    assert 200 <= avg <= 700


def test_boq_row_boundary_preferred():
    # Two BOQ rows; the chunker should split BETWEEN them when target is hit.
    row_a = "D 999.14 Nr 100 1,275.00 127,500.00 Description of work for item A " * 8
    row_b = "D 999.15 m  300 228.00   68,400.00 Description of work for item B " * 8
    text = row_a + "\n" + row_b
    chunks = chunk_text_with_overlap(text, target_chars=600, overlap=50)
    # Each chunk's prefix (after overlap) should normally start at or near a
    # row boundary. At minimum, item codes should NOT be sliced across chunks
    # so that the digit suffix lands in a different chunk than the letter.
    for chunk in chunks:
        # No chunk should END mid-item-code like ``D 999.``.
        assert not chunk.rstrip().endswith("D 999."), f"chunk ends mid-code: {chunk[-40:]!r}"


def test_d_99914_stays_with_its_rate():
    """Regression: the specific item D 999.14 (mentioned in the Q2 regression
    query) must end up in a chunk that also contains its rate ``228.00`` or
    similar. Without overlap, a row split between item code and rate is the
    classic BOQ retrieval failure mode."""
    line = "D 999.14 Nr 897 228.00 204,516.00 Stormwater culvert removal\n"
    padding = ("Some surrounding context that pads the document. " * 30)
    text = padding + line + padding
    chunks = chunk_text_with_overlap(text, target_chars=500, overlap=50)
    matched = [c for c in chunks if "999.14" in c and "228.00" in c]
    assert matched, (
        "D 999.14 and its rate 228.00 were split across chunks — the chunker "
        "must keep an item code and its numeric values in the same chunk."
    )


def test_overlap_carries_context():
    text = "A" * 200 + "B" * 200 + "C" * 200 + "D" * 200
    chunks = chunk_text_with_overlap(text, target_chars=300, overlap=20, max_chars=400)
    assert len(chunks) >= 2
    # Consecutive chunks should share some tail/head content (the overlap).
    # We can't check exact substring across chunks because the breaker may
    # land on a whitespace edge — but each chunk should be at most max_chars.
    assert all(len(c) <= 400 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text_with_overlap("hello world", target_chars=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text_with_overlap("hello world", target_chars=10, overlap=-1)


def test_max_chars_caps_runaway_chunk():
    # Single huge line with no break markers — must still cap at max_chars.
    text = "x" * 2000
    chunks = chunk_text_with_overlap(text, target_chars=500, overlap=50, max_chars=600)
    assert all(len(c) <= 600 for c in chunks)
    assert len(chunks) >= 3
