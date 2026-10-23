"""An encrypted upload is written block by block: memory does not grow with size.

With DATA_ENCRYPTION_KEY set, a streamed upload used to be buffered whole and
encrypted as one Fernet token (peak ~4x the file). It is now written as
length-prefixed per-block tokens after a plain header, and every reader
(read_document, open_plaintext, plaintext_size) understands both that and the
older whole-file token. Synthetic bytes only.
"""
from __future__ import annotations

import hashlib
import io
import os
import tracemalloc

import pytest
from cryptography.fernet import Fernet


@pytest.fixture
def fc(monkeypatch):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    import importlib
    from app.core import file_crypto
    importlib.reload(file_crypto)
    yield file_crypto
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    importlib.reload(file_crypto)


def _body(n):
    unit = b"synthetic reference text 0123456789 "
    return (unit * (n // len(unit) + 1))[:n]


def _peak_for(fc, tmp_path, n):
    body = _body(n)
    path = str(tmp_path / f"book_{n}.pdf")
    tracemalloc.start()
    tracemalloc.reset_peak()
    h = hashlib.sha256()
    size = fc.write_document_stream(path, io.BytesIO(body), hasher=h)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert size == n
    assert h.hexdigest() == hashlib.sha256(body).hexdigest()
    with open(path, "rb") as raw:
        assert body[:64] not in raw.read(4096)  # ciphertext on disk
    return peak


def test_encrypted_stream_memory_does_not_grow_with_file_size(fc, tmp_path):
    small = _peak_for(fc, tmp_path, 4 * 1024 * 1024)
    large = _peak_for(fc, tmp_path, 32 * 1024 * 1024)
    # One block's worth of Fernet work either way: 8x the data, ~same peak.
    assert large < small * 1.5 + 1024 * 1024, (small, large)
    assert large < 32 * 1024 * 1024 / 2


def test_every_reader_round_trips_the_streamed_file(fc, tmp_path):
    body = _body(3 * 1024 * 1024 + 17)
    path = str(tmp_path / "doc.pdf")
    fc.write_document_stream(path, io.BytesIO(body))
    assert fc.read_document(path) == body
    assert fc.plaintext_size(path) == len(body)
    with fc.open_plaintext(path) as p:
        assert p != path
        with open(p, "rb") as fh:
            assert fh.read() == body


def test_older_whole_file_tokens_still_read(fc, tmp_path):
    path = str(tmp_path / "old.txt")
    fc.write_document(path, b"older encrypted document")
    assert fc.read_document(path) == b"older encrypted document"
    with fc.open_plaintext(path) as p:
        with open(p, "rb") as fh:
            assert fh.read() == b"older encrypted document"


def test_oversize_encrypted_stream_leaves_nothing(fc, tmp_path):
    path = str(tmp_path / "big.pdf")
    with pytest.raises(fc.UploadTooLarge):
        fc.write_document_stream(path, io.BytesIO(_body(3 * 1024 * 1024)), max_bytes=1024 * 1024)
    assert not os.path.exists(path)
