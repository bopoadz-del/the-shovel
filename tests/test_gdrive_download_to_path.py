"""Unit tests for ``gdrive_service.download_file_to_path``.

The streamed download is the ingest memory bound: it must never hold a whole
file in RAM, must hash incrementally, and must clean up its partial file on
every failure path. Each branch below is driven by at least two inputs (valid
vs invalid id, token present vs absent, 200 vs non-200, clean stream vs raised
stream, empty vs non-empty body).
"""
from __future__ import annotations

import hashlib

import httpx
import pytest

from app.core import gdrive_service


class _Resp:
    """Fake streamed response. ``chunks`` is an iterable of byte strings."""

    def __init__(self, chunks, status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self, chunk_size=None):
        yield from self._chunks


class _Client:
    """Fake httpx.Client; ``stream`` records its call and returns ``resp``."""

    def __init__(self, resp: _Resp, **kwargs):
        self.resp = resp
        self.kwargs = kwargs
        self.stream_args = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url, **kwargs):
        self.stream_args = (method, url, kwargs)
        return self.resp


@pytest.fixture()
def fake_client(monkeypatch):
    """Patch the httpx.Client the module imports, plus a minted token."""
    monkeypatch.setattr(gdrive_service, "_mint_access_token", lambda: "tok")
    holder = {}

    def _install(resp: _Resp):
        holder["client"] = _Client(resp, timeout=120)
        monkeypatch.setattr(httpx, "Client", lambda **kw: holder["client"])
        return holder["client"]

    holder["install"] = _install
    return holder


VALID_ID = "1234567890abcdef"


@pytest.mark.parametrize("chunks", [[b"hello ", b"world"], [b"abc", b"defghij"]])
def test_streams_to_path_and_hashes_incrementally(tmp_path, fake_client, chunks):
    """Success: chunks land on disk in order with the right size+digest."""
    body = b"".join(chunks)
    fake_client["install"](_Resp(chunks))

    dest = tmp_path / "out.bin"
    size, sha, err = gdrive_service.download_file_to_path(VALID_ID, dest)

    assert err is None
    assert size == len(body)
    assert sha == hashlib.sha256(body).hexdigest()
    assert dest.read_bytes() == body
    args = fake_client["client"].stream_args
    assert args[0] == "GET"
    assert args[1].endswith(f"/files/{VALID_ID}")
    assert args[2]["params"] == {"alt": "media", "supportsAllDrives": "true"}


@pytest.mark.parametrize("bad_id", ["not a drive id", "short"])
def test_invalid_file_id_is_rejected_without_a_token(tmp_path, monkeypatch, bad_id):
    """An id that is not a Drive id fails before any token is minted."""
    minted = []
    monkeypatch.setattr(gdrive_service, "_mint_access_token", lambda: minted.append(1) or "tok")

    size, sha, err = gdrive_service.download_file_to_path(bad_id, tmp_path / "x")

    assert (size, sha) == (None, None)
    assert err == "invalid drive file id"
    assert minted == []


@pytest.mark.parametrize("file_id", ["1234567890abcdef", "ABCDEFGHIJ_12345"])
def test_missing_token_returns_service_unavailable(tmp_path, monkeypatch, file_id):
    monkeypatch.setattr(gdrive_service, "_mint_access_token", lambda: None)

    size, sha, err = gdrive_service.download_file_to_path(file_id, tmp_path / "x")

    assert (size, sha) == (None, None)
    assert err == "service account unavailable"


@pytest.mark.parametrize("status", [404, 500])
def test_non_200_status_returns_error_and_leaves_no_file(tmp_path, fake_client, status):
    fake_client["install"](_Resp([b"partial"], status_code=status))

    dest = tmp_path / "out.bin"
    size, sha, err = gdrive_service.download_file_to_path(VALID_ID, dest)

    assert (size, sha) == (None, None)
    assert err == f"Drive download returned {status}"
    assert not dest.exists()


@pytest.mark.parametrize("msg", ["connection dropped", "read timed out"])
def test_stream_error_removes_the_partial_file(tmp_path, fake_client, msg):
    def _chunks():
        yield b"partial"
        raise RuntimeError(msg)

    fake_client["install"](_Resp(_chunks()))

    dest = tmp_path / "out.bin"
    size, sha, err = gdrive_service.download_file_to_path(VALID_ID, dest)

    assert (size, sha) == (None, None)
    assert err == f"RuntimeError: {msg}"
    assert not dest.exists()


@pytest.mark.parametrize("chunks", [[], [b""]])
def test_empty_body_yields_zero_size_and_the_empty_digest(tmp_path, fake_client, chunks):
    fake_client["install"](_Resp(chunks))

    dest = tmp_path / "out.bin"
    size, sha, err = gdrive_service.download_file_to_path(VALID_ID, dest)

    assert err is None
    assert size == 0
    assert sha == hashlib.sha256(b"").hexdigest()
    assert dest.read_bytes() == b""
