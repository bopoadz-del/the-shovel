"""Decrypted-temp plaintext must have a bounded lifetime, always.

The voice/doc decrypt paths clean up after themselves, but a crash between
decrypt and cleanup — or a failed unlink — used to orphan a file holding
CLIENT PLAINTEXT in /tmp indefinitely, and the failure branch was
unverifiable from outside. This makes the property structural instead:

* a failed unlink ZERO-OVERWRITES the contents in place (shred_file), so
  even a surviving name holds no plaintext;
* a sweeper reaps stale ``fork_dec_*`` orphans at boot, hourly, and on
  demand via /v1/admin/debug/sweep-plaintext — so the worst case is one
  sweep interval, not forever.
"""
from __future__ import annotations

import os
import time

import pytest

from app.core import file_crypto


def test_sweeper_reaps_stale_and_keeps_fresh(tmp_path):
    stale = tmp_path / "fork_dec_stale.wav"
    stale.write_bytes(b"OLD PLAINTEXT AUDIO")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    fresh = tmp_path / "fork_dec_fresh.wav"
    fresh.write_bytes(b"IN-FLIGHT AUDIO")
    unrelated = tmp_path / "other_tmpfile.bin"
    unrelated.write_bytes(b"NOT OURS")
    old2 = time.time() - 7200
    os.utime(unrelated, (old2, old2))

    res = file_crypto.sweep_stale_plaintext(3600, tmp_dir=str(tmp_path))

    assert not stale.exists(), "stale plaintext must be reaped"
    assert fresh.exists(), "in-flight file must be left alone"
    assert unrelated.exists(), "only the fork_dec_ pattern is ours to touch"
    assert res["reaped"] == 1 and res["failed"] == []


def test_shred_zeroes_contents_when_unlink_is_stuck(tmp_path, monkeypatch):
    p = tmp_path / "fork_dec_stuck.wav"
    secret = b"the concrete pour is on monday" * 10
    p.write_bytes(secret)

    real_remove = os.remove
    monkeypatch.setattr(os, "remove",
                        lambda path: (_ for _ in ()).throw(OSError("stuck")))
    try:
        assert file_crypto.shred_file(str(p)) is True
    finally:
        monkeypatch.setattr(os, "remove", real_remove)

    data = p.read_bytes()
    assert len(data) == len(secret)
    assert data == b"\0" * len(secret), "plaintext must not survive a stuck unlink"


def test_sweeper_shreds_what_it_cannot_remove(tmp_path, monkeypatch):
    stale = tmp_path / "fork_dec_locked.wav"
    secret = b"SECRET SITE AUDIO"
    stale.write_bytes(secret)
    old = time.time() - 7200
    os.utime(stale, (old, old))

    real_remove = os.remove

    def failing_remove(path):
        if "fork_dec_locked" in str(path):
            raise OSError("locked")
        return real_remove(path)

    # remove fails inside the sweep AND inside shred's retry; the zeroing
    # itself uses open(r+b) and still succeeds.
    monkeypatch.setattr(os, "remove", failing_remove)
    res = file_crypto.sweep_stale_plaintext(3600, tmp_dir=str(tmp_path))
    assert res["shredded"] == 1 and res["failed"] == []
    assert stale.read_bytes() == b"\0" * len(secret)


def test_voice_decrypted_temp_uses_the_sweepable_prefix():
    import inspect
    import app.blocks.voice as voice_mod
    src = inspect.getsource(voice_mod)
    # the block's decrypted copy must carry the fork_dec_ prefix or the
    # sweeper cannot see its orphans
    assert 'prefix="fork_dec_"' in src


def test_open_plaintext_shreds_on_failed_cleanup(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    stored = tmp_path / "doc.bin"
    secret = b"encrypted-at-rest document body"
    file_crypto.write_document(str(stored), secret)

    real_remove = os.remove
    tmp_seen = {}

    def failing_remove(path):
        if "fork_dec_" in str(path) and "once" not in tmp_seen:
            tmp_seen["once"] = path
            raise OSError("busy")
        return real_remove(path)

    monkeypatch.setattr(os, "remove", failing_remove)
    with file_crypto.open_plaintext(str(stored)) as plain:
        assert open(plain, "rb").read() == secret
        tmp_seen["plain"] = plain
    # cleanup's unlink failed once -> shred kicked in: file either gone
    # (shred's retry succeeded) or zeroed
    leftover = tmp_seen["plain"]
    if os.path.exists(leftover):
        assert open(leftover, "rb").read() == b"\0" * len(secret)
        real_remove(leftover)
