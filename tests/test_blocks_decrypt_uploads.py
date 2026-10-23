"""Blocks that read uploaded files must read PLAINTEXT, not ciphertext.

Phase-4 full-surface campaign finds (F29, F30). Platform uploads are
encrypted at rest (file_crypto, per-file nonce). A live STT probe on a real
uploaded MP3 failed with "pydub could not decode ... install ffmpeg and
retry" -- ffmpeg was installed; the block had fed CIPHERTEXT to the
decoder and the error blamed the wrong cause. An audit of every
file-consuming block found a second case: file_hasher hashed the stored
bytes, so the same content re-uploaded (new nonce) hashes differently --
which defeats a content hash's entire purpose (dedup, integrity,
comparison).

Both now route reads through open_plaintext; unencrypted local files pass
through untouched.
"""
from __future__ import annotations

import hashlib

import pytest

from app.core import file_crypto


@pytest.fixture(autouse=True)
def _test_encryption_key(monkeypatch):
    """A throwaway Fernet key so the encrypted-at-rest contract is exercised
    everywhere, independent of the host environment."""
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())

WAV_HEADER = (b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
              b"@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")


def _encrypted_file(tmp_path, name, data):
    p = str(tmp_path / name)
    file_crypto.write_document(p, data)
    # sanity: the bytes on disk must actually be ciphertext
    assert open(p, "rb").read() != data
    return p


@pytest.mark.asyncio
async def test_file_hasher_hashes_the_content_not_the_ciphertext(tmp_path):
    """RED before the fix: digest of the stored (encrypted) bytes."""
    from app.blocks.file_hasher import FileHasherBlock
    data = b"the content identity is the plaintext" * 100
    p = _encrypted_file(tmp_path, "doc.bin", data)
    res = await FileHasherBlock().process({"file_path": p}, {"algorithms": ["sha256"]})
    assert res["status"] == "success"
    assert res["hashes"]["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.asyncio
async def test_file_hasher_is_stable_across_reencryption(tmp_path):
    """Same content, two encrypted saves (fresh nonce each) -> same hash."""
    from app.blocks.file_hasher import FileHasherBlock
    data = b"same document saved twice"
    p1 = _encrypted_file(tmp_path, "a.bin", data)
    p2 = _encrypted_file(tmp_path, "b.bin", data)
    assert open(p1, "rb").read() != open(p2, "rb").read()  # nonces differ
    b = FileHasherBlock()
    h1 = (await b.process({"file_path": p1}, {}))["hashes"]["sha256"]
    h2 = (await b.process({"file_path": p2}, {}))["hashes"]["sha256"]
    assert h1 == h2


@pytest.mark.asyncio
async def test_voice_stt_decodes_the_decrypted_audio(tmp_path, monkeypatch):
    """RED before the fix: pydub/speech_recognition received ciphertext.
    The recognizer is faked; the assertion is that the WAV bytes reaching
    the STT layer equal the PLAINTEXT audio."""
    import app.blocks.voice as voice_mod
    seen = {}

    def fake_stt_sync(path):
        seen["bytes"] = open(path, "rb").read()
        return "the concrete pour is on monday"
    # the block's recognition entry point (sync worker)
    target = None
    for cand in ("_stt_sync", "_recognize_sync", "_transcribe_sync"):
        if hasattr(voice_mod, cand):
            target = cand
            break
    assert target, "voice STT sync worker not found -- block restructured?"
    monkeypatch.setattr(voice_mod, target, fake_stt_sync)

    p = _encrypted_file(tmp_path, "note.wav", WAV_HEADER)
    res = await voice_mod.VoiceBlock().process(
        {"file_path": p}, {"action": "stt"})
    assert res.get("status") == "success", res
    assert seen["bytes"] == WAV_HEADER, "STT layer received non-plaintext bytes"
