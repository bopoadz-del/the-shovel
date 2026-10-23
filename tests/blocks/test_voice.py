"""Tests for Voice Block."""

import base64
import os
import tempfile

import pytest

from app.blocks import VoiceBlock
from app.blocks import voice as voice_module


@pytest.fixture
def voice_block():
    return VoiceBlock()


@pytest.mark.asyncio
async def test_voice_block_execute_structure(voice_block):
    """Test that Voice block returns standardized JSON structure."""
    result = await voice_block.execute(
        "Hello world",
        {"operation": "tts", "provider": "mock"}
    )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "voice"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_voice_block_metadata(voice_block):
    """Test Voice block metadata."""
    assert voice_block.name == "voice"
    assert voice_block.version == "2.2"
    # assert "text" in voice_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "audio" in voice_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert voice_block.config.requires_api_key == False  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_voice_block_tts(voice_block):
    """Test Voice block text-to-speech."""
    result = await voice_block.execute(
        "Hello world",
        {"operation": "tts", "provider": "mock"}
    )
    
    assert result["block"] == "voice"
    assert "result" in result
    assert result["result"]["operation"] == "tts"


@pytest.mark.asyncio
async def test_voice_block_stt(voice_block):
    """Test Voice block speech-to-text."""
    # Mock audio input
    result = await voice_block.execute(
        {"audio_base64": "bW9ja19hdWRpb19kYXRh"},  # base64 encoded mock data
        {"operation": "stt"}
    )

    assert result["block"] == "voice"
    assert "result" in result


# ── Production hardening — input shapes accepted by STT ─────────────────────
# Pre-2.1 the block read only the `file_path` key. The construction
# `daily_site_report` mixin sent `audio_path` instead, so transcripts were
# silently empty. These tests lock in every accepted key + the base64 path,
# and use a monkeypatched `_stt_sync` so they don't hit Google's STT.


@pytest.mark.asyncio
async def test_stt_treats_text_as_path_when_file_exists(voice_block, monkeypatch, tmp_path):
    """The platform's InputAdapter wraps a positional string into
    {"text": "<path>"} for voice (no typed file_path schema). On STT,
    voice must recover that case by treating `text` as a file path when
    it actually exists on disk — otherwise the construction
    daily_site_report call (which goes through .execute(voice_file, …))
    silently drops the audio. This is the regression guard for the
    user-requested 'text parameter' contract."""
    fake_wav = tmp_path / "site_note.wav"
    fake_wav.write_bytes(b"RIFF....WAVEfmt ")
    monkeypatch.setattr(voice_module, "_stt_sync", lambda fp: "morning report")

    envelope = await voice_block.execute(
        {"text": str(fake_wav)}, {"operation": "stt"}
    )
    inner = envelope["result"]
    assert inner["status"] == "success", inner
    assert inner["text"] == "morning report"
    assert inner["file"] == "site_note.wav"


@pytest.mark.asyncio
async def test_stt_text_that_is_not_a_path_still_errors(voice_block):
    """When `text` is set but isn't a file path (e.g. caller confused TTS
    and STT), STT must NOT try to treat the literal text as audio. It
    must error cleanly with the accepted_keys hint."""
    envelope = await voice_block.execute(
        {"text": "Just a sentence, not a path"}, {"operation": "stt"}
    )
    inner = envelope["result"]
    assert inner["status"] == "error"
    assert "audio source required" in inner["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["file_path", "audio_path", "path", "audio"])
async def test_stt_accepts_path_aliases(voice_block, monkeypatch, tmp_path, key):
    """Voice 2.1 must accept file_path / audio_path / path / audio as
    equivalent ways to point at the audio file. This is the regression
    guard for the schedule.py:396 bug."""
    fake_wav = tmp_path / "note.wav"
    fake_wav.write_bytes(b"RIFF....WAVEfmt ")
    monkeypatch.setattr(voice_module, "_stt_sync", lambda fp: "hello from the site")

    envelope = await voice_block.execute(
        {key: str(fake_wav)}, {"operation": "stt"}
    )
    inner = envelope["result"]
    assert inner["status"] == "success", inner
    assert inner["operation"] == "stt"
    assert inner["text"] == "hello from the site"
    assert inner["file"] == "note.wav"


@pytest.mark.asyncio
async def test_stt_with_audio_base64_decodes_to_temp_wav(voice_block, monkeypatch):
    """Inline base64 audio (the natural shape from the browser recorder)
    must be decoded to a temp WAV and transcribed."""
    captured: dict = {}

    def _fake_stt(fp: str) -> str:
        # Record the temp path so we can prove it was created + readable
        captured["path"] = fp
        captured["exists"] = os.path.exists(fp)
        captured["bytes"] = open(fp, "rb").read()
        return "voice note transcript"

    monkeypatch.setattr(voice_module, "_stt_sync", _fake_stt)

    payload = base64.b64encode(b"RIFF....WAVEfmt fake-audio-bytes").decode()
    envelope = await voice_block.execute(
        {"audio_base64": payload}, {"operation": "stt"}
    )
    inner = envelope["result"]
    assert inner["status"] == "success", inner
    assert inner["text"] == "voice note transcript"
    # _stt_sync saw a real readable temp file
    assert captured["exists"] is True
    assert captured["bytes"].startswith(b"RIFF")
    # And the temp file was cleaned up after the call
    assert not os.path.exists(captured["path"])


@pytest.mark.asyncio
async def test_stt_with_no_source_returns_accepted_keys(voice_block):
    """An STT call with no audio source must return a structured error
    that lists every accepted input key — the lack of that hint is what
    let the audio_path bug live for so long."""
    envelope = await voice_block.execute({"text": "irrelevant"}, {"operation": "stt"})
    inner = envelope["result"]
    assert inner["status"] == "error"
    assert "audio source required" in inner["error"]
    keys = set(inner["accepted_keys"])
    assert {"file_path", "audio_path", "path", "audio", "audio_base64"}.issubset(keys)


@pytest.mark.asyncio
async def test_stt_with_missing_file_returns_clear_error(voice_block):
    envelope = await voice_block.execute(
        {"audio_path": "/tmp/does_not_exist_xyz.wav"}, {"operation": "stt"}
    )
    inner = envelope["result"]
    assert inner["status"] == "error"
    assert "Audio file not found" in inner["error"]


@pytest.mark.asyncio
async def test_stt_with_bad_base64_returns_clear_error(voice_block):
    envelope = await voice_block.execute(
        {"audio_base64": "!!!not-base64!!!"}, {"operation": "stt"}
    )
    inner = envelope["result"]
    # base64 is permissive (validate=False) so non-b64 chars are dropped;
    # the decoded bytes still go to a temp file. The real error surfaces
    # later when _stt_sync tries to parse the file. The test asserts that
    # *whichever* path produces the error, status is error and the message
    # is non-empty — both branches are acceptable production behavior.
    assert inner["status"] == "error"
    assert isinstance(inner.get("error"), str) and inner["error"]


def test_voice_block_version_is_2_2():
    assert VoiceBlock.version == "2.2"


# ── 2.2 — pydub conversion path for browser-recorded audio ──────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".webm", ".mp3", ".m4a", ".ogg"])
async def test_stt_converts_browser_formats_via_pydub(voice_block, monkeypatch, tmp_path, ext):
    """Non-WAV uploads must be transcoded via pydub before STT.
    Mocks both the transcoder (so no ffmpeg needed in CI) and the
    underlying STT call. Verifies the converted_from field is surfaced
    so callers can observe the conversion happened."""
    src = tmp_path / f"recording{ext}"
    src.write_bytes(b"fake-browser-recording-bytes")

    converted_path = tmp_path / "converted.wav"
    converted_path.write_bytes(b"RIFF....WAVEfmt fake-converted")

    monkeypatch.setattr(voice_module, "_pydub_available", lambda: True)
    monkeypatch.setattr(
        voice_module, "_transcode_to_wav_sync",
        lambda src_path: str(converted_path),
    )
    monkeypatch.setattr(voice_module, "_stt_sync", lambda fp: f"converted {ext} ok")

    envelope = await voice_block.execute(str(src), {"operation": "stt"})
    inner = envelope["result"]
    assert inner["status"] == "success", inner
    assert inner["text"] == f"converted {ext} ok"
    assert inner["converted_from"] == ext
    assert inner["file"] == src.name


@pytest.mark.asyncio
async def test_stt_browser_format_without_pydub_returns_clear_error(voice_block, monkeypatch, tmp_path):
    """When pydub isn't installed, a non-WAV upload must error with a
    message that tells the operator exactly what to install."""
    src = tmp_path / "browser_clip.webm"
    src.write_bytes(b"fake-webm")
    monkeypatch.setattr(voice_module, "_pydub_available", lambda: False)

    envelope = await voice_block.execute(str(src), {"operation": "stt"})
    inner = envelope["result"]
    assert inner["status"] == "error"
    assert "pydub" in inner["error"].lower()
    assert ".webm" in inner["error"] or "'.webm'" in inner["error"]
    assert inner["source_format"] == ".webm"


@pytest.mark.asyncio
async def test_stt_native_wav_skips_conversion(voice_block, monkeypatch, tmp_path):
    """WAV input must NOT go through the conversion path — verifies we
    didn't accidentally route everything through pydub."""
    src = tmp_path / "native.wav"
    src.write_bytes(b"RIFF....WAVEfmt ")

    def _explode(*a, **k):
        raise AssertionError("_transcode_to_wav_sync should not have been called for WAV")

    monkeypatch.setattr(voice_module, "_transcode_to_wav_sync", _explode)
    monkeypatch.setattr(voice_module, "_stt_sync", lambda fp: "native ok")

    envelope = await voice_block.execute(str(src), {"operation": "stt"})
    inner = envelope["result"]
    assert inner["status"] == "success", inner
    assert inner["text"] == "native ok"
    # Native path doesn't add a converted_from field.
    assert "converted_from" not in inner


@pytest.mark.asyncio
async def test_stt_transcode_error_surfaces_ffmpeg_hint(voice_block, monkeypatch, tmp_path):
    """When pydub raises (typically because ffmpeg is missing), the
    block surfaces the install-ffmpeg hint to the caller."""
    src = tmp_path / "clip.mp3"
    src.write_bytes(b"fake")

    def _boom(src_path):
        raise RuntimeError(
            "ffmpeg not found on PATH — install ffmpeg "
            "(Render: `apt install ffmpeg`)."
        )

    monkeypatch.setattr(voice_module, "_pydub_available", lambda: True)
    monkeypatch.setattr(voice_module, "_transcode_to_wav_sync", _boom)

    envelope = await voice_block.execute(str(src), {"operation": "stt"})
    inner = envelope["result"]
    assert inner["status"] == "error"
    assert "ffmpeg" in inner["error"]
    assert inner["source_format"] == ".mp3"
