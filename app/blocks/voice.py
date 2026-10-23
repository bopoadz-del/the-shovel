"""Voice Block - gTTS (TTS) + SpeechRecognition/Google (STT).

STT input format support:
  - Native (no conversion): WAV, AIFF, FLAC — read directly by
    ``speech_recognition.AudioFile``.
  - Converted via pydub + ffmpeg: WebM (Chrome/Firefox MediaRecorder),
    MP3, m4a (Safari), Ogg/Opus. These are the actual formats the
    browser push-to-talk path produces.

The conversion path is gated on pydub import success AND ffmpeg being
on PATH at runtime. When either is missing, the block falls back to a
clear error pointing at the supported native formats so callers can
report a concrete install gap rather than a cryptic decoder failure.
"""

import asyncio
import base64
import logging
import os
import tempfile
from typing import Any

from app.core.universal_base import UniversalBlock

_LOG = logging.getLogger(__name__)

# Native formats speech_recognition.AudioFile reads directly.
_NATIVE_STT_FORMATS = frozenset({".wav", ".aiff", ".aif", ".flac"})

# Browser/mobile recording formats that need transcoding to WAV.
# WebM is the default for Chrome/Firefox MediaRecorder; m4a is Safari's;
# MP3 covers many shared-file uploads; Ogg/Opus is Firefox's older default.
_TRANSCODE_STT_FORMATS = frozenset({".webm", ".mp3", ".m4a", ".mp4",
                                    ".ogg", ".oga", ".opus"})


def _pydub_available() -> bool:
    """True iff pydub is importable. Does NOT verify ffmpeg-on-PATH;
    that's surfaced by the conversion call itself at use time."""
    try:
        import pydub  # noqa: F401
        return True
    except ImportError:
        return False


def _transcode_to_wav_sync(src_path: str) -> str:
    """Convert any pydub-readable audio file to a 16 kHz mono WAV in a
    temp file. Returns the temp WAV path. Caller owns cleanup.

    Raises:
        RuntimeError: when pydub can't decode the file (typically: ffmpeg
            isn't on PATH, or the source format is exotic). The message
            includes a hint about installing ffmpeg so the operator can
            act on it directly.
    """
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    try:
        audio = AudioSegment.from_file(src_path)
    except CouldntDecodeError as e:
        raise RuntimeError(
            f"pydub could not decode {os.path.basename(src_path)!r}. "
            f"Most likely cause: ffmpeg is not installed on PATH. "
            f"Install ffmpeg and retry. Underlying error: {e}"
        ) from e
    except FileNotFoundError as e:
        # pydub raises this when the ffmpeg subprocess can't be launched.
        raise RuntimeError(
            f"ffmpeg not found on PATH — pydub needs it to decode "
            f"non-WAV audio. Install ffmpeg (Render: `apt install ffmpeg` "
            f"in build steps; local: `choco install ffmpeg` / "
            f"`brew install ffmpeg`). Underlying error: {e}"
        ) from e
    # 16 kHz mono 16-bit PCM — Google STT's preferred shape and matches
    # the base64 → temp WAV path so we stay consistent.
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    audio.export(out_path, format="wav")
    return out_path


_SUPPORTED_LANGS = {
    "en": "English", "es": "Spanish", "ar": "Arabic", "fr": "French",
    "de": "German", "zh": "Chinese", "ja": "Japanese", "hi": "Hindi",
    "pt": "Portuguese", "ru": "Russian", "tr": "Turkish", "ko": "Korean",
    "it": "Italian", "nl": "Dutch", "pl": "Polish",
}


def _tts_sync(text: str, lang: str) -> tuple[bytes, str]:
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        tmp = f.name
    try:
        with open(tmp, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            _LOG.debug(
                "swallowed %s in _tts_sync() — continuing",
                "OSError", exc_info=True,
            )
    return data, "mp3"


def _stt_sync(file_path: str) -> str:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.record(source)
    # Uses Google's free STT API — no key required
    return r.recognize_google(audio)


class VoiceBlock(UniversalBlock):
    """TTS via gTTS (free, no key) · STT via Google SpeechRecognition (free, no key)"""

    auto_validate = False
    name = "voice"
    version = "2.2"
    description = "Text-to-speech (gTTS) and speech-to-text (Google STT) — no API key needed; WebM/MP3/m4a auto-converted to WAV via pydub+ffmpeg"
    layer = 3
    tags = ["domain", "audio", "tts", "stt"]
    requires = []

    # Accepted keys for the audio-file location on STT. The first hit wins.
    # `audio_path` is the natural name from the construction daily-site-report
    # caller; `file_path`/`path` are the historical block convention; `audio`
    # is the common shorthand. Keep this list in sync with the schedule
    # mixin and any future caller — silent key-mismatches were the original
    # production bug (caller used `audio_path`, block read only `file_path`).
    _STT_PATH_KEYS = ("file_path", "audio_path", "path", "audio")

    ui_schema = {
        "input": {
            "type": "audio",
            "accept": [".mp3", ".wav", ".webm", ".m4a"],
            "placeholder": "Record or upload audio...",
            "multiline": False,
        },
        "output": {
            "type": "text",
            "fields": [{"name": "text", "type": "text", "label": "Transcription"}],
        },
        "quick_actions": [
            {"icon": "🔊", "label": "Text to Speech", "prompt": "Convert this to speech"},
            {"icon": "🎤", "label": "Transcribe", "prompt": "Transcribe audio"},
        ],
    }

    async def process(self, input_data: Any, params: dict = None) -> dict:
        params = params or {}
        operation = params.get("operation", params.get("action", "tts"))

        # Resolve text / file_path / inline base64 from various input shapes
        text = ""
        file_path = None
        audio_b64 = None
        if isinstance(input_data, str):
            if os.path.exists(input_data):
                file_path = input_data
            else:
                text = input_data
        elif isinstance(input_data, dict):
            text = input_data.get("text") or input_data.get("input") or ""
            for key in self._STT_PATH_KEYS:
                v = input_data.get(key)
                if v:
                    file_path = v
                    break
            audio_b64 = input_data.get("audio_base64") or input_data.get("audio_data")
            operation = input_data.get("operation", operation)

        # On STT, the platform's InputAdapter wraps a positional string into
        # {"text": "<path>"} (it can't distinguish a TTS phrase from an STT
        # path at the wrapper layer). Recover by treating `text` as a path
        # when it actually exists on disk and the caller asked for
        # transcription. This is what makes `voice_block.execute(path, …)`
        # work transparently for callers like construction.daily_site_report.
        if operation in ("stt", "transcribe") and not file_path and text and os.path.exists(text):
            file_path = text
            text = ""

        # ── TTS ──────────────────────────────────────────────────────────────
        if operation == "tts":
            if not text:
                return {"status": "error", "error": "Text required for TTS"}
            lang = params.get("lang", params.get("language", "en"))
            if lang not in _SUPPORTED_LANGS:
                lang = "en"

            try:
                loop = asyncio.get_event_loop()
                audio_bytes, fmt = await loop.run_in_executor(None, _tts_sync, text[:3000], lang)
                b64 = base64.b64encode(audio_bytes).decode()
                return {
                    "status": "success",
                    "operation": "tts",
                    "audio_base64": b64,
                    "format": fmt,
                    "lang": lang,
                    "lang_name": _SUPPORTED_LANGS[lang],
                    "chars": len(text),
                    "size_bytes": len(audio_bytes),
                    "note": "Decode audio_base64 and play as MP3",
                }
            except Exception as e:
                return {"status": "error", "error": str(e), "operation": "tts"}

        # ── STT ──────────────────────────────────────────────────────────────
        elif operation in ("stt", "transcribe"):
            # Materialise inline base64 audio to a temp WAV so the underlying
            # speech_recognition.AudioFile can read it. Without this branch
            # any caller that uploads a raw recording (the browser UI does)
            # would fail with the bare "file_path required" message even
            # though the audio is present in the payload.
            cleanup_tmp: str | None = None
            if not file_path and audio_b64:
                try:
                    raw = base64.b64decode(audio_b64, validate=False)
                except Exception as e:
                    return {"status": "error", "error": f"audio_base64 decode failed: {e}", "operation": "stt"}
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(raw)
                    file_path = f.name
                    cleanup_tmp = f.name
            if not file_path:
                return {
                    "status": "error",
                    "error": "audio source required for STT — pass file_path/audio_path/path or audio_base64 (WAV/AIFF/FLAC)",
                    "operation": "stt",
                    "accepted_keys": list(self._STT_PATH_KEYS) + ["audio_base64", "audio_data"],
                }
            if not os.path.exists(file_path):
                return {"status": "error", "error": f"Audio file not found: {file_path}", "operation": "stt"}

            # ── Encryption at rest (F29) ────────────────────────────────────
            # Platform uploads are encrypted on disk (file_crypto); pydub and
            # speech_recognition must read PLAINTEXT. The stored path used to
            # be fed straight to the decoder, which chewed ciphertext and
            # blamed ffmpeg ("install ffmpeg and retry" — ffmpeg was fine).
            # Materialise a decrypted temp copy for the STT chain; plain
            # local files pass through untouched (open_plaintext yields the
            # original path for unencrypted files).
            try:
                import shutil

                from app.core.file_crypto import open_plaintext, shred_file
                with open_plaintext(file_path) as _plain:
                    if str(_plain) != str(file_path):
                        suffix = os.path.splitext(file_path)[1] or ".bin"
                        # fork_dec_ prefix: the plaintext sweeper
                        # (file_crypto.sweep_stale_plaintext) reaps orphans
                        # of this pattern, so a crash here has a bounded
                        # plaintext lifetime instead of an indefinite one.
                        with tempfile.NamedTemporaryFile(
                                suffix=suffix, prefix="fork_dec_",
                                delete=False) as tf:
                            decrypted_tmp = tf.name
                        shutil.copyfile(_plain, decrypted_tmp)
                        if cleanup_tmp:
                            try:
                                os.unlink(cleanup_tmp)
                            except OSError:
                                # Never silent: this temp file holds DECRYPTED
                                # audio. Shred the contents so nothing readable
                                # remains even if the name survives.
                                shredded = shred_file(cleanup_tmp)
                                _LOG.warning(
                                    "could not remove decrypted temp file %s — "
                                    "contents %s",
                                    cleanup_tmp,
                                    "zero-overwritten" if shredded
                                    else "COULD NOT be overwritten; plaintext "
                                         "may remain on disk",
                                    exc_info=True,
                                )
                        file_path = decrypted_tmp
                        cleanup_tmp = decrypted_tmp
            except Exception as e:  # noqa: BLE001 — a decrypt failure is a clear error, not a crash
                return {"status": "error", "operation": "stt",
                        "error": f"could not decrypt stored audio: {e}"}

            # ── Browser/mobile format conversion (2.2) ──────────────────────
            # speech_recognition.AudioFile only reads WAV/AIFF/FLAC. The
            # browser push-to-talk path emits WebM (Chrome/Firefox), MP3,
            # m4a (Safari), or Ogg/Opus — so without a conversion step
            # field engineers' recordings would all fail at STT.
            #
            # When the source extension is non-native AND pydub is
            # available, transcode to a 16 kHz mono WAV in a temp file
            # and point STT at that. Track the temp path so it's cleaned
            # up regardless of success/error.
            original_path = file_path
            source_format = os.path.splitext(file_path)[1].lower()
            converted = False
            if source_format in _TRANSCODE_STT_FORMATS:
                if not _pydub_available():
                    return {
                        "status": "error",
                        "error": (
                            f"audio format {source_format!r} requires conversion to WAV "
                            f"but pydub is not installed. Install pydub + ffmpeg, or "
                            f"upload one of: {sorted(_NATIVE_STT_FORMATS)}."
                        ),
                        "operation": "stt",
                        "source_format": source_format,
                    }
                try:
                    loop = asyncio.get_event_loop()
                    converted_path = await loop.run_in_executor(
                        None, _transcode_to_wav_sync, file_path,
                    )
                    file_path = converted_path
                    # Chain the converted temp file into cleanup_tmp so
                    # both paths (base64 + transcode) free their tmps.
                    if cleanup_tmp:
                        # base64 wrote its own tmp; transcode produced
                        # another. Clean up the older one immediately
                        # so we don't leak.
                        try:
                            os.unlink(cleanup_tmp)
                        except OSError:
                            _LOG.debug(
                                "swallowed %s in process() — continuing",
                                "OSError", exc_info=True,
                            )
                    cleanup_tmp = converted_path
                    converted = True
                    _LOG.info("voice STT: converted %s (%s) -> %s via pydub",
                              os.path.basename(original_path), source_format,
                              os.path.basename(converted_path))
                except RuntimeError as e:
                    return {
                        "status": "error",
                        "error": str(e),
                        "operation": "stt",
                        "source_format": source_format,
                    }

            try:
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(None, _stt_sync, file_path)
                response = {
                    "status": "success",
                    "operation": "stt",
                    "text": transcript,
                    "file": os.path.basename(original_path),
                }
                if converted:
                    response["converted_from"] = source_format
                return response
            except Exception as e:
                # Google STT raises UnknownValueError with an empty message
                # on silent/unintelligible audio. Surface a meaningful
                # error so daily-report callers don't see a bare "" string.
                msg = str(e) or f"{type(e).__name__}: provider returned no transcription"
                return {"status": "error", "error": msg, "operation": "stt"}
            finally:
                if cleanup_tmp:
                    try:
                        os.unlink(cleanup_tmp)
                    except OSError:
                        _LOG.debug(
                            "swallowed %s in process() — continuing",
                            "OSError", exc_info=True,
                        )

        # ── Languages list ────────────────────────────────────────────────────
        elif operation == "languages":
            return {"status": "success", "languages": _SUPPORTED_LANGS}

        return {"status": "error", "error": f"Unknown operation: {operation}. Use: tts, stt, languages"}
