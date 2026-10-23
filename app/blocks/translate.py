"""Translate Block - Google Translate via public HTTP API (no API key)."""

import asyncio
import time
from typing import Any, Dict, Tuple

import requests

from app.core.universal_base import UniversalBlock

_LANG_CODES = {
    "english": "en", "spanish": "es", "arabic": "ar", "french": "fr",
    "german": "de", "chinese": "zh-CN", "japanese": "ja", "hindi": "hi",
    "portuguese": "pt", "russian": "ru", "turkish": "tr", "korean": "ko",
    "italian": "it", "dutch": "nl", "polish": "pl", "thai": "th",
    "vietnamese": "vi", "indonesian": "id", "malay": "ms", "ukrainian": "uk",
}

_GTX_URL = "https://translate.googleapis.com/translate_a/single"
_MAX_TEXT_LEN = 5000
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 3
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_USER_AGENT = "CerebrumBlocks/2.0 (translate; +https://github.com/bopoadz-del/The_Fork)"


def _normalize_lang(lang: str) -> str:
    if not lang:
        return "en"
    l = lang.lower().strip()
    return _LANG_CODES.get(l, l)


def _parse_google_response(payload: Any, source: str) -> Tuple[str, str]:
    """Extract translated text and detected language from Google GTX JSON."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("Unexpected translate response shape")

    segments = payload[0] if payload[0] else []
    translated = "".join(part[0] for part in segments if part and part[0])
    if not translated.strip():
        raise ValueError("Empty translation in response")

    detected = source
    if source == "auto" and len(payload) > 2 and payload[2]:
        detected = payload[2]
    return translated, detected


def _mock_translate(text: str, source: str, target: str) -> Tuple[str, str]:
    """Deterministic offline translation for tests and CI (provider=mock)."""
    detected = "en" if source == "auto" else source
    return f"[{target}] {text}", detected


def _google_translate_request(text: str, source: str, target: str) -> Tuple[str, str]:
    """Call the public GTX endpoint (same path deep-translator used)."""
    params = {
        "client": "gtx",
        "sl": source,
        "tl": target,
        "dt": "t",
        "q": text,
    }
    headers = {"User-Agent": _USER_AGENT}
    last_error = "Translation request failed"

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                _GTX_URL,
                params=params,
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code in _RETRYABLE_STATUS:
                last_error = f"Translation service unavailable (HTTP {resp.status_code})"
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            return _parse_google_response(resp.json(), source)

        except requests.Timeout as exc:
            last_error = "Translation request timed out"
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise TimeoutError(last_error) from exc

        except requests.RequestException as exc:
            last_error = f"Translation request failed: {exc}"
            if attempt < _MAX_RETRIES - 1 and (
                isinstance(exc, requests.ConnectionError)
                or (getattr(exc, "response", None) is not None
                    and exc.response.status_code in _RETRYABLE_STATUS)
            ):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error) from exc

        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    raise RuntimeError(last_error)


def _translate_sync(text: str, source: str, target: str, *, use_mock: bool = False) -> tuple[str, str]:
    if use_mock:
        return _mock_translate(text, source, target)
    return _google_translate_request(text, source, target)


class TranslateBlock(UniversalBlock):
    """Multi-language translation via Google Translate (HTTP, no API key)"""

    auto_validate = False
    name = "translate"
    version = "2.0"
    description = "Translate text between 20+ languages — no API key needed"
    layer = 3
    tags = ["domain", "nlp", "translation"]
    requires = []
    required_input_one_of = ["text", "input"]

    # Canonical text key for chain unwrapping — overrides the orchestrator's
    # global priority list. Without this, a translate -> chat chain leans on
    # the fact that "translated" happens to be in _TEXT_OUTPUT_FIELDS; with
    # this, the contract is explicit and survives any list reordering.
    # See CONTRIBUTING.md "Block output contracts".
    text_output_field = "translated"

    ui_schema = {
        "input": {
            "type": "text",
            "accept": None,
            "placeholder": "Enter text to translate...",
            "multiline": True,
        },
        "output": {
            "type": "text",
            "fields": [
                {"name": "translated", "type": "text", "label": "Translation"},
                {"name": "source_language", "type": "text", "label": "From"},
                {"name": "target_language", "type": "text", "label": "To"},
            ],
        },
        "quick_actions": [
            {"icon": "🇪🇸", "label": "To Spanish", "prompt": "Translate to Spanish: "},
            {"icon": "🇸🇦", "label": "To Arabic", "prompt": "Translate to Arabic: "},
            {"icon": "🇫🇷", "label": "To French", "prompt": "Translate to French: "},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}

        text = ""
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, dict):
            text = (
                input_data.get("text")
                or input_data.get("input")
                or input_data.get("message")
                or ""
            )
        text = text.strip()

        if params.get("operation") == "languages":
            return {"status": "success", "languages": _LANG_CODES}

        if not text:
            return {"status": "error", "error": "Text is required"}

        # The language may arrive in the input dict alongside the text (the
        # /v1/execute shape agents naturally produce). Ignoring it there and
        # silently defaulting to Spanish translated to the WRONG LANGUAGE
        # while reporting success -- params still win when both are given.
        _in = input_data if isinstance(input_data, dict) else {}
        target = _normalize_lang(
            params.get("target") or params.get("target_language")
            or _in.get("target") or _in.get("target_language") or "es"
        )
        source = _normalize_lang(
            params.get("source") or params.get("source_language")
            or _in.get("source") or _in.get("source_language") or "auto"
        )
        use_mock = params.get("provider") == "mock"

        try:
            loop = asyncio.get_event_loop()
            translated, detected = await loop.run_in_executor(
                None,
                lambda: _translate_sync(text[:_MAX_TEXT_LEN], source, target, use_mock=use_mock),
            )
            result = {
                "status": "success",
                "original": text,
                "translated": translated,
                "source_language": detected if source == "auto" else source,
                "target_language": target,
                "char_count": len(text),
            }
            if use_mock:
                result["provider"] = "mock"
            return result
        except TimeoutError as e:
            return {"status": "error", "error": str(e), "target": target, "retryable": True}
        except Exception as e:
            return {"status": "error", "error": str(e), "target": target}
