"""Tests for Translate Block."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.blocks import TranslateBlock
from app.blocks.translate import (
    _google_translate_request,
    _mock_translate,
    _parse_google_response,
    _translate_sync,
)


@pytest.fixture
def translate_block():
    return TranslateBlock()


@pytest.mark.asyncio
async def test_translate_block_execute_structure(translate_block):
    """Test that Translate block returns standardized JSON structure."""
    result = await translate_block.execute(
        "Hello world",
        {"target": "es", "provider": "mock"},
    )

    assert "block" in result
    assert result["block"] == "translate"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_translate_block_metadata(translate_block):
    """Test Translate block metadata."""
    assert translate_block.name == "translate"
    assert translate_block.version == "2.0"


@pytest.mark.asyncio
async def test_translate_block_chaining(translate_block):
    """Test Translate block can receive output from previous block."""
    previous_result = {
        "result": {
            "text": "This is a document in English.",
        },
    }

    result = await translate_block.execute(
        previous_result,
        {"target": "fr", "provider": "mock"},
    )

    assert result["block"] == "translate"
    assert "result" in result


@pytest.mark.asyncio
async def test_translate_mock_provider(translate_block):
    """Mock provider must not hit the network and returns deterministic text."""
    result = await translate_block.process("Hello", {"target": "es", "provider": "mock"})

    assert result["status"] == "success"
    assert result["translated"] == "[es] Hello"
    assert result["provider"] == "mock"
    assert result["target_language"] == "es"


@pytest.mark.asyncio
async def test_translate_languages_operation(translate_block):
    result = await translate_block.process("", {"operation": "languages"})

    assert result["status"] == "success"
    assert "english" in result["languages"]
    assert result["languages"]["spanish"] == "es"


@pytest.mark.asyncio
async def test_translate_empty_text(translate_block):
    result = await translate_block.process("   ", {"target": "es", "provider": "mock"})

    assert result["status"] == "error"
    assert result["error"] == "Text is required"


def test_parse_google_response_auto_detect():
    payload = [[["Hola", "Hello", None, 0.9]], None, "en"]
    translated, detected = _parse_google_response(payload, "auto")

    assert translated == "Hola"
    assert detected == "en"


def test_parse_google_response_rejects_empty():
    with pytest.raises(ValueError, match="Empty translation"):
        _parse_google_response([[]], "en")


def test_mock_translate_auto_source():
    translated, detected = _mock_translate("Hi", "auto", "fr")

    assert translated == "[fr] Hi"
    assert detected == "en"


@patch("app.blocks.translate.time.sleep")
@patch("app.blocks.translate.requests.get")
def test_google_translate_request_retries_on_503(mock_get, mock_sleep):
    bad = MagicMock(status_code=503)
    bad.raise_for_status.side_effect = requests.HTTPError(response=bad)
    good = MagicMock(status_code=200)
    good.json.return_value = [[["Hola", "Hello", None, 0.9]], None, "en"]
    mock_get.side_effect = [bad, good]

    translated, detected = _google_translate_request("Hello", "en", "es")

    assert translated == "Hola"
    assert detected == "en"
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(1)


@patch("app.blocks.translate.time.sleep")
@patch("app.blocks.translate.requests.get")
def test_google_translate_request_timeout(mock_get, mock_sleep):
    mock_get.side_effect = requests.Timeout("slow")

    with pytest.raises(TimeoutError, match="timed out"):
        _google_translate_request("Hello", "en", "es")

    assert mock_get.call_count == 3
    assert mock_sleep.call_count == 2


@patch("app.blocks.translate.requests.get")
def test_google_translate_request_malformed_json(mock_get):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"error": "bad"}
    mock_get.return_value = resp

    with pytest.raises(RuntimeError, match="Unexpected translate response"):
        _google_translate_request("Hello", "en", "es")


def test_translate_sync_routes_mock_flag():
    translated, detected = _translate_sync("Hi", "auto", "de", use_mock=True)

    assert translated == "[de] Hi"
    assert detected == "en"
