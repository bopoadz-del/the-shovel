"""SSRF guard — the web and webhook blocks must refuse to reach private,
loopback, or link-local (cloud-metadata) addresses, or non-http schemes."""

import pytest

from app.core.url_guard import UnsafeURLError, validate_public_url


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
    "http://127.0.0.1:8000/admin",               # loopback
    "http://localhost/",                         # loopback by name
    "http://10.0.0.5/",                          # private range
    "http://192.168.1.1/",                       # private range
    "http://[::1]/",                             # IPv6 loopback
    "file:///etc/passwd",                        # non-http scheme
    "ftp://example.com/x",                       # non-http scheme
    "",                                          # empty
    "not-a-url",                                 # no scheme / host
])
def test_unsafe_urls_are_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_public_url(url)


def test_public_url_is_accepted():
    # A literal public IP — deterministic, needs no DNS.
    assert validate_public_url("http://8.8.8.8/") == "http://8.8.8.8/"


async def test_web_block_refuses_metadata_endpoint():
    from app.blocks.web import WebBlock

    result = await WebBlock().process("http://169.254.169.254/latest/meta-data/")
    assert result["status"] == "error", result
    assert "non-public" in result["error"] or "address" in result["error"]


async def test_webhook_block_refuses_internal_url():
    from app.blocks.webhook import WebhookBlock

    result = await WebhookBlock(None, {}).process(
        {"action": "send", "url": "http://127.0.0.1:9999/", "payload": {"x": 1}}
    )
    assert "error" in result, result
    assert "Unsafe webhook URL" in result["error"]
