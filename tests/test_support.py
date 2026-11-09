"""Tests for the in-app support form (app/routers/support.py).

Every decision is parametrized with >= 2 distinct inputs so the mutation
gate's two-input lint holds and the CI-M mutants die.
"""
from __future__ import annotations

import sys
import types

import pytest
from fastapi import HTTPException

from app.core import email as email_svc
from app.routers import support


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("help@example.com", "help@example.com"),   # owner-set wins
        ("", "support@theshovel.ai"),               # unset -> default
        ("   ", "support@theshovel.ai"),            # blank -> default
    ],
)
def test_support_address(monkeypatch, env_value, expected):
    monkeypatch.setenv("SUPPORT_EMAIL", env_value)
    assert support.support_address() == expected


@pytest.mark.parametrize(
    "env_value, expected_domain",
    [
        ("support@theshovel.ai", "theshovel.ai"),
        ("help@sub.example.com", "sub.example.com"),
    ],
)
def test_support_domain(monkeypatch, env_value, expected_domain):
    monkeypatch.setenv("SUPPORT_EMAIL", env_value)
    assert support.support_domain() == expected_domain


class _FakeResolver:
    def __init__(self, answers, exc=None):
        self._answers = answers
        self._exc = exc

    def resolve(self, domain, type, lifetime):
        if self._exc is not None:
            raise self._exc
        return self._answers


def _install_fake_dns(monkeypatch, answers, exc=None):
    # `import dns.resolver` resolves the submodule; the function then reads
    # the `resolver` attribute off the `dns` name, so both keys must be faked.
    fake_resolver = types.ModuleType("dns.resolver")
    fake_resolver.resolve = _FakeResolver(answers, exc).resolve
    fake_dns = types.ModuleType("dns")
    fake_dns.resolver = fake_resolver
    monkeypatch.setitem(sys.modules, "dns", fake_dns)
    monkeypatch.setitem(sys.modules, "dns.resolver", fake_resolver)


@pytest.mark.parametrize(
    "answers, expected",
    [
        (["mx1.example.com."], True),   # has MX
        ([], False),                    # no MX
    ],
)
def test_support_domain_has_mx(monkeypatch, answers, expected):
    _install_fake_dns(monkeypatch, answers)
    assert support.support_domain_has_mx("example.com") is expected


def test_support_domain_has_mx_lookup_failure_is_false(monkeypatch):
    _install_fake_dns(monkeypatch, [], exc=RuntimeError("nxdomain"))
    assert support.support_domain_has_mx("example.com") is False


@pytest.mark.parametrize(
    "send_effect, expected_status",
    [
        ("ok", 200),
        ("not_configured", 503),
        ("send_failed", 503),
    ],
)
async def test_submit_support(monkeypatch, send_effect, expected_status):
    async def fake_send(to, subject, html):
        if send_effect == "ok":
            return "msg-123"
        if send_effect == "not_configured":
            raise email_svc.EmailNotConfigured("missing key")
        raise email_svc.EmailSendFailed("provider refused")

    monkeypatch.setattr(email_svc, "send_email", fake_send)
    req = support.SupportRequest(message="I cannot sign in", contact_email="a@b.com")

    if expected_status == 200:
        out = await support.submit_support(req)
        assert out["ok"] is True
        assert out["message_id"] == "msg-123"
    else:
        with pytest.raises(HTTPException) as excinfo:
            await support.submit_support(req)
        assert excinfo.value.status_code == expected_status


@pytest.mark.parametrize(
    "message, contact, ok",
    [
        ("hi", None, True),
        ("hi", "a@b.com", True),
    ],
)
async def test_submit_support_email_contents(monkeypatch, message, contact, ok):
    sent = {}

    async def fake_send(to, subject, html):
        sent.update(to=to, subject=subject, html=html)
        return "msg-1"

    monkeypatch.setattr(email_svc, "send_email", fake_send)
    req = support.SupportRequest(message=message, contact_email=contact)
    await support.submit_support(req)
    assert sent["to"] == support.support_address()
    assert message in sent["html"]
    if contact:
        assert contact in sent["html"]
