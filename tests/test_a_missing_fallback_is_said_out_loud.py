"""A chat with no usable fallback must say so -- at boot and on /health.

Live 2026-09-19: LLM_FALLBACK_PROVIDER on Render was `kimi`, a provider the
code had removed. It resolved back to the primary, so there was no fallback
at all, and nothing anywhere said so. An unset fallback key does the same.
"""
import logging

import pytest

from app.core.health_probes import probe_llm
from tests._fixture_values import STUB_TOKEN_06, STUB_TOKEN_08

_ENV = ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER", "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY", "OPENROUTER_MODEL")

# probe_llm() returns the two readiness booleans plus live provider telemetry.
# These are the ONLY non-readiness keys allowed; the readiness contract is
# asserted exactly after separating them.
_LLM_TELEMETRY = {"providers", "counts", "last_served_age_s", "serving", "error"}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_06)


def test_deepseek_with_an_openrouter_fallback_is_ready(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_08)
    out = probe_llm()
    # Exact readiness contract (PR-B had loosened this to a subset check).
    # probe_llm now also returns live telemetry keys; separate the known
    # telemetry, assert the readiness dict EXACTLY, and assert no UNKNOWN key
    # appears (so a future stray readiness key cannot slip in unnoticed).
    assert {k: v for k, v in out.items() if k not in _LLM_TELEMETRY} == {
        "primary_ready": True, "fallback_ready": True}
    assert set(out) - {"primary_ready", "fallback_ready"} <= _LLM_TELEMETRY


@pytest.mark.parametrize("fallback,key", [
    ("kimi", STUB_TOKEN_08),        # the live value: a removed provider
    ("deepseek", STUB_TOKEN_08),    # names the primary itself
    ("openrouter", None),       # the right provider, no key
    (None, STUB_TOKEN_08),          # unset
])
def test_a_fallback_that_cannot_work_is_reported_as_not_ready(monkeypatch, fallback, key):
    if fallback:
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", fallback)
    if key:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    assert probe_llm()["fallback_ready"] is False


def test_a_missing_primary_key_is_reported(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert probe_llm()["primary_ready"] is False


class _Capture(logging.Handler):
    """Attached straight to the logger: the app's logging setup may stop
    propagation to the root, which is where caplog listens."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def boot_log():
    handler, logger = _Capture(), logging.getLogger("app.main")
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)


def test_boot_warns_when_there_is_no_usable_fallback(monkeypatch, boot_log):
    from app.main import _warn_when_the_llm_has_no_fallback

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "kimi")
    _warn_when_the_llm_has_no_fallback()
    text = " ".join(boot_log.lines)
    assert "NO USABLE FALLBACK" in text
    assert STUB_TOKEN_06 not in text


def test_boot_is_quiet_when_the_fallback_works(monkeypatch, boot_log):
    from app.main import _warn_when_the_llm_has_no_fallback

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_08)
    _warn_when_the_llm_has_no_fallback()
    assert "FALLBACK" not in " ".join(boot_log.lines)


def test_health_shows_it_and_names_nothing(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "kimi")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_08)
    with TestClient(app) as c:
        body = c.get("/health").json()
    llm = body["checks"]["llm"]
    # Exact readiness contract restored (see note above); telemetry keys split
    # off, and no unknown key tolerated.
    assert {k: v for k, v in llm.items() if k not in _LLM_TELEMETRY} == {
        "primary_ready": True, "fallback_ready": False}
    assert set(llm) - {"primary_ready", "fallback_ready"} <= _LLM_TELEMETRY
    # The name-withholding guard now covers the WHOLE llm payload, including the
    # new providers/counts telemetry -- it must still name nothing (keys are
    # ROLE labels, never provider names).
    text = str(llm).lower()
    for word in ("deepseek", "openrouter", "kimi", "key", "test"):
        assert word not in text.replace("_ready", ""), word
