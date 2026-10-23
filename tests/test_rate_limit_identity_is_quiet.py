"""Rate-limit identification must be silent on the NORMAL path.

Live-fire campaign, run 1 (2026-08-14): production logs carried one full
traceback per request --

    WARNING app.main swallowed Exception in _rate_limit_identity() - continuing
    jwt.exceptions.DecodeError: Not enough segments

An API key is not a JWT, so ``decode_token`` raised for every key-authed
request. The result was correct (fall through to hashing the key) but it was
reached by exception, and the handler logged ``exc_info=True`` every time.
Real warnings were buried under per-request noise.

These tests fence the OBSERVABILITY property, not the return value: the normal
paths must emit no warning, while a genuinely unexpected failure must still be
loud. A test that only checked the returned identity would have stayed green
through the entire incident.
"""

from app import main as app_main
from tests._fixture_values import API_KEY_A, API_KEY_B, API_KEY_C, GARBAGE_TOKEN


class _Req:
    """Minimal stand-in for starlette Request — only what the function reads."""

    def __init__(self, authz: str | None = None, host: str = "10.0.0.1"):
        self.headers = {"Authorization": authz} if authz else {}
        self.client = type("C", (), {"host": host})()


class _StubLogger:
    """Records warning CALLS. Not a logging.Handler on purpose.

    Two earlier attempts measured the logging framework instead of the code and
    both passed alone while failing inside the full CI suite:

    1. ``caplog`` -- observes records only if they PROPAGATE to root, so any
       earlier test that touched propagation or handlers changed the answer.
    2. A handler attached to ``app.main``'s logger -- still subject to logger
       level, ``logging.disable()``, and whatever global state 3000-odd other
       tests leave behind.

    The property under test is "does this function report the failure?", and
    that is a question about the call, not about delivery. Substituting the
    module-level ``logger`` name that ``_rate_limit_identity`` closes over
    answers exactly that question and cannot be perturbed by ambient config.
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg, *args, **kwargs) -> None:
        try:
            self.warnings.append(str(msg) % args if args else str(msg))
        except (TypeError, ValueError):  # %-formatting must never fail a test
            self.warnings.append(str(msg))

    def __getattr__(self, _name):  # info/debug/error are irrelevant here
        return lambda *a, **k: None


def _identity(req, _unused=None):
    """Run the function with its module-level ``logger`` swapped for a stub."""
    stub = _StubLogger()
    real = app_main.logger
    app_main.logger = stub
    try:
        ident = app_main._rate_limit_identity(req)
    finally:
        app_main.logger = real
    _identity.records = stub.warnings
    return ident


def test_api_key_caller_is_identified_without_logging():
    """The regression: a non-JWT bearer token must not produce a warning."""
    ident = _identity(_Req("Bearer " + API_KEY_A))
    assert ident.startswith("key:"), ident
    assert not _identity.records, (
        "an API key is not a JWT - that is the normal path and must be silent; "
        f"got {_identity.records}"
    )


def test_jwt_caller_is_identified_as_user_without_logging(monkeypatch):
    monkeypatch.setattr(app_main._jwt_auth, "decode_token",
                        lambda _t: {"user_id": "u-42"})
    ident = _identity(_Req("Bearer a.b.c"))
    assert ident == "user:u-42"
    assert not _identity.records


def test_unexpected_failure_is_still_loud(monkeypatch):
    """The other direction: quieting the normal case must not silence real bugs.

    A missing signing secret is not "this isn't a JWT" — it must still warn,
    or the fix would have traded noise for blindness.
    """
    called: list[str] = []

    def _boom(_t):
        called.append(_t)
        raise RuntimeError("signing secret unavailable")

    monkeypatch.setattr(app_main._jwt_auth, "decode_token", _boom)
    ident = _identity(_Req("Bearer " + GARBAGE_TOKEN))
    assert ident.startswith("key:")
    # Diagnostic, not decoration: an earlier CI-only failure here was
    # indistinguishable between "the warning was not emitted" and "the patched
    # decode_token was never reached". Report both so a repeat names itself.
    assert called, (
        "decode_token was never invoked - the monkeypatch did not reach the "
        "function under test, so this says nothing about logging"
    )
    assert any("_rate_limit_identity" in m for m in _identity.records), (
        "an unexpected exception must still be logged loudly; "
        f"decode_token calls={len(called)} warnings={_identity.records}"
    )


def test_anonymous_caller_falls_back_to_ip():
    assert _identity(_Req(None, host="203.0.113.9")) == "ip:203.0.113.9"
    assert not _identity.records


def test_same_key_yields_a_stable_identity():
    """Rate limiting is meaningless if the identity churns per request."""
    a = _identity(_Req("Bearer " + API_KEY_B))
    b = _identity(_Req("Bearer " + API_KEY_B))
    c = _identity(_Req("Bearer " + API_KEY_C))
    assert a == b and a != c
