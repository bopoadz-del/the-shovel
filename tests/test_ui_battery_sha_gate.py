"""R11 sha gate (ui_battery_runner._await_live_sha): a battery run must measure
the code the deploy shipped. The gate polls the live /health build_sha and
refuses when the live service is not serving the expected commit."""
from __future__ import annotations

import pytest

import scripts.ui_battery_runner as r


class _Resp:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, sha: str, status: int = 200) -> None:
        self._sha = sha
        self._status = status
        self.calls = 0

    def get(self, url: str, timeout: float | None = None) -> _Resp:
        self.calls += 1
        return _Resp(self._status, {"build_sha": self._sha})


def test_returns_immediately_when_live_matches_expected():
    c = _Client("abc123")
    assert r._await_live_sha(c, "https://x", "abc123", wait_s=5) == "abc123"
    assert c.calls == 1  # matched on the first probe; no polling loop


def test_no_expected_records_live_and_does_not_gate():
    # A manual dispatch carries no EXPECTED_SHA: record whatever is live, never fail.
    c = _Client("def456")
    assert r._await_live_sha(c, "https://x", None, wait_s=5) == "def456"


def test_refuses_when_live_never_reaches_expected():
    # The live service serves an older commit: the run would measure the wrong
    # code, so the gate raises rather than returning. wait_s=0 => decide on the
    # first probe without real waiting.
    c = _Client("oldsha")
    with pytest.raises(SystemExit):
        r._await_live_sha(c, "https://x", "newsha", wait_s=0)
