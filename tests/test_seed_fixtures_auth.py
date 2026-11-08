"""seed_fixtures._auth_header resolves its credential through the ONE auth
implementation (scripts.fork_cli.resolve_auth), so a stale FORK_TOKEN /
FORK_API_KEY falls through to a valid FORK_EMAIL+FORK_PASSWORD instead of
wedging the seeder (ORDERS_07NOV_0840Z M10). Mocked API, no network.
"""
from __future__ import annotations

import pytest

from scripts import fork_cli, seed_fixtures

_BASE = "https://example.test"
_LOGIN_TOK = "login-session-token"


class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._p = payload or {}

    def json(self) -> dict:
        return self._p

    def raise_for_status(self) -> None:
        return None


def _install_fake_api(monkeypatch, *, accepted: set[str], valid_login: tuple[str, str] | None):
    def fake_get(url, headers=None, timeout=None):
        bearer = (headers or {}).get("Authorization", "").removeprefix("Bearer ")
        return _Resp(200 if bearer in accepted else 401)

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/v1/users/login") and valid_login \
                and (json or {}).get("email") == valid_login[0] \
                and (json or {}).get("password") == valid_login[1]:
            return _Resp(200, {"token": _LOGIN_TOK})
        return _Resp(401)

    monkeypatch.setattr(fork_cli.httpx, "get", fake_get)
    monkeypatch.setattr(fork_cli.httpx, "post", fake_post)


def _clear_env(monkeypatch):
    for k in ("FORK_TOKEN", "FORK_API_KEY", "FORK_EMAIL", "FORK_PASSWORD"):
        monkeypatch.delenv(k, raising=False)


def test_seed_auth_uses_one_implementation_and_falls_through(monkeypatch):
    # A stale api key (rejected 401) + a valid login -> the login token is used.
    _clear_env(monkeypatch)
    monkeypatch.setenv("FORK_API_KEY", "stale-key")
    monkeypatch.setenv("FORK_EMAIL", "e@x")
    monkeypatch.setenv("FORK_PASSWORD", "pw")
    _install_fake_api(monkeypatch, accepted={_LOGIN_TOK}, valid_login=("e@x", "pw"))

    header = seed_fixtures._auth_header(_BASE)

    assert header == {"Authorization": f"Bearer {_LOGIN_TOK}"}


def test_seed_auth_names_only_labels_not_values(monkeypatch, capsys):
    # A stale token with no login configured -> SystemExit naming the LABEL
    # (never the value), the same contract fork_cli.resolve_auth already holds.
    _clear_env(monkeypatch)
    monkeypatch.setenv("FORK_API_KEY", "stale-key")
    _install_fake_api(monkeypatch, accepted=set(), valid_login=None)

    with pytest.raises(SystemExit) as ei:
        seed_fixtures._auth_header(_BASE)

    assert "FORK_API_KEY" in str(ei.value)
    assert "stale-key" not in str(ei.value)
