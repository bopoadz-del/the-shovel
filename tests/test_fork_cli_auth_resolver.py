"""fork_cli.resolve_auth tries each credential in precedence order and VERIFIES
it against the live API, falling through on 401/403 — so a stale FORK_API_KEY no
longer wedges the clients when a valid FORK_EMAIL+FORK_PASSWORD is also set
(ORDERS_02NOV_0310Z). Mocked server, no network. Full dotted import.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import fork_cli

_BASE = "https://example.test"
_LOGIN_TOK = "login-session-token"


class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._p = payload or {}

    def json(self) -> dict:
        return self._p

    def raise_for_status(self) -> None:
        return None  # only ever called on a 2xx in these paths


def _install_fake_api(monkeypatch, *, accepted: set[str], valid_login: tuple[str, str] | None):
    """A fake live API: GET /v1/projects returns 200 only for an accepted bearer
    (else 401); POST /v1/users/login returns a token for the valid pair."""
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


@pytest.mark.parametrize("env,accepted,valid_login,expected", [
    # a STALE api key (rejected 401) with a valid login -> the login token is used
    ({"FORK_API_KEY": "stale-key", "FORK_EMAIL": "e@x", "FORK_PASSWORD": "pw"},
     {_LOGIN_TOK}, ("e@x", "pw"), _LOGIN_TOK),
    # a valid token is accepted on the probe -> used directly, no login
    ({"FORK_TOKEN": "good-token"},
     {"good-token"}, None, "good-token"),
])
def test_resolve_auth_verifies_and_falls_through(monkeypatch, env, accepted, valid_login, expected):
    _clear_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _install_fake_api(monkeypatch, accepted=accepted, valid_login=valid_login)
    args = SimpleNamespace(token=None, api_key=None, email=None, password=None, base=_BASE)
    assert fork_cli.resolve_auth(args) == expected


def test_resolve_auth_exits_when_all_rejected(monkeypatch):
    # stale key, no login configured -> no accepted credential -> SystemExit,
    # naming the label tried (never the value).
    _clear_env(monkeypatch)
    monkeypatch.setenv("FORK_API_KEY", "stale-key")
    _install_fake_api(monkeypatch, accepted=set(), valid_login=None)
    args = SimpleNamespace(token=None, api_key=None, email=None, password=None, base=_BASE)
    with pytest.raises(SystemExit) as ei:
        fork_cli.resolve_auth(args)
    assert "FORK_API_KEY" in str(ei.value) and "stale-key" not in str(ei.value)
