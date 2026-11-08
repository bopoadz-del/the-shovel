"""smoke-live preflight must fall through a rejected key via resolve_auth.

The workflow used to stop when FORK_API_KEY was refused, so a valid email
login sitting in the same secrets never ran. Two rejection statuses, and a
negative workflow step that still does that, lock the mechanism.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import fork_cli

_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_LIVE = _ROOT / ".github" / "workflows" / "smoke-live.yml"
_SMOKE_SH = _ROOT / "scripts" / "smoke.sh"
_BASE = "https://example.test"
_LOGIN_TOK = "login-session-token"


def _preflight_step(doc: dict) -> dict:
    for step in doc["jobs"]["smoke"]["steps"]:
        if str(step.get("name", "")).startswith("Preflight auth"):
            return step
    raise AssertionError("smoke-live has no preflight auth step")


def auth_fallthrough_problems(step: dict) -> list[str]:
    """Empty when the step supplies every slot and calls the shared resolver."""
    problems: list[str] = []
    env = step.get("env") or {}
    for key in ("FORK_API_KEY", "FORK_EMAIL", "FORK_PASSWORD", "FORK_TOKEN"):
        if key not in env:
            problems.append(f"missing {key}")
    run = step.get("run") or ""
    if "resolve_auth" not in run:
        problems.append("preflight does not call resolve_auth")
    if "FORK_API_KEY is wrong" in run:
        problems.append("preflight blames the api key and does not fall through")
    if "/v1/users/login" in run:
        problems.append("preflight implements its own login")
    return problems


@pytest.mark.parametrize(
    "needle,present",
    [
        ("resolve_auth", True),
        ("headers(cred)", True),
        ("key_len", False),
        ("FORK_API_KEY is wrong", False),
        ("/v1/users/login", False),
    ],
)
def test_smoke_live_preflight_uses_resolve_auth(needle, present):
    doc = yaml.safe_load(_SMOKE_LIVE.read_text(encoding="utf-8"))
    step = _preflight_step(doc)
    assert auth_fallthrough_problems(step) == []
    assert (needle in step["run"]) is present


@pytest.mark.parametrize(
    "path,must_contain,must_lack",
    [
        (_SMOKE_SH, "fork_cli.py", "/v1/users/login"),
        (_ROOT / "scripts" / "fork_cli.py", "def resolve_auth", "logged in as"),
    ],
)
def test_login_lives_only_in_resolve_auth(path, must_contain, must_lack):
    text = path.read_text(encoding="utf-8")
    assert must_contain in text
    assert must_lack not in text


@pytest.mark.parametrize(
    "step,clean",
    [
        (
            {
                "name": "Preflight auth",
                "env": {
                    "FORK_API_KEY": "x",
                    "FORK_EMAIL": "x",
                    "FORK_PASSWORD": "x",
                    "FORK_TOKEN": "x",
                },
                "run": "resolve_auth(args)\nheaders(cred)\n",
            },
            True,
        ),
        (
            {
                "name": "Preflight auth",
                "env": {"FORK_API_KEY": "x"},
                "run": "sys.exit('FORK_API_KEY is wrong')\n",
            },
            False,
        ),
    ],
)
def test_fallthrough_checker_accepts_two_step_shapes(step, clean):
    assert (auth_fallthrough_problems(step) == []) is clean


class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._p = payload or {}

    def json(self) -> dict:
        return self._p

    def raise_for_status(self) -> None:
        return None


@pytest.mark.parametrize("rejected", [401, 403])
def test_rejected_api_key_falls_through_to_email_login(monkeypatch, capsys, rejected):
    for key in ("FORK_TOKEN", "FORK_API_KEY", "FORK_EMAIL", "FORK_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FORK_API_KEY", "stale-key")
    monkeypatch.setenv("FORK_EMAIL", "e@x")
    monkeypatch.setenv("FORK_PASSWORD", "pw")

    def fake_get(url, headers=None, timeout=None):
        bearer = (headers or {}).get("Authorization", "").removeprefix("Bearer ")
        if bearer == "stale-key":
            return _Resp(rejected)
        if bearer == _LOGIN_TOK:
            return _Resp(200)
        return _Resp(401)

    def fake_post(url, json=None, timeout=None):
        body = json or {}
        if body.get("email") == "e@x" and body.get("password") == "pw":
            return _Resp(200, {"token": _LOGIN_TOK})
        return _Resp(401)

    monkeypatch.setattr(fork_cli.httpx, "get", fake_get)
    monkeypatch.setattr(fork_cli.httpx, "post", fake_post)
    args = SimpleNamespace(
        token=None, api_key=None, email=None, password=None, base=_BASE,
    )
    assert fork_cli.resolve_auth(args) == _LOGIN_TOK
    err = capsys.readouterr().err
    assert "FORK_EMAIL" in err
    assert "e@x" not in err
    assert "pw" not in err
    assert "stale-key" not in err
