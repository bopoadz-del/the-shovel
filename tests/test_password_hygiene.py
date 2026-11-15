"""SEC-2 exam: a password never outlives the request that carries it.

Two halves:
- The server: a rejected request must not echo what was submitted (a 422
  for a short password used to send the password straight back in
  ``errors[].input``, into the browser's network log and any proxy log).
- The browser source (frontend/src, read as text so the check runs in the
  Python suite): no password reaches web storage or the console; every
  password field takes its autocomplete hint from the one shared rule; the
  sign-in form forgets the password once it has been sent.

Every rule is checked twice: on invented snippets (one that must pass, one
that must fail) and then on the real tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "src"
LOGIN = SRC / "pages" / "Login.tsx"
RULE = SRC / "lib" / "passwordField.ts"


# ── the server never echoes a submitted value ────────────────────────────

@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "path,body,secret",
    [
        ("/v1/users/register", {"email": "a@example.invalid", "password": "Sh0rt7x"}, "Sh0rt7x"),
        ("/v1/users/register", {"email": "b@example.invalid", "password": "pw!2"}, "pw!2"),
        ("/v1/users/reset-password", {"token": "t", "new_password": "N3w!sh"}, "N3w!sh"),
        ("/v1/users/login", {"email": 7, "password": "Inv3nted-login-pw"}, "Inv3nted-login-pw"),
    ],
)
def test_a_rejected_request_does_not_echo_the_password(client, path, body, secret):
    r = client.post(path, json=body)
    assert r.status_code == 422, r.text
    assert secret not in r.text


@pytest.mark.parametrize(
    "path,body,field",
    [
        ("/v1/users/register", {"email": "a@example.invalid", "password": "Sh0rt7x"}, "password"),
        ("/v1/users/register", {"password": "L0ng-enough-pw"}, "email"),
    ],
)
def test_a_rejected_request_still_names_the_field(client, path, body, field):
    r = client.post(path, json=body)
    assert r.status_code == 422
    errors = r.json()["error"]["details"]["errors"]
    locs = [e["loc"][-1] for e in errors]
    assert field in locs
    assert all("msg" in e and "input" not in e for e in errors)


# ── the browser source: scanners ──────────────────────────────────────────

_PASSWORD_WORD = re.compile(r"passw", re.I)
_STORAGE_WRITE = re.compile(r"\b(?:localStorage|sessionStorage|document\.cookie)\b[^;\n]*")
_CONSOLE_CALL = re.compile(r"\bconsole\.\w+\(([^;\n]*)")
_PASSWORD_INPUT = re.compile(r"<input\b(?:(?!/>).)*?type=[\"']password[\"'](?:(?!/>).)*?/>", re.S)
_SHARED_HINT = re.compile(r"autoComplete=\{passwordAutoComplete\(")


def _storage_leaks(text: str) -> list[str]:
    return [m.group(0) for m in _STORAGE_WRITE.finditer(text)
            if ("setItem" in m.group(0) or "cookie" in m.group(0)) and _PASSWORD_WORD.search(m.group(0))]


def _console_leaks(text: str) -> list[str]:
    return [m.group(0) for m in _CONSOLE_CALL.finditer(text) if _PASSWORD_WORD.search(m.group(1))]


def _unshared_hints(text: str) -> list[str]:
    return [m.group(0) for m in _PASSWORD_INPUT.finditer(text) if not _SHARED_HINT.search(m.group(0))]


@pytest.mark.parametrize(
    "snippet,leaks",
    [
        ("localStorage.setItem('k', password)", 1),
        ("safe(() => sessionStorage.setItem(PW_KEY, form.passwordValue))", 1),
        ("document.cookie = `p=${newPassword}`", 1),
        ("localStorage.setItem(EMAIL_KEY, email)", 0),
        ("localStorage.removeItem(TOKEN_KEY)", 0),
    ],
)
def test_storage_scanner(snippet, leaks):
    assert len(_storage_leaks(snippet)) == leaks


@pytest.mark.parametrize(
    "snippet,leaks",
    [
        ("console.log('submit', email, password)", 1),
        ("console.error(newPassword)", 1),
        ("console.error('login failed', err.message)", 0),
        ("console.warn(status)", 0),
    ],
)
def test_console_scanner(snippet, leaks):
    assert len(_console_leaks(snippet)) == leaks


@pytest.mark.parametrize(
    "snippet,bad",
    [
        ('<input id="p" type="password" autoComplete="current-password" />', 1),
        ('<input id="p" type="password" />', 1),
        ('<input id="p" type="password" autoComplete={passwordAutoComplete(purpose)} />', 0),
        ('<input id="e" type="email" autoComplete="email" />', 0),
    ],
)
def test_hint_scanner(snippet, bad):
    assert len(_unshared_hints(snippet)) == bad


def _sources() -> list[Path]:
    return sorted(p for p in SRC.rglob("*") if p.suffix in {".ts", ".tsx", ".js", ".jsx"}
                  and ".test." not in p.name)


@pytest.mark.parametrize("scanner", [_storage_leaks, _console_leaks])
def test_no_password_reaches_storage_or_the_console(scanner):
    found = {str(p.relative_to(ROOT)): scanner(p.read_text(encoding="utf-8")) for p in _sources()}
    found = {k: v for k, v in found.items() if v}
    assert not found


def test_every_password_field_takes_the_shared_hint():
    fields = {str(p.relative_to(ROOT)): _PASSWORD_INPUT.findall(p.read_text(encoding="utf-8")) for p in _sources()}
    assert sum(len(v) for v in fields.values()) >= 2, "the sign-in and reset forms have password fields"
    bad = {k: _unshared_hints("\n".join(v)) for k, v in fields.items()}
    assert not {k: v for k, v in bad.items() if v}


# ── the shared rule itself ────────────────────────────────────────────────

def _rule_table() -> dict[str, str]:
    text = RULE.read_text(encoding="utf-8")
    body = re.search(r"PASSWORD_AUTOCOMPLETE\s*=\s*\{(.*?)\}", text, re.S)
    assert body, "the shared rule is one table"
    return dict(re.findall(r"['\"]?([\w-]+)['\"]?\s*:\s*['\"]([\w-]+)['\"]", body.group(1)))


@pytest.mark.parametrize(
    "purpose,hint",
    [("sign-in", "current-password"), ("new", "new-password")],
)
def test_the_rule_maps_each_purpose_to_the_standard_hint(purpose, hint):
    # The hint values are the HTML autofill tokens (WHATWG HTML, autofill
    # field names): an existing secret is current-password, a chosen one is
    # new-password.
    assert _rule_table()[purpose] == hint


@pytest.mark.parametrize(
    "form_marker,purpose_expr",
    [
        ('id="newPassword"', "'new'"),
        ('id="password"', "mode === 'signin' ? 'sign-in' : 'new'"),
    ],
)
def test_each_login_field_asks_for_the_right_purpose(form_marker, purpose_expr):
    text = LOGIN.read_text(encoding="utf-8")
    field = next(m.group(0) for m in _PASSWORD_INPUT.finditer(text) if form_marker in m.group(0))
    assert f"passwordAutoComplete({purpose_expr})" in field


# ── the sign-in form forgets the password once it is sent ────────────────

def _function_body(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    depth, i = 0, text.index("{", text.index(")", start))
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j + 1]
    raise AssertionError(name)


@pytest.mark.parametrize(
    "handler,sent_call",
    [("handleSubmit", "await login("), ("handleReset", "await apiPost('/v1/users/reset-password'")],
)
def test_the_password_is_forgotten_once_sent(handler, sent_call):
    body = _function_body(LOGIN.read_text(encoding="utf-8"), handler)
    after = body[body.index(sent_call):]
    assert "forgetPassword()" in after, f"{handler} keeps the password after it is sent"


def test_forget_password_clears_state_and_the_field():
    body = _function_body(LOGIN.read_text(encoding="utf-8"), "forgetPassword")
    assert "setPassword('')" in body
    assert re.search(r"passwordRef\.current\)?\s*\.?value\s*=\s*''|passwordRef\.current\.value = ''", body)
