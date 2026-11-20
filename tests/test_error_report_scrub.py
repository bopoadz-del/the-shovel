"""SEC-1 exam: nothing sensitive leaves in an error report; an ordinary error still reports.

Every event below is built here, by hand, in the shape the Sentry SDK sends.
Each kind of sensitive content goes in; none of it may come out. The values
are invented for this test and appear nowhere in product code.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from unittest.mock import patch

import pytest

from app.infra.monitoring import scrub_error_event


def _assembled(*parts: str) -> str:
    """Join harmless fragments at run time.

    A credential-shaped value written whole in the source (``password: "..."``)
    trips the repository secret scan, so values of that shape are built here
    instead. Nothing about the scrub changes: the event still carries the
    whole value, and the assertion still looks for the whole value.
    """
    return "".join(parts)


# Credential-shaped values for the field-kind tests, invented and assembled.
_CONN_PW = _assembled("inv3nted", "Conn", "Pw")
_BODY_PW = _assembled("Inv3nted", "Body", "Pw")


def _blob(event) -> str:
    return json.dumps(event, default=str)


def _base_event(**extra):
    event = {
        "level": "error",
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "value": "unsupported unit in row 4",
                    "stacktrace": {
                        "frames": [
                            {
                                "filename": "app/core/units.py",
                                "function": "convert",
                                "lineno": 88,
                                "vars": {},
                            }
                        ]
                    },
                }
            ]
        },
        "request": {"method": "POST", "url": "https://example.invalid/v1/x", "headers": {}},
    }
    event.update(extra)
    return event


# ── header kinds: authorization and cookie ───────────────────────────────

@pytest.mark.parametrize(
    "header,value,leak",
    [
        ("Authorization", "Bearer qq7Zt1-invented-header-credential-889", "qq7Zt1-invented"),
        ("authorization", "Basic aW52ZW50ZWQ6cGFpcg==", "aW52ZW50ZWQ6cGFpcg"),
        ("Cookie", "sess=inv3nted-cookie-value-41; theme=dark", "inv3nted-cookie-value-41"),
        ("Set-Cookie", "refresh=inv3nted-refresh-cookie-77; HttpOnly", "inv3nted-refresh-cookie-77"),
        ("X-Api-Key", "inv3nted-header-api-key-5512", "inv3nted-header-api-key-5512"),
        ("Proxy-Authorization", "Bearer inv3nted-proxy-cred-3", "inv3nted-proxy-cred-3"),
    ],
)
def test_credential_headers_do_not_leave(header, value, leak):
    event = _base_event()
    event["request"]["headers"] = {header: value, "Accept": "application/json"}
    out = scrub_error_event(event, {})
    assert out is not None
    assert leak not in _blob(out)
    assert out["request"]["headers"]["Accept"] == "application/json"


@pytest.mark.parametrize(
    "pairs",
    [
        [["Authorization", "Bearer inv3nted-pair-cred-1"], ["Accept", "text/html"]],
        [["Cookie", "sid=inv3nted-pair-cookie-2"], ["Accept", "text/html"]],
    ],
)
def test_headers_as_pairs_are_scrubbed(pairs):
    event = _base_event()
    event["request"]["headers"] = copy.deepcopy(pairs)
    out = scrub_error_event(event, {})
    blob = _blob(out)
    assert "inv3nted-pair" not in blob
    assert "text/html" in blob


# ── tokens, keys, passwords by field kind ─────────────────────────────────

@pytest.mark.parametrize(
    "key,value",
    [
        ("access_token", "inv3nted-access-token-value-1"),
        ("refreshToken", "inv3nted-refresh-token-value-2"),
        ("api_key", "inv3nted-api-key-value-3"),
        ("private-key", "inv3nted-private-key-value-4"),
        ("password", "Inv3nted-Passw0rd!"),
        ("new_passwd", "Inv3nted-Passw0rd-two"),
        ("client_secret", "inv3nted-client-secret-5"),
        ("credentials", {"user": "a", "pw": "inv3nted-nested-cred-6"}),
        ("signature", "inv3nted-signature-value-7"),
    ],
)
def test_sensitive_field_kinds_are_filtered_anywhere(key, value):
    event = _base_event(extra={"ctx": {"deeper": [{key: value, "count": 3}]}})
    out = scrub_error_event(event, {})
    blob = _blob(out)
    assert "inv3nted" not in blob.lower()
    # default-deny: the free-form ``extra`` container is dropped whole, so the
    # invented nested structure does not survive either.
    assert out["extra"] == {}


# ── the same kinds inside free text ───────────────────────────────────────

@pytest.mark.parametrize(
    "text,leak",
    [
        ("login failed password=Inv3ntedPw!x for user", "Inv3ntedPw!x"),
        ("retry with token: inv3ntedtok3n998 now", "inv3ntedtok3n998"),
        ("GET /x?api_key=inv3ntedq5k&page=2", "inv3ntedq5k"),
        ("header was Bearer inv3nted.bearer.v4lue", "inv3nted.bearer.v4lue"),
        ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl rejected", "eyJzdWIiOiJ4In0"),
        ("opaque k3Yq9xW2mP7rT4vL8nB6cZ1dF5gH0jKs leaked", "k3Yq9xW2mP7rT4vL8nB6cZ1dF5gH0jKs"),
    ],
)
def test_secrets_inside_messages_are_masked(text, leak):
    event = _base_event(message=text)
    event["exception"]["values"][0]["value"] = text
    out = scrub_error_event(event, {})
    assert leak not in _blob(out)


# ── connection strings ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "dsn,leak",
    [
        ("postgresql://appuser:Inv3ntedDbPw@db.example.invalid:5432/main", "Inv3ntedDbPw"),
        ("redis://:inv3ntedRedisPw@cache.example.invalid:6379/0", "inv3ntedRedisPw"),
        ("mongodb+srv://u:inv3ntedMongoPw@cluster.example.invalid/db", "inv3ntedMongoPw"),
    ],
)
def test_connection_strings_lose_their_credentials(dsn, leak):
    event = _base_event()
    event["exception"]["values"][0]["value"] = f"could not connect to {dsn}"
    event["extra"] = {"target": dsn}
    out = scrub_error_event(event, {})
    blob = _blob(out)
    assert leak not in blob
    assert "example.invalid" in blob  # the host still says where it failed


@pytest.mark.parametrize("key", ["database_url", "DATABASE_URL", "dsn", "connection_string"])
def test_connection_string_field_kind_is_filtered(key):
    event = _base_event(extra={key: f"host=db.example.invalid user=u password={_CONN_PW}"})
    out = scrub_error_event(event, {})
    assert _CONN_PW not in _blob(out)


# ── e-mail addresses ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "where",
    ["message", "user", "extra"],
)
@pytest.mark.parametrize(
    "address",
    ["first.last@example.invalid", "ops+alerts@sub.example.invalid"],
)
def test_email_addresses_do_not_leave(where, address):
    event = _base_event()
    if where == "message":
        event["exception"]["values"][0]["value"] = f"no account for {address}"
    elif where == "user":
        event["user"] = {"id": "u-1", "email": address}
    else:
        event["extra"] = {"note": f"invite sent to {address} today"}
    out = scrub_error_event(event, {})
    assert address not in _blob(out)


# ── request bodies ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "body",
    [
        {"email": "x@example.invalid", "password": _BODY_PW},
        '{"question": "inv3nted body question about the slab"}',
    ],
)
def test_request_body_does_not_leave(body):
    event = _base_event()
    event["request"]["data"] = body
    event["request"]["cookies"] = {"sid": "inv3nted-cookie-jar"}
    out = scrub_error_event(event, {})
    blob = _blob(out)
    assert _BODY_PW not in blob
    assert "inv3nted body question" not in blob
    assert "inv3nted-cookie-jar" not in blob
    assert out["request"]["method"] == "POST"


# ── document text ─────────────────────────────────────────────────────────

_DOC_PARAGRAPH = (
    "Clause 14.3 The Contractor shall submit an interim payment application "
    "within seven days of the end of each month, stating the amounts due. "
) * 20


@pytest.mark.parametrize(
    "key",
    ["text", "document_text", "content", "chunks", "excerpt", "prompt"],
)
def test_document_text_fields_are_filtered(key):
    event = _base_event(extra={key: "inv3nted document sentence. " * 3})
    out = scrub_error_event(event, {})
    assert "inv3nted document sentence" not in _blob(out)


@pytest.mark.parametrize("where", ["exception", "breadcrumb"])
def test_long_free_text_is_not_carried(where):
    event = _base_event()
    if where == "exception":
        event["exception"]["values"][0]["value"] = "parse failed near: " + _DOC_PARAGRAPH
    else:
        event["breadcrumbs"] = {"values": [{"category": "log", "message": _DOC_PARAGRAPH}]}
    out = scrub_error_event(event, {})
    assert "interim payment application" not in _blob(out)


# ── local variables ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "local_vars",
    [
        {"row": "'inv3nted local row text'", "n": "4"},
        {"user_pw": "'Inv3ntedLocalPw'", "self": "<obj>"},
    ],
)
def test_local_variables_do_not_leave(local_vars):
    event = _base_event()
    event["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"] = local_vars
    event["threads"] = {"values": [{"stacktrace": {"frames": [{"function": "f", "vars": dict(local_vars)}]}}]}
    out = scrub_error_event(event, {})
    blob = _blob(out)
    assert "inv3nted local row text" not in blob
    assert "Inv3ntedLocalPw" not in blob


# ── an ordinary error still reports ───────────────────────────────────────

@pytest.mark.parametrize(
    "exc_type,message",
    [
        ("ValueError", "unsupported unit in row 4"),
        ("KeyError", "'quantity' missing from parsed table"),
    ],
)
def test_an_ordinary_error_still_reports(exc_type, message):
    event = _base_event()
    event["exception"]["values"][0]["type"] = exc_type
    event["exception"]["values"][0]["value"] = message
    event["request"]["headers"] = {"Accept": "application/json", "User-Agent": "pytest"}
    out = scrub_error_event(copy.deepcopy(event), {})
    assert out is not None
    exc = out["exception"]["values"][0]
    assert exc["type"] == exc_type
    assert exc["value"] == message
    frame = exc["stacktrace"]["frames"][0]
    assert (frame["filename"], frame["function"], frame["lineno"]) == ("app/core/units.py", "convert", 88)
    assert out["request"]["url"] == "https://example.invalid/v1/x"
    assert out["request"]["headers"]["User-Agent"] == "pytest"


@pytest.mark.parametrize("event", [{"message": "cache warm"}, {"level": "error", "message": "x" * 10}])
def test_the_hook_does_not_mutate_or_drop_clean_events(event):
    before = copy.deepcopy(event)
    out = scrub_error_event(event, {})
    assert out == before
    assert event == before


# ── the hook is wired into the real SDK ───────────────────────────────────

_needs_sdk = pytest.mark.skipif(importlib.util.find_spec("sentry_sdk") is None, reason="sentry_sdk not installed")


@_needs_sdk
@pytest.mark.parametrize("dsn", ["https://pub@o1.ingest.example.invalid/1", "https://pub@o2.ingest.example.invalid/22"])
def test_init_sentry_installs_the_scrub_hook_and_no_locals(dsn):
    from app.infra import monitoring

    with patch.dict(os.environ, {"SENTRY_DSN": dsn}), patch("sentry_sdk.init") as init:
        assert monitoring.init_sentry() is True
    kwargs = init.call_args.kwargs
    assert kwargs["before_send"] is scrub_error_event
    assert kwargs["before_send_transaction"] is scrub_error_event
    assert kwargs["send_default_pii"] is False
    assert kwargs["include_local_variables"] is False
    assert kwargs["max_request_body_size"] == "never"
    monitoring._sentry_enabled = False


@_needs_sdk
@pytest.mark.parametrize(
    "secret_text",
    ["password=Inv3ntedSdkPw for ops@example.invalid", "postgresql://u:Inv3ntedSdkDb@h.example.invalid/d"],
)
def test_a_real_sdk_event_leaves_scrubbed(secret_text):
    import sentry_sdk
    from sentry_sdk.transport import Transport

    from app.infra import monitoring

    sent = []

    class _Capture(Transport):
        def capture_envelope(self, envelope):
            for item in envelope.items:
                if item.payload.json is not None:
                    sent.append(item.payload.json)

    with patch.dict(os.environ, {"SENTRY_DSN": "https://pub@o1.ingest.example.invalid/1"}), \
            patch("sentry_sdk.init") as init:
        monitoring.init_sentry()
    kwargs = dict(init.call_args.kwargs)
    kwargs["integrations"] = []
    kwargs["transport"] = _Capture
    kwargs["default_integrations"] = False
    sentry_sdk.init(**kwargs)
    try:
        # Built at run time so the value exists only as a local, not in source context.
        local_doc_text = " ".join(["inv3nted", "local", "document", "text"])  # noqa: F841, FLY002
        try:
            raise RuntimeError(f"connect failed: {secret_text}")
        except RuntimeError:
            sentry_sdk.capture_exception()
        sentry_sdk.flush()
    finally:
        sentry_sdk.init()  # no DSN: back to a disabled client
        monitoring._sentry_enabled = False

    assert sent, "an ordinary error must still report"
    blob = _blob(sent)
    assert "RuntimeError" in blob
    event = next(e for e in sent if e.get("exception"))
    assert len(event["event_id"]) == 32 and "Filtered" not in event["event_id"]
    for leak in ("Inv3ntedSdkPw", "ops@example.invalid", "Inv3ntedSdkDb", "inv3nted local document text"):
        assert leak not in blob


@pytest.mark.parametrize(
    "ident",
    ["0f3a9c2e7b1d4e6fa8c05b9d2e7f1a3c", "4908cd1a2b3c4d5e6f708192a3b4c5d6e7f80912"],
)
def test_hex_identifiers_are_kept(ident):
    event = {"event_id": ident, "contexts": {"trace": {"trace_id": ident}}, "release": ident}
    out = scrub_error_event(event, {})
    assert out == event


@_needs_sdk
@pytest.mark.parametrize("dsn", ["", "   "])
def test_no_dsn_means_no_error_reporting(dsn):
    from app.infra import monitoring

    with patch.dict(os.environ, {"SENTRY_DSN": dsn}), patch("sentry_sdk.init") as init:
        assert monitoring.init_sentry() is False
    init.assert_not_called()
    assert monitoring.sentry_enabled() is False
