"""Direct tests for the G-10 error-report scrub helpers.

``scrub_error_event`` dispatches the free-form containers to a small set of
default-deny helpers through ``_CONTAINER_HANDLERS``, so an end-to-end test
naming only ``scrub_error_event`` never references ``_scrub_extra`` & friends
by name.  The CI mutation gate (CI-M) selects the tests that NAME the changed
function (or an in-module caller), so those helpers were reported UNCOVERED:
no test could kill their mutants.  This file calls each helper by name to
close that gap.

For each helper there are three behaviours under test:

* an ALLOWED diagnostic field still reports (kills ``constant_return``);
* an unlisted field is dropped whole (default-deny);
* a non-container input is scrubbed through the fallback path rather than
  crashing -- which is exactly what a ``drop_rule`` mutant removing the
  ``isinstance(..., dict)`` guard would make impossible.

Every field name and value below is invented for this test; the product code
must not list any of them.  Values are assembled at run time where a literal
would look like a credential to the repository secret scan.
"""

from __future__ import annotations

import json

import pytest

from app.infra.monitoring import (
    _is_credential_field,
    _scrub_breadcrumb,
    _scrub_breadcrumbs,
    _scrub_contexts,
    _scrub_extra,
    _scrub_headers,
    _scrub_request,
    _scrub_user,
)


def _assembled(*parts: str) -> str:
    return "".join(parts)


def _blob(value) -> str:
    return json.dumps(value, default=str)


# -- _is_credential_field ---------------------------------------------------

@pytest.mark.parametrize("name", ["password", "access_token", "api_key", "pin", "otp"])
def test_is_credential_field_recognises_credentials(name):
    assert _is_credential_field(name) is True


@pytest.mark.parametrize("name", ["spinner", "pipeline", "category", "method", "message"])
def test_is_credential_field_rejects_ordinary_words(name):
    assert _is_credential_field(name) is False


# -- _scrub_extra -----------------------------------------------------------

@pytest.mark.parametrize("value", [
    {"zq_client_note": "the slab was measured twice"},
    {"rows": [{"qty": 4}]},
])
def test_scrub_extra_drops_the_whole_container(value):
    assert _scrub_extra(value) == {}


@pytest.mark.parametrize("value", ["plain diagnostic", "another plain string"])
def test_scrub_extra_non_dict_is_scrubbed(value):
    assert _scrub_extra(value) == value


# -- _scrub_contexts --------------------------------------------------------

@pytest.mark.parametrize("ctx,fields", [
    ("runtime", {"name": "CPython", "version": "3.11.18"}),
    ("trace", {"trace_id": "a" * 32, "span_id": "b" * 16, "op": "http.server"}),
])
def test_scrub_contexts_keeps_allowed_context_fields(ctx, fields):
    assert _scrub_contexts({ctx: dict(fields)}) == {ctx: fields}


@pytest.mark.parametrize("ctx", ["zq_turn", "billing_view"])
def test_scrub_contexts_drops_unknown_contexts(ctx):
    assert _scrub_contexts({ctx: {"note": "free form"}}) == {}


@pytest.mark.parametrize("field", ["recall_answer", "row_listing"])
def test_scrub_contexts_drops_unknown_fields_inside_allowed_context(field):
    out = _scrub_contexts({"runtime": {field: "free form", "name": "CPython"}})
    assert out == {"runtime": {"name": "CPython"}}


@pytest.mark.parametrize("value", ["plain diagnostic", "another plain string"])
def test_scrub_contexts_non_dict_is_scrubbed(value):
    assert _scrub_contexts(value) == value


# -- _scrub_headers ---------------------------------------------------------

@pytest.mark.parametrize("header", ["X-Zq-Echo", "X-Wobble-Note"])
def test_scrub_headers_masks_unknown_header_values(header):
    out = _scrub_headers({header: "free-form value", "Content-Type": "application/json"})
    assert out[header] == "[Filtered]"
    assert out["Content-Type"] == "application/json"


@pytest.mark.parametrize("header,value", [
    ("Accept", "application/json"),
    ("User-Agent", "pytest"),
])
def test_scrub_headers_keeps_allowed_header_values(header, value):
    assert _scrub_headers({header: value}) == {header: value}


@pytest.mark.parametrize("pairs", [
    [["X-Zq-Echo", "free-form value"], ["Accept", "text/html"]],
    [["X-Wobble-Note", "free-form value"], ["Accept", "text/html"]],
])
def test_scrub_headers_masks_unknown_header_values_in_pairs(pairs):
    out = _scrub_headers(pairs)
    blob = _blob(out)
    assert "free-form value" not in blob
    assert "text/html" in blob


@pytest.mark.parametrize("value", ["plain diagnostic", "another plain string"])
def test_scrub_headers_non_dict_is_scrubbed(value):
    assert _scrub_headers(value) == value


# -- _scrub_request ---------------------------------------------------------

@pytest.mark.parametrize("payload,payload_key", [
    ({"method": "POST", "url": "https://example.invalid/v1/a", "data": {"q": "free-form value"}}, "data"),
    ({"method": "GET", "url": "https://example.invalid/v1/b", "query_string": "q=free-form value"}, "query_string"),
])
def test_scrub_request_keeps_diagnostic_drops_payload(payload, payload_key):
    out = _scrub_request(payload)
    assert out["method"] == payload["method"]
    assert out["url"] == payload["url"]
    assert payload_key not in out
    assert "free-form value" not in _blob(out)


@pytest.mark.parametrize("value", ["plain diagnostic", "another plain string"])
def test_scrub_request_non_dict_is_scrubbed(value):
    assert _scrub_request(value) == value


# -- _scrub_breadcrumb ------------------------------------------------------

@pytest.mark.parametrize("crumb", [
    {"category": "query", "message": "search finished", "level": "info", "data": {"zq_echo": "free-form value"}},
    {"type": "http", "message": "request sent", "timestamp": "12:00:00Z", "data": {"wobble": "free-form value"}},
])
def test_scrub_breadcrumb_keeps_diagnostic_drops_data(crumb):
    out = _scrub_breadcrumb(crumb)
    assert out["message"] == crumb["message"]
    assert "data" not in out
    assert "free-form value" not in _blob(out)


# -- _scrub_breadcrumbs -----------------------------------------------------

@pytest.mark.parametrize("value", [
    {"values": [{"category": "query", "message": "search finished", "data": {"zq_echo": "free-form value"}}]},
    {"values": [{"type": "http", "message": "request sent", "data": {"wobble": "free-form value"}}]},
])
def test_scrub_breadcrumbs_runs_each_crumb(value):
    out = _scrub_breadcrumbs(value)
    assert out["values"][0]["message"] == value["values"][0]["message"]
    assert "free-form value" not in _blob(out)


@pytest.mark.parametrize("value", ["plain diagnostic", "another plain string"])
def test_scrub_breadcrumbs_non_dict_is_scrubbed(value):
    assert _scrub_breadcrumbs(value) == value


# -- _scrub_user ------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("id", "u-1"),
    ("username", "alice"),
    ("ip_address", "10.0.0.1"),
])
def test_scrub_user_keeps_allowed_fields(field, value):
    assert _scrub_user({field: value}) == {field: value}


@pytest.mark.parametrize("name,secret", [
    ("password", _assembled("invented", "pw", "11")),
    ("api_key", _assembled("invented", "key", "22")),
])
def test_scrub_user_masks_credential_fields(name, secret):
    out = _scrub_user({name: secret, "id": "u-1"})
    assert out[name] == "[Filtered]"
    assert secret not in _blob(out)


@pytest.mark.parametrize("field", ["zq_note", "preferences"])
def test_scrub_user_drops_unknown_fields(field):
    out = _scrub_user({field: "free-form value", "id": "u-1"})
    assert out == {"id": "u-1"}


@pytest.mark.parametrize("value", ["plain diagnostic", "another plain string"])
def test_scrub_user_non_dict_is_scrubbed(value):
    assert _scrub_user(value) == value
