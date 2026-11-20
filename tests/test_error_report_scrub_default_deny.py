"""G-10 exam: the error-report scrub is DEFAULT-DENY for the free-form parts of
an event, not a list of field names.

Grader finding on #348: free-form fields were judged by a suffix list, so a
field nobody listed (``answer``, ``rows``, ``params``, an invented name) was
carried whole; short credential spellings (``pin``, ``pw``, ``otp``,
``passcode``) were not credentials; and text up to 1024 characters was carried,
so a document passage could leave.

Every field name below is INVENTED for this test, and the product code must not
list any of them: a field is dropped because it is NOT on the short allowed
list, not because someone listed it. Values are assembled at run time from
harmless parts so the repository secret scan never sees a credential shape.
"""

from __future__ import annotations

import json

import pytest

from app.infra.monitoring import scrub_error_event


def _assembled(*parts: str) -> str:
    return "".join(parts)


def _blob(event) -> str:
    return json.dumps(event, default=str)


def _event(**extra):
    event = {
        "level": "error",
        "exception": {"values": [{"type": "ValueError", "value": "unsupported unit in row 4",
                                  "stacktrace": {"frames": [{"filename": "app/core/units.py",
                                                             "function": "convert", "lineno": 88}]}}]},
        "request": {"method": "POST", "url": "https://example.invalid/v1/x", "headers": {}},
    }
    event.update(extra)
    return event


# A sentence that reads like document text, invented for this test.
_SENTENCE = "the contractor shall reinstate the verge to the original level within fourteen days"
# ~900 characters of document-like prose: a passage that must never leave.
_PASSAGE = (" ".join([_SENTENCE] * 11))[:900]


# -- 1. free-form containers: unlisted fields are dropped -----------------------

@pytest.mark.parametrize("container,field", [
    ("extra", "zq_client_note"),
    ("extra", "wobble7"),
])
def test_an_unlisted_extra_field_does_not_leave(container, field):
    out = scrub_error_event(_event(**{container: {field: _SENTENCE}}), {})
    assert out is not None
    assert _SENTENCE not in _blob(out)


@pytest.mark.parametrize("ctx,field", [
    ("zq_turn", "recall_answer"),
    ("billing_view", "row_listing"),
])
def test_an_unlisted_context_field_does_not_leave(ctx, field):
    out = scrub_error_event(_event(contexts={ctx: {field: _SENTENCE}}), {})
    assert out is not None
    assert _SENTENCE not in _blob(out)


@pytest.mark.parametrize("field", ["zq_query_echo", "matched_rows"])
def test_unlisted_breadcrumb_data_does_not_leave(field):
    crumbs = {"values": [{"category": "query", "message": "search finished", "level": "info",
                          "data": {field: _SENTENCE}}]}
    out = scrub_error_event(_event(breadcrumbs=crumbs), {})
    assert out is not None
    assert _SENTENCE not in _blob(out)
    # the crumb itself, and its short diagnostic message, still report
    assert out["breadcrumbs"]["values"][0]["message"] == "search finished"


@pytest.mark.parametrize("field", ["zq_form_echo", "env"])
def test_unlisted_request_fields_do_not_leave(field):
    ev = _event()
    ev["request"][field] = {"zq_inner": _SENTENCE}
    out = scrub_error_event(ev, {})
    assert out is not None
    assert _SENTENCE not in _blob(out)
    assert out["request"]["method"] == "POST"


@pytest.mark.parametrize("header", ["X-Zq-Echo", "X-Wobble-Trace-Note"])
def test_an_unlisted_header_does_not_leave(header):
    ev = _event()
    ev["request"]["headers"] = {header: _SENTENCE, "Content-Type": "application/json"}
    out = scrub_error_event(ev, {})
    assert _SENTENCE not in _blob(out)
    assert out["request"]["headers"]["Content-Type"] == "application/json"


# -- 2. allowed diagnostic fields still report ----------------------------------

@pytest.mark.parametrize("ctx,fields", [
    ("runtime", {"name": "CPython", "version": "3.11.18"}),
    ("trace", {"trace_id": "a" * 32, "span_id": "b" * 16, "op": "http.server", "status": "internal_error"}),
])
def test_allowed_context_fields_still_report(ctx, fields):
    out = scrub_error_event(_event(contexts={ctx: dict(fields)}), {})
    assert out["contexts"][ctx] == fields


# -- 3. short credential spellings are credentials ------------------------------

@pytest.mark.parametrize("name,secret", [
    ("pin", _assembled("48", "21")),
    ("pw", _assembled("hun", "ter", "7q")),
    ("otp", _assembled("612", "093")),
    ("passcode", _assembled("zq", "pass", "77")),
    ("pass", _assembled("zq", "wd", "88")),
    ("sid", _assembled("zqs", "id", "991")),
])
def test_short_credential_spellings_are_masked_in_text_and_fields(name, secret):
    text = f"login retry failed {name}={secret} after 3 attempts"
    out = scrub_error_event(_event(exception={"values": [{"type": "AuthError", "value": text}]},
                                   user={name: secret}), {})
    assert secret not in _blob(out)
    assert "login retry failed" in out["exception"]["values"][0]["value"]


@pytest.mark.parametrize("word", ["spinner", "pipeline"])
def test_words_that_merely_contain_a_short_spelling_are_not_masked(word):
    text = f"{word}=ready in stage 2"
    out = scrub_error_event(_event(exception={"values": [{"type": "StateError", "value": text}]}), {})
    assert out["exception"]["values"][0]["value"] == text


# -- 4. carried text is short: a document passage never leaves ------------------

@pytest.mark.parametrize("where", ["exception", "message"])
def test_a_900_character_passage_does_not_leave(where):
    if where == "exception":
        ev = _event(exception={"values": [{"type": "ParseError", "value": _PASSAGE}]})
    else:
        ev = _event(message=_PASSAGE)
    out = scrub_error_event(ev, {})
    assert out is not None
    assert _SENTENCE not in _blob(out)


@pytest.mark.parametrize("message", [
    "row 14 has no unit; expected one of m, m2, m3, nr",
    "timeout after 30 s waiting for the extraction child of a 12-page upload",
])
def test_an_ordinary_short_error_message_still_reports(message):
    out = scrub_error_event(_event(exception={"values": [{"type": "ValueError", "value": message}]}), {})
    assert out["exception"]["values"][0]["value"] == message
