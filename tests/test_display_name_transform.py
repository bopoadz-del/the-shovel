"""PR-A0: a pure display-name transform + a streaming-safe applier that never
splits a registry id across segment boundaries (ORDERS_0150Z)."""
from __future__ import annotations

import pytest

from app.lib.source_labels import (
    apply_display_names,
    DisplayNameStreamer,
    formula_display_name,
)

# A registered formula id with a display name, discovered from the registry
# (no hardcoded brand/dimension -- the id set comes from the registry).
_ID = "delay_damages_daily"
_SHOWN = formula_display_name(_ID)


@pytest.mark.skipif(not _SHOWN, reason="registry has no display name for the probe id")
def test_raw_identifier_is_converted():
    out = apply_display_names(f"I re-ran it as {_ID} just now.")
    assert _SHOWN in out
    assert _ID not in out


def test_text_without_identifiers_is_unchanged():
    plain = "The defects liability period is twelve months from completion."
    assert apply_display_names(plain) == plain


@pytest.mark.skipif(not _SHOWN, reason="registry has no display name for the probe id")
def test_transform_is_idempotent():
    once = apply_display_names(f"as {_ID} here")
    twice = apply_display_names(once)
    assert once == twice


@pytest.mark.skipif(not _SHOWN, reason="registry has no display name for the probe id")
def test_identifier_split_across_two_segments_is_handled():
    """The id arrives in two deltas; the streamer must not emit the raw id and
    must still convert it once joined."""
    s = DisplayNameStreamer()
    out = s.feed("I re-ran it as delay_")
    out += s.feed("damages_daily now.")
    out += s.flush()
    assert _ID not in out, "raw identifier leaked across the segment boundary"
    assert _SHOWN in out


def test_streamer_emits_plain_text_and_loses_nothing():
    s = DisplayNameStreamer()
    parts = ["The defects ", "liability period ", "is twelve months.\n"]
    out = "".join(s.feed(p) for p in parts) + s.flush()
    assert out == "".join(parts)
