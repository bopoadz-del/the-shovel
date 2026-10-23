"""Per-block ``text_output_field`` override for chain coercion.

The orchestrator's ``_coerce_dict_to_text`` historically relied on a
global priority-ordered list (``_TEXT_OUTPUT_FIELDS``) to pull the
canonical text out of a JSON-producing block's dict before passing it
to a text-expecting block. That works when the producing block uses one
of the well-known keys (``text`` / ``translated`` / ``response`` / ...).

The contract added by this change: blocks whose canonical text lives
under a non-standard key can declare ``text_output_field = "<key>"``
as a class attribute. The orchestrator checks the producing block's
override BEFORE falling back to the global list, so:

  - a future block can ship a dict like ``{"my_weird_field": "hi"}``
    without ``my_weird_field`` ever needing to land in the global tuple
  - existing blocks (translate, chat) get explicit contracts where
    before the chain coincidentally worked because their keys were in
    the global tuple already

This module exercises the new path without touching the existing
chain-coercion tests (``tests/test_chain_json_text_coercion.py``); the
global-fallback path must remain unchanged.
"""

from __future__ import annotations

import pytest

from app.blocks.orchestrator import _coerce_dict_to_text


class _BlockWithCustomField:
    """Stand-in for a block whose canonical text lives under a key NOT in
    the orchestrator's global _TEXT_OUTPUT_FIELDS list. Used to prove the
    per-block override actually wins over the global scan."""

    name = "weird"
    text_output_field = "haiku"  # not in the global tuple


class _BlockWithStandardField:
    """Stand-in for a well-behaved block declaring its canonical key
    explicitly, even though the key is already in the global tuple."""

    name = "translate"
    text_output_field = "translated"


class _BlockNoField:
    """Legacy block — no override. Must fall through to the global tuple."""

    name = "legacy"


# ── per-block override path ──────────────────────────────────────────────────


def test_override_picks_blocks_declared_field_even_when_not_in_global_list():
    """The whole point of the override: a key NOT in _TEXT_OUTPUT_FIELDS
    still wins when the producing block declares it."""
    block = _BlockWithCustomField()
    data = {
        "haiku": "an old silent pond",
        # Decoys that the global tuple WOULD have grabbed first:
        "text": "wrong field",
        "response": "also wrong",
    }
    out = _coerce_dict_to_text(data, source_block=block)
    assert out == "an old silent pond"


def test_override_wins_over_global_tuple_position():
    """Declared field is consulted FIRST — global priority order doesn't
    overrule it even when the global tuple's earlier entries are present."""
    block = _BlockWithStandardField()
    data = {
        "text": "global-list earlier match",     # first in global tuple
        "translated": "what translate declared",  # second in global tuple
    }
    out = _coerce_dict_to_text(data, source_block=block)
    assert out == "what translate declared"


def test_override_falls_back_when_declared_field_missing():
    """If the declared key isn't present on this particular call, the
    global tuple still rescues us — backward-compatible with blocks that
    return slightly different shapes across operations."""
    block = _BlockWithCustomField()  # declares "haiku" but data doesn't have it
    data = {"text": "fallback works"}
    out = _coerce_dict_to_text(data, source_block=block)
    assert out == "fallback works"


# ── legacy / global-list path is unaffected ──────────────────────────────────


def test_no_block_passed_uses_global_list():
    """Calls that pass no source_block (e.g. callers outside the
    orchestrator's main loop) still get the priority-ordered fallback."""
    data = {"answer": "hi there"}
    out = _coerce_dict_to_text(data)
    assert out == "hi there"


def test_block_without_override_uses_global_list():
    """A block that doesn't declare text_output_field behaves as before."""
    block = _BlockNoField()
    data = {"translated": "fall-through to global"}
    out = _coerce_dict_to_text(data, source_block=block)
    assert out == "fall-through to global"


def test_block_with_none_override_uses_global_list():
    """Defensive: text_output_field=None (the class default on
    UniversalBlock) must NOT match a "None" string in the dict."""
    block = _BlockNoField()
    block.text_output_field = None  # explicit
    data = {"response": "still uses global"}
    out = _coerce_dict_to_text(data, source_block=block)
    assert out == "still uses global"


def test_single_string_heuristic_still_works():
    """The third-tier "exactly one string value" fallback is preserved."""
    data = {"weird_key": "only string here", "count": 42, "ratio": 0.5}
    out = _coerce_dict_to_text(data)
    assert out == "only string here"


def test_returns_none_when_no_text_anywhere():
    """Nothing matches → None, so the caller can fall back to its own
    type handling rather than guess wrongly."""
    data = {"count": 42, "ratio": 0.5}
    out = _coerce_dict_to_text(data)
    assert out is None


# ── real-block declarations land (smoke check) ──────────────────────────────


def test_translate_block_has_translated_declared():
    """Lock in TranslateBlock's declared output contract."""
    from app.blocks.translate import TranslateBlock
    assert TranslateBlock.text_output_field == "translated"


def test_chat_block_has_text_declared():
    """Lock in ChatBlock's declared output contract."""
    from app.blocks.chat import ChatBlock
    assert ChatBlock.text_output_field == "text"


def test_universal_block_default_is_none():
    """Blocks that don't override inherit None — the orchestrator falls
    back to the global tuple, preserving prior behaviour."""
    from app.core.universal_base import UniversalBlock
    assert UniversalBlock.text_output_field is None
