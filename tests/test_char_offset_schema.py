"""char_offset must reach the model, and must not reach the block.

Saying what was lost stopped the model claiming absence. It did not let it
find the answer. Live on 733ef49, after the loss numbers landed:

    "The tool extracted text from all 16 pages, but the result was truncated
     -- only the first 7,116 of 22,189 characters were returned ... I cannot
     see whether an m3 demolition item exists later in the bill, and I
     therefore cannot provide a quantity or CESMM code."

Honest, correctly scoped, and still a dead end: the model knew exactly what
it was missing and had no way to ask for it.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _FILE_TOOL_SCHEMAS

FILE_TOOLS = ("boq_processor", "drawing_qto", "spec_analyzer",
              "primavera_parser")


@pytest.mark.parametrize("name", FILE_TOOLS)
def test_every_file_tool_declares_char_offset(name):
    """A parameter the model cannot see is a parameter it cannot use."""
    props = _FILE_TOOL_SCHEMAS[name]["properties"]
    assert "char_offset" in props
    assert props["char_offset"]["type"] == "integer"


@pytest.mark.parametrize("name", FILE_TOOLS)
def test_the_description_tells_the_model_when_to_stop(name):
    """An offset with no stopping rule is an infinite loop. The description
    names both the field to read next and the one that ends it."""
    desc = _FILE_TOOL_SCHEMAS[name]["properties"]["char_offset"]["description"]
    assert "next_char_offset" in desc
    assert "chars_remaining" in desc
    # And the rule this exists to enforce.
    assert "absent" in desc


@pytest.mark.parametrize("name", FILE_TOOLS)
def test_char_offset_is_never_required(name):
    """The first call must stay a one-argument call. Requiring an offset
    would make the model guess a number before it has one to guess from."""
    schema = _FILE_TOOL_SCHEMAS[name]
    assert schema["required"] == ["file_path"]
    assert "file_path" in schema["properties"]


def test_the_table_covers_exactly_the_file_consuming_blocks():
    assert set(_FILE_TOOL_SCHEMAS) == set(FILE_TOOLS)


# -- and it must not reach the block --------------------------------------


def test_char_offset_is_stripped_before_dispatch():
    """It selects a window of the SERIALIZED result; the block reads the whole
    file either way. Forwarding it hands every file block an argument it never
    declared -- and these are vendored third-party blocks, where an unknown
    param is discovered in production rather than at the call site."""
    from app.agents.runtime import _strip_presentation_args

    args = {"file_path": "BOQ.pdf", "char_offset": 7116}
    out = _strip_presentation_args(args)
    assert out == {"file_path": "BOQ.pdf"}
    assert args == {"file_path": "BOQ.pdf", "char_offset": 7116}, "must not mutate"


def test_stripping_leaves_an_ordinary_call_identical():
    """Same object back, so the common path costs nothing and cannot reorder
    or drop a real argument."""
    from app.agents.runtime import _strip_presentation_args

    args = {"file_path": "BOQ.pdf"}
    assert _strip_presentation_args(args) is args
    assert _strip_presentation_args("not a dict") == "not a dict"
    assert _strip_presentation_args(None) is None


def test_the_offset_still_reaches_the_presentation_layer():
    """Stripped from dispatch is not the same as thrown away: the window the
    model asked for still has to be the window it gets."""
    from app.agents.runtime import _requested_char_offset

    call = {"function": {"arguments": '{"file_path":"BOQ.pdf","char_offset":7116}'}}
    assert _requested_char_offset(call) == 7116


def test_dispatch_args_is_the_only_door():
    """The strip lives inside the parse, so a call site cannot get raw
    arguments without also getting them cleaned.

    Mutation killed: dropping the strip at the call site in _run_tool_call --
    which every test that exercises the helper alone survives.
    """
    import json as _json

    import pytest as _pytest

    from app.agents.runtime import _dispatch_args

    assert _dispatch_args('{"file_path":"BOQ.pdf","char_offset":7116}') == {
        "file_path": "BOQ.pdf"
    }
    # Already-parsed args go through the same door.
    assert _dispatch_args({"file_path": "x", "char_offset": 9}) == {"file_path": "x"}
    # And bad JSON still raises for the caller's invalid-args branch.
    with _pytest.raises(_json.JSONDecodeError):
        _dispatch_args("not json")


def test_run_tool_call_parses_through_dispatch_args():
    """Pins the call site itself: _run_tool_call must not re-implement the
    parse and skip the strip."""
    import inspect

    from app.agents.runtime import Agent

    src = inspect.getsource(Agent._run_tool_call)
    assert "_dispatch_args(raw_args)" in src
    assert "json.loads(raw_args)" not in src
