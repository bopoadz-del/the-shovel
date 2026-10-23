"""Cover core helpers the suite only ever grazed.

The 2026-08-15 coverage sweep found 77 functions in core/routers/lib running
under 34% of their body. These two are the kind that fail quietly: they sit on
the boundary between blocks, so when they get it wrong the next block receives
a shape it did not expect and the error surfaces somewhere else entirely.

`InputAdapter._empty_for_schema` builds the minimal valid input when a caller
sends None. If it returns the wrong SHAPE, the receiving block raises a
KeyError that looks like the block's bug, not the adapter's.

`DataTransformer._construction_to_text` converts a construction analysis into
the text the chat block reads. If it drops a section, the model simply never
learns that measurements or costs existed -- an omission that reads as the
model being unhelpful rather than as data loss.
"""
from __future__ import annotations

from typing import ClassVar

from app.core.data_transformer import DataTransformer
from app.core.input_adapter import InputAdapter

# ── InputAdapter._empty_for_schema ────────────────────────────────────────

class _Block:
    """Minimal stand-in: the adapter only reads these attributes."""

    def __init__(self, input_schema=None, accepted_input_types=None):
        if input_schema is not None:
            self.input_schema = input_schema
        if accepted_input_types is not None:
            self.accepted_input_types = accepted_input_types


def test_no_schema_yields_an_empty_dict_not_none():
    """Downstream code does `input.get(...)`; None would raise instead."""
    assert InputAdapter._empty_for_schema(_Block()) == {}


def test_text_accepting_block_gets_the_text_envelope():
    block = _Block(input_schema={"type": "object"},
                   accepted_input_types=["TextContent"])
    out = InputAdapter._empty_for_schema(block)
    assert out == {"text": "", "source": "", "metadata": {}}


def test_json_schema_fields_get_type_correct_empties():
    """Each JSON-Schema type must map to its own zero value.

    A string defaulted to 0, or an array defaulted to "", is exactly the kind
    of wrong-shape that blows up one block downstream of here.
    """
    block = _Block(input_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "count": {"type": "number"},
            "enabled": {"type": "boolean"},
            "items": {"type": "array"},
            "meta": {"type": "object"},
        },
    })
    out = InputAdapter._empty_for_schema(block)
    assert out == {
        "title": "", "count": 0, "enabled": False, "items": [], "meta": {},
    }
    # types, not just values -- False is not 0 and [] is not {}
    assert isinstance(out["enabled"], bool)
    assert isinstance(out["items"], list)
    assert isinstance(out["meta"], dict)


def test_unknown_field_type_falls_back_to_string():
    """An unrecognised JSON-Schema type must not drop the field entirely."""
    block = _Block(input_schema={
        "type": "object",
        "properties": {"weird": {"type": "null"}, "name": {"type": "string"}},
    })
    out = InputAdapter._empty_for_schema(block)
    assert "name" in out
    # 'weird' may be absent or defaulted, but the KNOWN field must survive
    assert out["name"] == ""


def test_ui_schema_is_used_when_there_is_no_input_schema():
    class _UiBlock:
        ui_schema: ClassVar[dict] = {
            "input": {"type": "object",
                      "properties": {"q": {"type": "string"}}}
        }

    assert InputAdapter._empty_for_schema(_UiBlock()) == {"q": ""}


# ── DataTransformer._construction_to_text ─────────────────────────────────

def _text(payload):
    out = DataTransformer()._construction_to_text(payload)
    # the transformer returns a TextContent-shaped dict
    return out.get("text", "") if isinstance(out, dict) else str(out)


def test_every_section_of_a_construction_analysis_reaches_the_text():
    """Each section the model needs must actually appear. A silently dropped
    section reads as the model being unhelpful, not as data loss."""
    body = _text({
        "summary": "Ground floor slab",
        "measurements": [{"type": "length", "value": 12.5, "unit": "m"}],
        "quantities": {"concrete_m3": 945},
        "specifications": ["C40/50"],
        "detected_disciplines": ["structural", "civil"],
        "cost_estimate": {"total_with_overhead": 412000},
        "carbon_estimate": {"total_embodied_carbon_kg": 88000},
    })
    assert "Ground floor slab" in body
    assert "12.5" in body and "m" in body
    assert "945" in body
    assert "structural" in body and "civil" in body
    assert "412000" in body
    assert "88000" in body


def test_a_universalblock_wrapped_result_is_unwrapped():
    """Blocks hand back {"result": {...}}; without unwrapping, every section
    below would be missed and the text would come back empty."""
    body = _text({"result": {"summary": "Wrapped payload",
                             "detected_disciplines": ["mep"]}})
    assert "Wrapped payload" in body
    assert "mep" in body


def test_measurement_list_is_capped_but_reports_the_true_total():
    """Only 10 measurements are itemised. The COUNT must still be honest, or
    the model concludes there were only ten."""
    body = _text({"measurements": [
        {"type": "length", "value": i, "unit": "m"} for i in range(25)
    ]})
    assert "25" in body, "the real measurement count must survive the cap"


def test_raw_text_is_the_last_resort_only():
    """raw_text is used ONLY when nothing else produced content, so it must
    not shadow a real analysis."""
    fallback = _text({"raw_text": "unparsed drawing notes"})
    assert "unparsed drawing notes" in fallback

    with_summary = _text({"summary": "Real analysis", "raw_text": "IGNORE ME"})
    assert "Real analysis" in with_summary
    assert "IGNORE ME" not in with_summary


def test_empty_analysis_does_not_raise():
    assert isinstance(_text({}), str)
