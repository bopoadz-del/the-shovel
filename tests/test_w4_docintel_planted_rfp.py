"""W4 doc-intel — planted RFP through the document reasoner.

New test shape (2026-07-26): a synthetic brief with KNOWN facts (equipment
lead times, schedule targets, constraints). The reasoner must extract the
planted values verbatim, and anything it fills from the ontology must carry
source=ontology_default — extracted facts and defaults must never blur.
"""
import types

from app.document_engine.reasoner import DocumentReasoner

CONFIG = {
    "patterns": {
        "equipment_lead_time": [
            r"(?P<equipment>generator|chiller|transformer)s?\D{0,40}?lead\s*time\D{0,20}?(?P<days>\d{2,3})\s*days",
        ],
    },
    "ontology": {
        "crane": {"type": "equipment", "lead_time": 45},
        "generator": {"type": "equipment", "lead_time": 120},
    },
}

RFP_TEXT = (
    "Section 3 — Plant. The standby generators shall have a lead time of 180 days "
    "from PO. Chillers carry a lead time of 90 days.\n"
)


def _doc(text):
    return types.SimpleNamespace(text=text, tables=[], figures=[], glossary={})


def test_planted_lead_times_extracted_verbatim():
    out = DocumentReasoner(CONFIG).reason([_doc(RFP_TEXT)])
    by_name = {s["equipment"].lower(): s for s in out.equipment_specs}
    assert by_name["generator"]["lead_time_days"] == 180
    assert by_name["chiller"]["lead_time_days"] == 90
    # Extracted facts carry the source CONTEXT, never the default tag.
    assert "source" not in by_name["generator"]
    assert "lead time" in by_name["generator"]["context"].lower()


def test_ontology_defaults_are_tagged_and_never_override_extractions():
    out = DocumentReasoner(CONFIG).reason([_doc(RFP_TEXT)])
    by_name = {s["equipment"].lower(): s for s in out.equipment_specs}
    # Crane wasn't in the text -> filled from ontology, explicitly tagged.
    assert by_name["crane"]["source"] == "ontology_default"
    assert by_name["crane"]["lead_time_days"] == 45
    # Generator WAS in the text -> the 120-day ontology default must NOT
    # replace or duplicate the extracted 180.
    gens = [s for s in out.equipment_specs if s["equipment"].lower() == "generator"]
    assert len(gens) == 1
    assert gens[0]["lead_time_days"] == 180


def test_empty_document_yields_only_tagged_defaults():
    out = DocumentReasoner(CONFIG).reason([_doc("General notes. Nothing specific.")])
    assert out.equipment_specs, "ontology defaults should still be offered"
    assert all(s.get("source") == "ontology_default" for s in out.equipment_specs)


def test_table_rows_feed_extraction():
    doc = types.SimpleNamespace(
        text="",
        tables=[[["Equipment", "Note"],
                 ["Transformer", "lead time of 150 days ex-works"]]],
        figures=[], glossary={},
    )
    out = DocumentReasoner(CONFIG).reason([doc])
    by_name = {s["equipment"].lower(): s for s in out.equipment_specs}
    assert by_name["transformer"]["lead_time_days"] == 150
