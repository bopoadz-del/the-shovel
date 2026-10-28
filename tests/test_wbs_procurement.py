"""generate_wbs injects long-lead procurement from extracted equipment lead
times and links each as a predecessor of the matching install activity — the
document_engine -> schedule connection that makes the schedule equipment-driven.
"""
from __future__ import annotations

import asyncio

from app.containers.construction import ConstructionContainer

LEAD_TIMES = [
    {"equipment": "Chiller", "lead_time_days": 150},
    {"equipment": "Switchgear", "lead_time_days": 120},
    {"equipment": "Generator", "lead_time_days": 200},
]


def _wbs(**params):
    base = {"target_count": 120, "project_type": "data_center", "start_date": "2026-01-05"}
    base.update(params)
    return asyncio.run(ConstructionContainer().generate_wbs({}, base))


# PR#3 (ORDERS_28OCT_0345Z/0415Z): generate_wbs no longer emits a template, so
# a base activity set for the non-procurement tests comes from a real BOQ
# demolition/site-clearance scope. Procurement injection itself is unchanged —
# it rides on whatever base exists (BOQ rows here; none when only lead times
# are handed over, which the procurement-only tests above still exercise).
BOQ_ASK = "Generate a high-level WBS for the demolition and site clearance bill of quantities scope."
BOQ_ITEMS = [
    {"item_key": "D110", "description": "General site clearance", "quantity": 10, "unit": "m2"},
    {"item_key": "D290.1", "description": "Removal of trees in existing sidewalks", "quantity": 20, "unit": "m"},
    {"item_key": "D549.2", "description": "Removal of existing chain link fence", "quantity": 30, "unit": "nr"},
]


def _boq_wbs(**params):
    data = {"brief": BOQ_ASK, "boq_items": BOQ_ITEMS}
    base = {"target_count": 120, "start_date": "2026-01-05"}
    base.update(params)
    return asyncio.run(ConstructionContainer().generate_wbs(data, base))


def test_procurement_injected_and_counted():
    r = _wbs(procurement_lead_times=LEAD_TIMES)
    assert r["procurement_injected"] == 3
    procs = [a for a in r["activities"] if a.get("long_lead")]
    assert {p["name"] for p in procs} == {
        "Procure Chiller (long-lead)", "Procure Switchgear (long-lead)",
        "Procure Generator (long-lead)",
    }
    dur = {p["name"]: p["duration_days"] for p in procs}
    assert dur["Procure Chiller (long-lead)"] == 150
    assert dur["Procure Generator (long-lead)"] == 200


def test_procurement_injected_onto_a_real_base_without_fabricating_install():
    # Re-pointed from the template "Chiller install" link: there is no template
    # to carry install activities, so procurement is injected onto the real BOQ
    # base and must NOT fabricate an install activity to link to. Linking
    # procurement to a matching install is a Stage-2 capability (once an LLM-
    # extracted base WBS carries install activities); here the honest Stage-1
    # behaviour is: procurement present, no invented install, assumption surfaced.
    r = _boq_wbs(procurement_lead_times=LEAD_TIMES)
    assert r["status"] == "success"
    procs = [a for a in r["activities"] if a.get("long_lead")]
    assert len(procs) == 3
    # No install activity was invented to receive the link.
    assert not any(a["name"].endswith("install") for a in r["activities"])
    assert all(p.get("linked_to_install") in (False, None) for p in procs)
    assert any("long-lead procurement" in a.lower() for a in r["assumptions"])


def test_no_feed_means_no_procurement():
    # Re-pointed to a real BOQ base (a prose brief with no feed is now no-scope):
    # with a base but no lead times, nothing procurement is injected.
    r = _boq_wbs()
    assert r["procurement_injected"] == 0
    assert not [a for a in r["activities"] if a.get("long_lead")]


def test_zero_and_blank_lead_times_ignored():
    r = _wbs(procurement_lead_times=[
        {"equipment": "Chiller", "lead_time_days": 0},      # zero -> skip
        {"equipment": "", "lead_time_days": 90},            # no name -> skip
        {"equipment": "Transformer", "lead_time_days": 95},  # valid
    ])
    procs = [a for a in r["activities"] if a.get("long_lead")]
    assert len(procs) == 1 and procs[0]["name"] == "Procure Transformer (long-lead)"


def test_target_milestones_passed_through():
    ms = [{"name": "RFS", "target_date": "2027-12-01"}]
    r = _boq_wbs(target_milestones=ms)
    assert r["target_milestones"] == ms
