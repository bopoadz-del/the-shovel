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


def test_procurement_links_to_matching_install():
    r = _wbs(procurement_lead_times=LEAD_TIMES)
    # Chiller procurement must be a predecessor of "Chiller install".
    chiller_install = [a for a in r["activities"] if a["name"] == "Chiller install"]
    assert chiller_install, "template should contain a Chiller install activity"
    preds = chiller_install[0].get("predecessors", [])
    assert any(str(p).startswith("P.") for p in preds), "chiller install should depend on a procurement activity"
    # The assumption is surfaced.
    assert any("long-lead procurement" in a.lower() for a in r["assumptions"])


def test_no_feed_means_no_procurement():
    r = _wbs()
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
    r = _wbs(target_milestones=ms)
    assert r["target_milestones"] == ms
