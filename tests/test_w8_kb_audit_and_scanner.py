"""W8 un-park — KB coverage audit + STANDARDS_SCANNER_BROADENED safe-on.

New test shapes (2026-07-25):
- The KB audit is a structural sweep over docs/knowledge/*.md: domain
  coverage buckets, non-empty titled notes, and — the sharp edge — NO
  project/client identifiers in the GENERAL layer (scrub policy). This
  audit found and scrubbed live leaks on its first run. That check now lives
  in scripts/scan_secrets.py, env-fed and fail-closed, because a test that
  names the identifiers inline is itself the leak.
- The scanner check is a PLANTED-DEVIATION probe: texts written to trip
  each broadened rule must be flagged, off-topic text must not be, and
  in every case the scan is advisory — it never mutates or blocks the
  deliverable.
"""
import glob
import io
import os
import re

import pytest

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "knowledge")


def _notes():
    return sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md")))


# ── KB structural audit ─────────────────────────────────────────────────────

def test_kb_has_notes_and_all_are_titled_nonempty():
    notes = _notes()
    assert len(notes) >= 20, f"expected the seeded KB, found {len(notes)} notes"
    for path in notes:
        body = io.open(path, encoding="utf-8").read().strip()
        assert len(body) > 200, f"{os.path.basename(path)} is a stub"
        assert body.startswith("#") or "\n#" in body[:400], (
            f"{os.path.basename(path)} has no heading"
        )


# General-layer knowledge must carry NO project/client identifiers. That gate
# is scripts/scan_secrets.py over every tracked file, fed from the repository
# secret; it is not re-implemented here with the names written out.


# Domain buckets the pilot depends on — each must have at least one note.
_DOMAIN_BUCKETS = {
    "safety": ("osha",),
    "contracts": ("fidic",),
    "qs_measurement": ("cesmm", "boq"),
    "rates": ("rates",),
    "project_management": ("wbdg_project_management", "evm"),
    "bim": ("wbdg_bim",),
    "ksa_authorities": ("ksa_",),
    "procedures": ("_procedure",),
    "commissioning_om": ("commissioning", "operations_maintenance"),
}


def test_kb_domain_coverage():
    names = [os.path.basename(p).lower() for p in _notes()]
    missing = {
        bucket: keys
        for bucket, keys in _DOMAIN_BUCKETS.items()
        if not any(any(k in n for n in names) for k in keys)
    }
    assert not missing, f"KB domains with no notes: {missing}"


# ── STANDARDS_SCANNER_BROADENED safe-on ─────────────────────────────────────

PLANTED = {
    "unverified_material_substitution": (
        "Contractor proposes an approved equal for the specified HDPE pipe."
    ),
    "undocumented_welding_procedure": (
        "Field welds shall be completed prior to hydrotest by qualified crews."
    ),
    "missing_commissioning_signoff": (
        "System handover and taking over of the pump station next week."
    ),
    "missing_itp_reference": (
        "Works subject to inspection test plan hold points at each stage."
    ),
    "deviated_test_frequency": (
        "Compaction sample rate is 1 per 500 m2 of fill placed."
    ),
}

OFF_TOPIC = (
    "The monthly progress narrative summarises earthworks quantities and "
    "manpower histograms for the reporting period."
)


def _scanner(broadened):
    from app.core.standards_scanner import StandardsScanner

    return StandardsScanner(broadened=broadened)


@pytest.mark.parametrize("rule_id,text", sorted(PLANTED.items()))
def test_broadened_rules_catch_planted_deviations(rule_id, text):
    flags = _scanner(True).scan(text)
    assert any(fl.rule_id == rule_id for fl in flags), (
        f"{rule_id} missed its planted deviation: {[fl.rule_id for fl in flags]}"
    )
    # Advisory only — severity never escalates to blocking on plain text.
    assert all(fl.severity == "advisory" for fl in flags)


def test_off_topic_text_raises_no_broadened_flags():
    flags = _scanner(True).scan(OFF_TOPIC)
    assert flags == [], [fl.rule_id for fl in flags]


def test_flag_off_keeps_only_the_core_rule():
    off = _scanner(False)
    assert off.rule_ids() == ["no_approved_on_design"]
    # Broadened planted text must produce NOTHING with the flag off.
    for text in PLANTED.values():
        assert off.scan(text) == []


def test_scan_never_mutates_the_deliverable():
    sc = _scanner(True)
    text = PLANTED["undocumented_welding_procedure"]
    before = text
    advisory = sc.scan_to_markdown(text)
    assert text == before
    # The advisory is an APPENDIX block, clearly marked, never a rewrite.
    assert advisory.startswith("\n---")
    assert "advisory flags only" in advisory
    # Clean text yields an empty advisory — nothing appended at all.
    assert sc.scan_to_markdown(OFF_TOPIC) == ""
