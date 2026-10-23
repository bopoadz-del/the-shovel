"""W9 dashboard monotonicity + W10 designed-vs-built planted deviations.

New test shapes (2026-07-26):
- W9: ORDER oracles on the health aggregate — worsening one domain can never
  raise the overall score, and every cross-domain flag must trace back to a
  domain that was actually reported unhealthy.
- W10: planted design/as-built measurement sets where each deviation's size
  and severity class is known in advance, plus tolerance metamorphics: the
  same data must flag MORE with a tighter tolerance and NOTHING with a loose
  one — and the sign-off must follow the worst surviving severity.
"""
import asyncio

import pytest

from app.blocks.project_dashboard import ProjectDashboardBlock


# ── W9 — health aggregate order oracles ─────────────────────────────────────

def _health(domain_status):
    block = ProjectDashboardBlock()
    return asyncio.run(block.process({"action": "health_check",
                                      "domain_status": domain_status}))

HEALTHY = {
    "schedule": {"status": "healthy", "progress": 90},
    "cost": {"status": "healthy", "variance": 3},
    "quality": {"status": "healthy", "pass_rate": 95},
}


def test_degrading_one_domain_never_raises_the_score():
    base = _health(HEALTHY)["overall_score"]
    for domain, bad in (("schedule", {"status": "delayed"}),
                        ("cost", {"status": "overrun", "variance": 25}),
                        ("quality", {"status": "failed", "pass_rate": 60})):
        worse = dict(HEALTHY)
        worse[domain] = bad
        assert _health(worse)["overall_score"] <= base, domain


def test_every_flag_traces_to_an_unhealthy_domain():
    mixed = dict(HEALTHY)
    mixed["schedule"] = {"status": "delayed"}
    r = _health(mixed)
    unhealthy = {"schedule"}
    for flag in r["cross_domain_flags"]:
        assert flag["source"] in unhealthy, flag


def test_all_healthy_yields_no_flags_and_high_score():
    r = _health(HEALTHY)
    assert r["cross_domain_flags"] == []
    assert r["overall_status"] == "healthy"


# ── W10 — planted designed-vs-built deviations ──────────────────────────────

DESIGN = [
    {"type": "slab_thickness", "value": 0.300, "unit": "m"},
    {"type": "column_width", "value": 0.400, "unit": "m"},
    {"type": "wall_length", "value": 5.000, "unit": "m"},
]
AS_BUILT = [
    {"type": "slab_thickness", "value": 0.285, "unit": "m", "raw": "Slab S-01"},   # -15 mm
    {"type": "column_width", "value": 0.425, "unit": "m", "raw": "Column C-07"},   # +25 mm
    {"type": "wall_length", "value": 5.005, "unit": "m", "raw": "Wall W-03"},      # +5 mm
]


def _report(tolerance_mm=10.0):
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    return asyncio.run(c.as_built_deviation_report(
        {"measurements": AS_BUILT},
        {"design_measurements": DESIGN, "tolerance_mm": tolerance_mm},
    ))


def test_planted_deviations_flagged_with_correct_severity():
    r = _report(tolerance_mm=10.0)
    assert r["status"] == "success", r
    devs = {d["element"]: d for d in r["deviations"]}
    # 15 mm on the slab: over tolerance (10), under 2x -> minor.
    assert devs["Slab S-01"]["severity"] == "minor"
    assert devs["Slab S-01"]["deviation"] == pytest.approx(0.015, abs=1e-6)
    # 25 mm on the column: over 2x tolerance -> major.
    assert devs["Column C-07"]["severity"] == "major"
    # 5 mm on the wall: inside tolerance -> NOT flagged.
    assert "Wall W-03" not in devs
    assert r["deviation_summary"]["total_deviations"] == 2
    # Worst surviving severity is major -> conditional sign-off, never APPROVED.
    assert r["sign_off_status"] == "CONDITIONAL"


def test_loose_tolerance_clears_everything_and_approves():
    r = _report(tolerance_mm=30.0)
    assert r["deviation_summary"]["total_deviations"] == 0
    assert r["sign_off_status"] == "APPROVED"


def test_tight_tolerance_flags_the_wall_too():
    r = _report(tolerance_mm=4.0)
    elements = {d["element"] for d in r["deviations"]}
    assert elements == {"Slab S-01", "Column C-07", "Wall W-03"}


def test_no_inputs_is_an_honest_error_never_approved():
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    r = asyncio.run(c.as_built_deviation_report({}, {}))
    assert r["status"] == "error"
    assert "sign_off_status" not in r

    r2 = asyncio.run(c.as_built_deviation_report(
        {"measurements": AS_BUILT}, {},
    ))
    assert r2["status"] == "error"
    assert "design_measurements" in r2["error"]
