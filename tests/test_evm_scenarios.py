"""Tests for scripts.generate_evm_scenarios — deterministic Q&A pairs
derived from app/prompts/construction_evm.md.

The key property the tests defend: every answer is grounded in the source
markdown, formulas/thresholds/numbers are quoted verbatim, and the
generator is reproducible.
"""
from __future__ import annotations

from scripts.generate_evm_scenarios import (
    gen_section_concepts,
    gen_formulas_with_worked_examples,
    gen_traffic_light_thresholds,
    gen_tcpi_interpretation,
    gen_forecasting_scenarios,
    gen_commitment_tracking,
    gen_common_mistakes,
    gen_root_causes,
    gen_executive_dashboard,
    gen_section_20_worked_overrun,
    gen_cost_categories_and_pillars,
    gen_lifecycle_estimate_ranges,
    gen_cbs_hierarchy,
    gen_principles_and_rules,
    generate_all,
)


# ── Schema invariant ──────────────────────────────────────────────────────


def test_every_row_has_required_keys():
    rows = generate_all()
    assert rows, "generator produced no rows"
    for r in rows:
        assert set(r.keys()) == {"instruction", "response", "source"}, r
        assert r["instruction"].strip(), f"empty instruction: {r}"
        assert r["response"].strip(), f"empty response: {r}"
        assert r["source"].startswith("construction_evm.md:"), r


# ── Volume target ─────────────────────────────────────────────────────────


def test_total_meets_target():
    rows = generate_all()
    assert 500 <= len(rows) <= 800, f"got {len(rows)} rows, target band 500-800"


# ── Determinism ───────────────────────────────────────────────────────────


def test_generator_is_deterministic():
    a = generate_all()
    b = generate_all()
    assert a == b, "generator output is not deterministic"


# ── Formulas + worked examples ────────────────────────────────────────────


def test_formulas_include_worked_examples():
    """Sample several formula rows and assert each contains both the formula
    notation and digits."""
    rows = list(gen_formulas_with_worked_examples())
    assert rows, "no formula rows produced"

    # Pick the explicit "formula and worked example" rows.
    cv_rows = [r for r in rows if "Cost Variance (CV)" in r["instruction"]]
    cpi_rows = [r for r in rows if "Cost Performance Index (CPI)" in r["instruction"]]
    eac_rows = [r for r in rows if "Estimate at Completion (EAC)" in r["instruction"]]
    tcpi_rows = [r for r in rows if "To-Complete Performance Index (TCPI)" in r["instruction"]]
    vac_rows = [r for r in rows if "Variance at Completion (VAC)" in r["instruction"]]

    samples = cv_rows + cpi_rows + eac_rows + tcpi_rows + vac_rows
    assert samples, "expected at least one labelled formula row"

    for r in samples:
        resp = r["response"]
        # Must contain the symbolic formula (an '=' sign) and digits.
        assert "=" in resp, f"formula notation missing in: {resp}"
        assert any(ch.isdigit() for ch in resp), f"no digits in worked example: {resp}"

    # Specifically verify CPI formula row mentions both 'EV / AC' and a real CPI value.
    assert any(
        ("EV / AC" in r["response"] or "EV/AC" in r["response"]) and "0.89" in r["response"]
        for r in cpi_rows
    ), "no CPI row pairs formula with the Section 20 worked value 0.89"


def test_thresholds_quote_exact_numbers():
    """Verify the RED-CPI threshold row's response contains '0.90'."""
    rows = list(gen_traffic_light_thresholds())
    red_cpi = [r for r in rows if "What CPI value triggers RED?" in r["instruction"]]
    assert red_cpi, "missing the explicit RED-CPI threshold row"
    assert "0.90" in red_cpi[0]["response"], red_cpi[0]

    # Spot-check a few more numeric thresholds appear verbatim.
    green_cpi = [r for r in rows if "What CPI value triggers GREEN?" in r["instruction"]]
    assert green_cpi
    assert "1.00" in green_cpi[0]["response"]

    red_spi = [r for r in rows if "What SPI value triggers RED?" in r["instruction"]]
    assert red_spi
    assert "0.90" in red_spi[0]["response"]


def test_section_20_uses_real_numbers():
    """Section 20 has specific dollar figures; assert they appear verbatim in some row."""
    rows = generate_all()
    text = " ".join(r["response"] for r in rows) + " ".join(r["instruction"] for r in rows)

    # Verbatim source numbers from Section 20.
    for expected in [
        "$50,000,000",  # Contract Value / BAC
        "$32,000,000",  # AC
        "$28,500,000",  # EV
        "$56,500,000",  # EAC
        "0.89",         # CPI
        "0.93",         # SPI
        "1.19",         # TCPI
        "$6.5M",        # forecast overrun
    ]:
        assert expected in text, f"Section 20 number {expected!r} missing from generator output"


# ── Per-generator volume floors ───────────────────────────────────────────


def test_per_generator_targets():
    targets = {
        "section_concepts": (gen_section_concepts, 80),
        "formulas_with_worked_examples": (gen_formulas_with_worked_examples, 80),
        "traffic_light_thresholds": (gen_traffic_light_thresholds, 50),
        "tcpi_interpretation": (gen_tcpi_interpretation, 30),
        "forecasting_scenarios": (gen_forecasting_scenarios, 30),
        "commitment_tracking": (gen_commitment_tracking, 30),
        "common_mistakes": (gen_common_mistakes, 30),
        "root_causes": (gen_root_causes, 30),
        "executive_dashboard": (gen_executive_dashboard, 30),
        "section_20_worked_overrun": (gen_section_20_worked_overrun, 30),
        "cost_categories_and_pillars": (gen_cost_categories_and_pillars, 40),
        "lifecycle_estimate_ranges": (gen_lifecycle_estimate_ranges, 10),
        "cbs_hierarchy": (gen_cbs_hierarchy, 20),
        "principles_and_rules": (gen_principles_and_rules, 30),
    }
    for name, (fn, floor) in targets.items():
        rows = list(fn())
        assert len(rows) >= floor, f"{name}: got {len(rows)}, expected >= {floor}"


# ── Source faithfulness spot-checks ───────────────────────────────────────


def test_tcpi_formula_appears_verbatim():
    rows = list(gen_tcpi_interpretation()) + list(gen_formulas_with_worked_examples())
    text = " ".join(r["response"] for r in rows)
    # Either form is acceptable (the source uses parentheses + division).
    assert "(BAC - EV) / (BAC - AC)" in text, "exact TCPI formula not present"


def test_commitment_tracking_uses_real_numbers():
    rows = list(gen_commitment_tracking())
    text = " ".join(r["response"] for r in rows)
    for expected in ["$48,000,000", "$32,000,000", "$12,500,000", "$44,500,000", "$3,500,000", "7.3%"]:
        assert expected in text, f"commitment example number {expected!r} missing"


def test_section_concepts_cover_all_20_sections():
    rows = list(gen_section_concepts())
    sources = {r["source"] for r in rows}
    for n in range(1, 21):
        assert f"construction_evm.md:{n}" in sources, f"section {n} concept rows missing"


def test_traffic_light_combined_status_present():
    rows = list(gen_traffic_light_thresholds())
    text = " ".join(r["response"] for r in rows)
    # The 4 combined-status interpretations from Section 5.
    for phrase in [
        "Ahead & Under Budget",
        "Ahead But Over Budget",
        "On Budget But Behind",
        "Behind & Over Budget",
    ]:
        assert phrase in text, f"combined-status phrase {phrase!r} missing"


def test_principles_quote_exact_strings():
    rows = list(gen_principles_and_rules())
    text = " ".join(r["response"] for r in rows)
    for expected in [
        "Projects fail more often from poor cost control than poor planning",
        "If the cost structure is wrong, the reporting is wrong",
        "Treat Causes, Not Symptoms",
        "Good Reporting Drives Good Decisions",
        "You cannot improve what you do not measure",
        "A decision without monitoring is just a hope",
    ]:
        assert expected in text, f"principle text {expected!r} missing"
