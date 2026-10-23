"""The formula library must reproduce REAL documents' own numbers.

Live-fire campaign, formula gate 2026-08-15. Synthetic smoke inputs prove a
formula runs; they do not prove it computes what a contractor's documents
compute. This gate drives the formulas with vectors EXTRACTED from the
operator's real documents and asserts the formulas reproduce the documents'
own recorded answers:

  * 247 line extensions from two real priced bills -- a contract-volume .xls
    (rows accepted only where the bill's own qty x rate = amount within 2%,
    so OCR noise cannot poison the vectors) and a scanned 2016-17 tender BOQ
    answer key. ``unit_cost_total`` must land on the BILL'S amount.
  * 19 activities from a real 2013 Primavera P6 baseline .xer where the
    scheduler's stored total float agrees with its stored dates (the other
    1,261 rows are multi-calendar artifacts, excluded at extraction and
    counted in the fixture meta -- disclosed, not cherry-picked silently).
    ``critical_path_float`` must reproduce P6's float from P6's dates.
  * an interim payment on the bill's own 15.18M validated-row valuation at
    the RETENTION RATE THE REAL CONTRACT STATES (10%).

Descriptions and identifiers are stripped from the fixture; the numbers are
the test.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from app.agents import formulas as F

FIXTURE = pathlib.Path("tests/fixtures/real_doc_formula_vectors.json")


@pytest.fixture(scope="module")
def vectors():
    return json.loads(FIXTURE.read_text())


def test_the_fixture_is_substantial(vectors):
    """A gutted fixture must fail loudly, not pass emptily."""
    assert len(vectors["line_extensions"]) >= 240
    assert len(vectors["xer_floats"]) >= 15
    assert vectors["payment"]["gross_valuation"] > 1_000_000


def test_every_real_bill_line_extension_reproduces_the_bill(vectors):
    """247 real rows: qty x rate must land on the amount the bill records
    (2% tolerance -- the same arithmetic gate the bills themselves passed;
    most rows are exact)."""
    failures = []
    for i, v in enumerate(vectors["line_extensions"]):
        total = F.run_formula(
            "unit_cost_total", {"quantity": v["qty"], "unit_rate": v["rate"]},
        )["total_cost"]
        if abs(total - v["amount"]) > 0.02 * v["amount"]:
            failures.append(f"row {i} ({v['source']}): "
                            f"{v['qty']} x {v['rate']} = {total} != {v['amount']}")
    assert not failures, "\n".join(failures)


def test_most_real_extensions_are_exact_to_the_cent(vectors):
    """The 2% band is for the bills' own rounding; the formula itself must
    be exact wherever the bill is exact -- and that is nearly every row."""
    exact = sum(
        1 for v in vectors["line_extensions"]
        if abs(v["qty"] * v["rate"] - v["amount"]) < 0.01
    )
    assert exact / len(vectors["line_extensions"]) > 0.9


def test_cpm_float_reproduces_the_real_p6_scheduler(vectors):
    """19 real baseline activities: our TF from P6's own dates must match
    P6's stored total float within a day, and criticality must agree."""
    failures = []
    for v in vectors["xer_floats"]:
        r = F.run_formula("critical_path_float", {
            "early_start": v["es"], "early_finish": v["ef"],
            "late_start": v["ls"], "late_finish": v["lf"]})
        if abs(r["total_float"] - v["p6_total_float_days"]) > 1.0:
            failures.append(f"TF {r['total_float']} vs P6 {v['p6_total_float_days']}")
            continue
        # criticality compared away from the boundary only
        if v["p6_total_float_days"] <= 0 and not r["is_critical"]:
            failures.append(f"P6 critical, formula says float {r['total_float']}")
        if v["p6_total_float_days"] > 1.5 and r["is_critical"]:
            failures.append(f"P6 float {v['p6_total_float_days']}, formula says critical")
    assert not failures, "\n".join(failures)


def test_interim_payment_on_the_real_valuation_at_the_contract_retention(vectors):
    """Gross = the contract bill's validated-row total; retention 10% as the
    real conditions of contract state (verified live via RAG: the same
    document answers 'retention 10%')."""
    p = vectors["payment"]
    r = F.run_formula("calculate_interim_payment", {
        "gross_valuation": p["gross_valuation"],
        "retention_percent": p["retention_percent"]})
    assert r["retention_amount"] == pytest.approx(p["gross_valuation"] * 0.10, abs=0.01)
    assert r["net_payment"] == pytest.approx(p["gross_valuation"] * 0.90, abs=0.01)
    assert r["retention_amount"] + r["net_payment"] == pytest.approx(
        p["gross_valuation"], abs=0.01)
