"""Tests for app.core.boq_validation (BOQ Intelligence Layer)."""

from app.core.boq_validation import validate_boq


def test_clean_boq_no_flags():
    items = [
        {"description": "Concrete", "qty": 100, "rate": 250, "total": 25000},
        {"description": "Rebar", "qty": 10, "rate": 1500, "total": 15000},
    ]
    r = validate_boq(items, declared_total=40000)
    assert [f for f in r.flags if f.severity == "critical"] == []
    assert r.summary["parsed_lines"] == 2


def test_math_mismatch_flagged():
    # 100 * 250 = 25000, but line says 26000 -- typo / stale export.
    items = [{"description": "Concrete", "qty": 100, "rate": 250, "total": 26000}]
    r = validate_boq(items)
    crit = [f for f in r.flags if f.kind == "math_mismatch"]
    assert len(crit) == 1
    assert crit[0].expected == 25000
    assert crit[0].actual == 26000


def test_bottom_line_drift_flagged():
    items = [
        {"qty": 100, "rate": 250, "total": 25000},
        {"qty": 10, "rate": 1500, "total": 15000},
    ]
    # Cover sheet claims 50000 but lines sum to 40000.
    r = validate_boq(items, declared_total=50000)
    drift = [f for f in r.flags if f.kind == "bottom_line_drift"]
    assert len(drift) == 1
    assert drift[0].delta == 10000  # declared - sum


def test_suspicious_zero_qty_with_rate():
    items = [{"description": "Spare", "qty": 0, "rate": 500, "total": 0}]
    r = validate_boq(items)
    sus = [f for f in r.flags if f.kind == "suspicious_zero"]
    assert len(sus) == 1
    assert "zero qty" in sus[0].description


def test_suspicious_zero_rate_with_qty():
    items = [{"description": "Free issue", "qty": 100, "rate": 0, "total": 0}]
    r = validate_boq(items)
    sus = [f for f in r.flags if f.kind == "suspicious_zero"]
    assert len(sus) == 1
    assert "zero rate" in sus[0].description


def test_currency_string_parsing():
    items = [{"description": "Concrete", "qty": "100", "rate": "$250.00", "total": "25,000"}]
    r = validate_boq(items)
    assert [f for f in r.flags if f.kind == "math_mismatch"] == []


def test_malformed_row_does_not_crash():
    items = ["not a dict", {"qty": 10, "rate": 5, "total": 50}]
    r = validate_boq(items)
    bad = [f for f in r.flags if f.kind == "input_unparseable"]
    assert len(bad) == 1
    # The valid row still got processed.
    assert r.summary["parsed_lines"] == 1


def test_no_declared_total_skips_bottom_line_check():
    items = [{"qty": 100, "rate": 250, "total": 25000}]
    r = validate_boq(items, declared_total=None)
    assert [f for f in r.flags if f.kind == "bottom_line_drift"] == []


def test_within_tolerance_not_flagged():
    # 100 * 250 = 25000; declared 25000.20 = 0.0008% drift, well under default.
    items = [{"qty": 100, "rate": 250, "total": 25000.20}]
    r = validate_boq(items)
    assert [f for f in r.flags if f.kind == "math_mismatch"] == []
