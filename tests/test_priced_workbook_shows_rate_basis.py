"""A priced workbook must say which rates are real and which are guesses.

Live-fire phase 3, F11: pricing a real MEP-heavy bill produced 20mm pipework
at 250/m (tendered: 9) and a sump pump at 250 each (tendered: 9,971) -- both
weak asset-median fallbacks, because the rate card has no Pipework or
Mechanical rows. price_line_items stamps every line with rate_source and the
sample size n, but the workbook dropped both, so a reader could not tell a
rebar rate backed by n=45 from a guess backed by nothing.

The fence: the BOQ_Detail sheet must carry a Rate Basis column, weak fallbacks
must be visibly marked, and a NO RATE line must never masquerade as priced.
"""
from __future__ import annotations

from app.lib.boq_excel import generate_cost_boq

META = {"title": "T", "project": "P", "location": "", "currency": "AED", "date": ""}


def _detail_rows(wb):
    ws = wb["BOQ_Detail"]
    return [[c.value for c in row] for row in ws.iter_rows()]


def test_rate_basis_column_exists_and_carries_the_source():
    cats = [{"name": "STRUCTURE", "items": [
        {"item_no": "A", "description": "Rebar in raft", "unit": "t",
         "qty": 100, "rate": 3324, "rate_source": "exact (cat+unit)", "n": 45},
        {"item_no": "B", "description": "Sump pump", "unit": "nr",
         "qty": 2, "rate": 250, "rate_source": "weak (asset median)", "n": 443},
    ]}]
    rows = _detail_rows(generate_cost_boq(META, cats))
    header = next(r for r in rows if r and "Item No" in r)
    assert any(h and "Rate Basis" in str(h) for h in header), header

    flat = ["|".join(str(c) for c in r if c is not None) for r in rows]
    rebar = next(x for x in flat if "Rebar in raft" in x)
    assert "exact" in rebar and "n=45" in rebar, rebar
    pump = next(x for x in flat if "Sump pump" in x)
    assert "weak" in pump.lower(), pump


def test_a_no_rate_line_is_marked_not_silently_zero():
    cats = [{"name": "MEP", "items": [
        {"item_no": "C", "description": "Chilled water pipework", "unit": "m",
         "qty": 500, "rate": 0, "rate_source": "NO RATE", "n": 0},
    ]}]
    flat = ["|".join(str(c) for c in r if c is not None)
            for r in _detail_rows(generate_cost_boq(META, cats))]
    line = next(x for x in flat if "Chilled water" in x)
    assert "NO RATE" in line, line


def test_items_without_source_fields_still_render():
    """Callers that predate the field (cost-boq categories path) must not
    crash; the basis cell is simply blank."""
    cats = [{"name": "S", "items": [
        {"item_no": "A", "description": "Blockwork", "unit": "m2",
         "qty": 10, "rate": 88},
    ]}]
    flat = ["|".join(str(c) for c in r if c is not None)
            for r in _detail_rows(generate_cost_boq(META, cats))]
    assert any("Blockwork" in x for x in flat)
