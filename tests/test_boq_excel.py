"""The platform must GENERATE formula-linked cost BOQ workbooks (not static
values) matching the gold-standard medieval_modern_boq pattern: BOQ_Detail with
=Qty*Rate, category subtotals =SUM(), BOQ_Summary cross-referencing the detail
subtotals (=BOQ_Detail!G..), a Cover that links to the summary, and charts.

Construction is cost — a BOQ without live Rate x Qty = Amount is just a list.
"""
from __future__ import annotations


SAMPLE_META = {
    "title": "Test Villa — Bill of Quantities",
    "project": "Test Villa",
    "location": "Kingdom of Saudi Arabia",
    "currency": "SAR",
    "date": "April 2026",
}
SAMPLE_CATEGORIES = [
    {"name": "Site Works", "items": [
        {"item_no": "A.1", "description": "Site clearing", "unit": "Lot", "qty": 1, "rate": 120000},
        {"item_no": "A.2", "description": "Topsoil stripping", "unit": "sqm", "qty": 15000, "rate": 15},
    ]},
    {"name": "Substructure", "items": [
        {"item_no": "B.1", "description": "Bored piles", "unit": "nos", "qty": 120, "rate": 8500},
        {"item_no": "B.2", "description": "Pile caps", "unit": "cum", "qty": 680, "rate": 1200},
    ]},
]


def _formulas(ws):
    return [c.value for row in ws.iter_rows() for c in row
            if isinstance(c.value, str) and c.value.startswith("=")]


def test_cost_boq_detail_uses_qty_times_rate_formulas():
    from app.lib.boq_excel import generate_cost_boq
    wb = generate_cost_boq(SAMPLE_META, SAMPLE_CATEGORIES)
    assert {"Cover", "BOQ_Detail", "BOQ_Summary", "Cost_Charts"} <= set(wb.sheetnames)
    det = _formulas(wb["BOQ_Detail"])
    # every line-item amount is a LIVE =<qtycell>*<ratecell>, never a static number
    mult = [f for f in det if "*" in f]
    assert len(mult) == 4, f"expected 4 =Qty*Rate amount formulas, got {mult}"
    # each category has a =SUM() subtotal
    sums = [f for f in det if f.startswith("=SUM(")]
    assert len(sums) == 2, f"expected 2 category subtotals, got {sums}"


def test_cost_boq_summary_cross_references_detail_subtotals():
    from app.lib.boq_excel import generate_cost_boq
    wb = generate_cost_boq(SAMPLE_META, SAMPLE_CATEGORIES)
    summ = _formulas(wb["BOQ_Summary"])
    # category amounts pull from the detail subtotal cells (cross-sheet links)
    assert sum("BOQ_Detail!" in f for f in summ) >= 2, summ
    # a grand total exists
    assert any(f.startswith("=SUM(") or "+" in f for f in summ), summ


def test_cost_boq_has_charts_and_cover_links_summary():
    from app.lib.boq_excel import generate_cost_boq
    wb = generate_cost_boq(SAMPLE_META, SAMPLE_CATEGORIES)
    assert len(wb["Cost_Charts"]._charts) >= 1, "Cost_Charts must embed a real chart"
    cover = _formulas(wb["Cover"])
    assert any("BOQ_Summary!" in f for f in cover), "Cover must link to the summary total"


def test_cost_boq_formulas_compute_correct_totals():
    """Load-and-evaluate guard: the live formulas must actually produce the
    right numbers (Site Works 120000 + 225000 = 345000; grand total includes
    Substructure 120*8500 + 680*1200 = 1,020,000 + 816,000)."""
    import io
    from app.lib.boq_excel import generate_cost_boq, evaluate_workbook_total
    wb = generate_cost_boq(SAMPLE_META, SAMPLE_CATEGORIES)
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    total = evaluate_workbook_total(buf)
    # 120000 + 15000*15 + 120*8500 + 680*1200 = 120000+225000+1020000+816000
    assert total == 2181000, f"computed construction total wrong: {total}"
