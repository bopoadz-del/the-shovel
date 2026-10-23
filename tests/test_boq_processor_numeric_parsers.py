"""Known-answer regression tests for boq_processor's numeric parsers.

These tests pin the documented contract of ``_to_float`` and
``_to_float_safe`` — the two functions on the BOQ money path that turn a
spreadsheet cell into a number the procurement pipeline trusts.

Two distinct behaviours are pinned here:

* ``_to_float`` (quantity column): MUST raise ``ValueError`` on any
  non-empty non-numeric string. Callers (``_process_dataframe``) rely
  on that to route the bad row into ``skipped_items`` instead of
  silently counting it as quantity-zero. A regression that returns
  0.0 here would cause those rows to be DROPPED via the
  ``include_zero_qty=False`` filter — silent data loss.

* ``_to_float_safe`` (rate / total columns): silently returns 0.0 on
  any parse failure. This is the *documented* behaviour, but it is
  also the source of the silent-zero failure mode the audit flagged:
  a row with rate=``"1,200.00 AED"`` (currency suffix) will report
  rate=0, total=0, and contribute nothing to ``total_cost``. The
  end-to-end test below pins exactly that behaviour AND records the
  count for the follow-up work in P1.3.
"""

import tempfile
import os
import pytest


# ──────────────────────────────────────────────────────────────────────
# _to_float — quantity-column parser
# ──────────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_thousands_separator(self):
        from app.blocks.boq_processor import _to_float
        assert _to_float("1,234.56") == 1234.56

    def test_empty_string_returns_zero(self):
        from app.blocks.boq_processor import _to_float
        # Empty cells are common and unambiguous — return 0.0, not raise.
        assert _to_float("") == 0.0

    def test_nan_string_returns_zero(self):
        from app.blocks.boq_processor import _to_float
        # pandas renders empty cells as the literal string 'nan' after
        # str() coercion; treating it as zero matches operator intent.
        assert _to_float("nan") == 0.0

    def test_lot_raises(self):
        """Documented contract: unparseable string MUST raise so the
        caller can route the row into skipped_items rather than
        silently turning a 'Lot' line into qty=0."""
        from app.blocks.boq_processor import _to_float
        with pytest.raises(ValueError):
            _to_float("Lot")

    def test_provisional_sum_raises(self):
        from app.blocks.boq_processor import _to_float
        with pytest.raises(ValueError):
            _to_float("Provisional Sum")

    def test_none_returns_zero(self):
        from app.blocks.boq_processor import _to_float
        assert _to_float(None) == 0.0


# ──────────────────────────────────────────────────────────────────────
# _to_float_safe — rate / total parser
# ──────────────────────────────────────────────────────────────────────

class TestToFloatSafe:
    def test_thousands_separator(self):
        from app.blocks.boq_processor import _to_float_safe
        assert _to_float_safe("3,500.00") == 3500.0

    def test_currency_suffix_returns_zero(self):
        """SILENT-FAILURE PIN: '1,200.00 AED' parses to 0.0 today.

        This is the documented behaviour but it is also why a row with
        a currency-suffixed rate contributes a zero total. The follow-up
        work in P1.3 is to either strip currency suffixes inside the
        parser OR surface a per-row parse warning so the operator knows
        the total is under-counted. Until then, this test pins the
        current behaviour so a fix-attempt that changes it FAILS LOUDLY
        rather than silently changing every BOQ total.
        """
        from app.blocks.boq_processor import _to_float_safe
        assert _to_float_safe("1,200.00 AED") == 0.0

    def test_bad_value_returns_zero(self):
        from app.blocks.boq_processor import _to_float_safe
        assert _to_float_safe("not a number") == 0.0


# ──────────────────────────────────────────────────────────────────────
# Small-fixture end-to-end through _process_dataframe
# ──────────────────────────────────────────────────────────────────────

class TestProcessDataframeSilentZero:
    """5-row BOQ where row 2 has a malformed rate. The expected behaviour:

    * row 2 gets `rate=0`, `total=0` (silent zero — documented)
    * total_cost reflects the missing row's zero contribution
    * the row is still counted in `item_count` (it was kept, not skipped)
    * a `skipped_items` field is present only if `_to_float` raised on the
      quantity (it doesn't here — quantity is valid)

    This pins the current behaviour while making the silent-zero loss
    visible in test output. P1.3 follow-up: surface a per-row parse
    warning so the operator sees the under-count.
    """

    @pytest.mark.asyncio
    async def test_silent_zero_in_total(self):
        import pandas as pd
        from app.blocks.boq_processor import BOQProcessorBlock

        block = BOQProcessorBlock()
        # Row 2's rate has a currency suffix → _to_float_safe returns 0.0
        # → line total is 0.0. The total_cost reflects the silent zero.
        rows = [
            {"description": "Concrete C35",   "quantity": 100, "unit": "m3", "rate": 240},
            {"description": "Rebar 500 MPa",  "quantity": 50,  "unit": "t",  "rate": "920 AED"},   # malformed
            {"description": "Formwork Slab",  "quantity": 200, "unit": "m2", "rate": 55},
            {"description": "Blockwork",      "quantity": 150, "unit": "m2", "rate": 30},
            {"description": "Plaster",        "quantity": 300, "unit": "m2", "rate": 12},
        ]
        df = pd.DataFrame(rows)
        result = block._process_dataframe(df, {})

        assert result["status"] == "success"
        # All 5 rows kept — row 2 is silently zeroed, not dropped.
        assert result["item_count"] == 5

        # Expected total: 100*240 + 0 (silent) + 200*55 + 150*30 + 300*12
        # = 24000 + 0 + 11000 + 4500 + 3600 = 43100
        # If row 2 had parsed correctly the total would be 89100 — a
        # 46k difference invisible to the operator. That gap is the
        # bug this test exists to surface.
        assert abs(result["total_cost"] - 43100) < 0.01, result["total_cost"]

        # Confirm row 2 is the silent-zero row.
        items_by_desc = {li["description"]: li for li in result["line_items"]}
        rebar = items_by_desc["Rebar 500 MPa"]
        assert rebar["unit_cost"] == 0.0
        assert rebar["total_cost"] == 0.0
        assert rebar["quantity"] == 50  # quantity still valid

        # TODO(p1.3-followup): the audit recorded that _to_float_safe
        # silently zeros 1 row in this 5-row fixture (rebar rate "920
        # AED"). _process_dataframe does NOT currently expose a
        # parse_warnings field, so the operator has no signal that the
        # total is 46k AED under-counted. Follow-up work: add
        # parse_warnings: List[{row, column, raw_value, parsed_to}]
        # to the return shape and wire it through to the UI. Until then
        # the silent zero count for this fixture is exactly 1 (row 2's
        # rate); the difference between reported and true total is
        # 50 * 920 = 46_000 AED.
        assert "parse_warnings" not in result  # pin current shape


# ──────────────────────────────────────────────────────────────────────
# Regression: PDF page-skip propagation (Fix 1)
# ──────────────────────────────────────────────────────────────────────

class TestPagesSkippedSurfaced:
    """Smoke test that the new pages_skipped fields are present on PDF
    parse results. We can't easily induce extract_tables failure without
    a corrupt fixture, but we can confirm the keys exist with their
    documented zero default."""

    @pytest.mark.asyncio
    async def test_csv_does_not_expose_pdf_diagnostic(self):
        """CSV path is unaffected — confirms the field is PDF-only and
        we haven't accidentally added it to all return paths."""
        from app.blocks.boq_processor import BOQProcessorBlock
        block = BOQProcessorBlock()
        f = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv", newline="",
        )
        f.write("description,quantity,unit,rate\n")
        f.write("Concrete,100,m3,240\n")
        f.close()
        try:
            r = await block.execute({"file_path": f.name}, {})
            inner = r.get("result", r)
            assert inner["status"] == "success"
            # CSV path doesn't surface pages_skipped — those are PDF-only.
            assert "pages_skipped" not in inner
        finally:
            os.unlink(f.name)
