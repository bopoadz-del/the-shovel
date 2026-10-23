"""
Test the additional construction formulas added post-live-chat-test.

These cover gaps found in the 13-query battery:
- Q2: guardrail height (routing miss on safety_compliance_audit)
- Q12: interim payment (parameter extraction failure on payment_certificate)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.lib.construction_formulas_additions import (
    ADDITIONAL_CALCULATORS,
    calculate_interim_payment,
    guardrail_top_rail_height,
)

# Same pattern as test_manifest_bindings.py — add app/agents to path
# so we can import formulas.py and verify the binding registry.
# Package imports, not sys.path.insert(app/agents) -- see
# tests/test_no_implicit_relative_imports.py. The old form loaded these
# files a second time under top-level names, so a pydantic class from
# `models` failed isinstance against the same class from
# `app.agents.models`.


# -- Guardrail Height -----------------------------------------------------


class TestGuardrailHeight:
    """Q2 fix: guardrail_top_rail_height returns OSHA-standard values."""

    def test_returns_dict(self):
        result = guardrail_top_rail_height()
        assert isinstance(result, dict)

    def test_top_rail_42_inches(self):
        result = guardrail_top_rail_height()
        assert result["top_rail_height_in"] == 42

    def test_top_rail_1067_mm(self):
        result = guardrail_top_rail_height()
        assert result["top_rail_height_mm"] == 1067

    def test_top_rail_1_07_m(self):
        result = guardrail_top_rail_height()
        assert result["top_rail_height_m"] == 1.07

    def test_mid_rail_halfway(self):
        result = guardrail_top_rail_height()
        assert result["mid_rail_height_in"] == 21  # half of 42
        assert result["mid_rail_height_mm"] == 533

    def test_tolerance_3_inches(self):
        result = guardrail_top_rail_height()
        assert result["tolerance_in"] == 3
        assert result["tolerance_mm"] == 76

    def test_cites_osha_standard(self):
        result = guardrail_top_rail_height()
        assert "OSHA" in result["standard"]
        assert "1926.502" in result["standard"]

    def test_note_is_meaningful(self):
        result = guardrail_top_rail_height()
        assert len(result["note"]) > 20

    def test_range_acceptable(self):
        """42 +/- 3 inches = 39 to 45 inches is acceptable."""
        result = guardrail_top_rail_height()
        low = result["top_rail_height_in"] - result["tolerance_in"]
        high = result["top_rail_height_in"] + result["tolerance_in"]
        assert low == 39
        assert high == 45


# -- Interim Payment ------------------------------------------------------


class TestInterimPayment:
    """Q12 fix: calculate_interim_payment handles natural language amounts."""

    def test_returns_dict(self):
        result = calculate_interim_payment(900000, 10)
        assert isinstance(result, dict)

    def test_900k_at_10_percent(self):
        """The exact Q12 case: 900,000 gross, 10% retention."""
        result = calculate_interim_payment(900000, 10)
        assert result["gross_valuation"] == 900000.0
        assert result["retention_percent"] == 10.0
        assert result["retention_amount"] == 90000.0
        assert result["net_payment"] == 810000.0

    def test_900k_default_retention(self):
        """Default retention is 10%."""
        result = calculate_interim_payment(900000)
        assert result["retention_amount"] == 90000.0
        assert result["net_payment"] == 810000.0

    def test_zero_retention(self):
        result = calculate_interim_payment(500000, 0)
        assert result["retention_amount"] == 0.0
        assert result["net_payment"] == 500000.0

    def test_5_percent_retention(self):
        """FIDIC cap scenario: 5% retention."""
        result = calculate_interim_payment(1000000, 5)
        assert result["retention_amount"] == 50000.0
        assert result["net_payment"] == 950000.0

    def test_accepts_integers(self):
        result = calculate_interim_payment(900000, 10)
        assert result["gross_valuation"] == 900000.0

    def test_accepts_floats(self):
        result = calculate_interim_payment(875000.50, 10)
        assert result["gross_valuation"] == 875000.50
        assert result["retention_amount"] == 87500.05

    def test_calculation_string_present(self):
        result = calculate_interim_payment(900000, 10)
        assert "900000" in result["calculation"]
        assert "10%" in result["calculation"]
        assert "810000" in result["calculation"]

    def test_cites_fidic(self):
        result = calculate_interim_payment(100000)
        assert "FIDIC" in result["standard"]

    def test_note_warns_about_cap(self):
        result = calculate_interim_payment(100000)
        assert "cap" in result["note"].lower()


# -- Registry -------------------------------------------------------------


class TestAdditionalCalculatorsRegistry:
    """ADDITIONAL_CALCULATORS registry must be consistent."""

    def test_registry_has_guardrail(self):
        assert "guardrail_top_rail_height" in ADDITIONAL_CALCULATORS

    def test_registry_has_interim_payment(self):
        assert "calculate_interim_payment" in ADDITIONAL_CALCULATORS

    def test_all_entries_callable(self):
        for name, fn in ADDITIONAL_CALCULATORS.items():
            assert callable(fn), f"'{name}' is not callable"

    def test_guardrail_runs_without_error(self):
        fn = ADDITIONAL_CALCULATORS["guardrail_top_rail_height"]
        result = fn()
        assert "top_rail_height_in" in result

    def test_interim_payment_runs_without_error(self):
        fn = ADDITIONAL_CALCULATORS["calculate_interim_payment"]
        result = fn(100000, 10)
        assert "net_payment" in result


# -- Binding Resolver Coverage --------------------------------------------
# These exercise the NEW lines in formulas.py (_get_additional_calculators
# and the registry loop) so diff-cover sees them covered.


class TestBindingResolver:
    """Verify formulas.py _build_registry includes additions namespace."""

    def test_registry_includes_guardrail(self):
        from app.agents.formulas import resolve_binding
        fn = resolve_binding("guardrail_top_rail_height")
        assert fn is not None
        result = fn()
        assert result["top_rail_height_in"] == 42

    def test_registry_includes_interim_payment(self):
        from app.agents.formulas import resolve_binding
        fn = resolve_binding("calculate_interim_payment")
        assert fn is not None
        result = fn(900000, 10)
        assert result["net_payment"] == 810000.0

    def test_registry_includes_full_path(self):
        from app.agents.formulas import resolve_binding
        fn = resolve_binding("app.lib.construction_formulas_additions.guardrail_top_rail_height")
        assert fn is not None

    def test_registry_includes_bare_names(self):
        from app.agents.formulas import resolve_binding
        assert resolve_binding("guardrail_top_rail_height") is not None
        assert resolve_binding("calculate_interim_payment") is not None
