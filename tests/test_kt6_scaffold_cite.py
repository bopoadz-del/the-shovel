"""K-T6 PR-B: scaffold_load_capacity cites both OSHA 1926.451 and the BS EN
12811-1 service load classes, and still refuses an unrecognised duty.
Full-dotted import so the mutation gate selects this.
"""
from __future__ import annotations

import pytest

from app.agents.hats.safety.formulas.construction_formulas_safety import (
    scaffold_load_capacity,
)


@pytest.mark.parametrize("duty", ["light", "heavy"])
def test_standard_cites_both_osha_and_bs_en(duty):
    r = scaffold_load_capacity(10.0, duty=duty)
    assert "error" not in r, r
    std = r["standard"]
    assert "OSHA 1926.451" in std, std
    assert "BS EN 12811-1" in std, std


@pytest.mark.parametrize("duty", ["extreme", "class-7"])
def test_unrecognised_duty_is_refused(duty):
    # Kills the `if d not in _SCAFFOLD_DUTY_KPA` drop_rule: an unknown duty
    # refuses rather than computing against a missing value.
    r = scaffold_load_capacity(10.0, duty=duty)
    assert "error" in r, r
    assert "duty_load_kpa" not in r
