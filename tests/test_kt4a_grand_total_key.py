"""K-T4-a X1: grand_total stays a recognised cost-total key (runtime :5975)."""
from __future__ import annotations

import pytest

from app.agents.runtime import _COST_TOTAL_KEYS, _cost_calc_result_dict


def test_grand_total_is_a_cost_total_key():
    assert "grand_total" in _COST_TOTAL_KEYS


@pytest.mark.parametrize("total", [1200.0, 345.5])
def test_a_result_with_only_grand_total_is_a_cost_result(total):
    payload = {"calculation": "crane_cost_estimate", "result": {"grand_total": total}}
    assert _cost_calc_result_dict(payload) == {"grand_total": total}
