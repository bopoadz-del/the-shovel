"""Stage 3 / precedence weights — authority + layer re-rank term.

Unit-level: contractual outweighs historical; project_record outweighs
shared_domain; the combined bonus is small (tie-breaker magnitude by default).
"""
from app.core.rag import layers as L


def test_authority_weight_orders_by_precedence():
    assert L.authority_weight("contractual") > L.authority_weight("policy")
    assert L.authority_weight("policy") > L.authority_weight("personal")
    assert L.authority_weight(None) == 0.0
    assert L.authority_weight("nonsense") == 0.0


def test_layer_weight_project_record_dominates():
    assert L.layer_weight("project_record") > L.layer_weight("company_rules")
    assert L.layer_weight("company_rules") > L.layer_weight("shared_domain")
    assert L.layer_weight(None) == 0.0


def test_precedence_bonus_l2b_contractual_beats_l1_historical():
    hi = L.precedence_bonus("project_record", "contractual")
    lo = L.precedence_bonus("shared_domain", "historical")
    assert hi > lo > 0


def test_precedence_bonus_is_small_by_default():
    # default weights keep the max bonus a tie-breaker, not a semantic override
    assert L.precedence_bonus("project_record", "contractual") <= 0.15


def test_precedence_bonus_zero_for_unlayered():
    assert L.precedence_bonus(None, None) == 0.0


def test_precedence_weights_are_env_tunable(monkeypatch):
    monkeypatch.setenv("RAG_AUTHORITY_WEIGHT", "1.0")
    monkeypatch.setenv("RAG_LAYER_WEIGHT", "0.0")
    # authority weight now dominates: contractual bonus == 1.0 * 1.0
    assert abs(L.precedence_bonus("shared_domain", "contractual") - 1.0) < 1e-9
