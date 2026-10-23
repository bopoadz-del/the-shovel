"""A figure the user wrote is an operand even when it equals the default.

Lane 4 on 4834817 (3/3): "Wall pour, cube 7 MPa at striking, want to strike
at 8 h - ok?" -- the striking-time calculator was refused as a defaulted run
(status not_run): the model passed the user's 7, which equals the
calculator's default 7, and the guard counted a value equal to its default
as no operand. A passed value equal to its default still counts when the
user's own words write that figure. A run on true defaults is still caught.
"""
from app.lib import construction_formulas as cf

ASK = "Wall pour, cube 7 MPa at striking, want to strike at 8 h - ok?"


def test_the_users_figure_is_an_operand_though_it_equals_the_default():
    assert cf.run_carries_operand("formwork_striking_time", {"concrete_strength_7h": 7}, "", stated=ASK)


def test_a_default_the_user_never_wrote_is_still_no_operand():
    assert not cf.run_carries_operand("formwork_striking_time", {"concrete_strength_7h": 7}, "",
                                      stated="When can I strike the wall formwork?")


def test_the_runtime_guard_lets_the_users_figure_run():
    from app.agents.runtime import _defaulted_run_refusal

    assert _defaulted_run_refusal("formwork_striking_time", {"concrete_strength_7h": 7}, ASK) is None
    refused = _defaulted_run_refusal("formwork_striking_time", {"concrete_strength_7h": 7},
                                     "When can I strike the wall formwork?")
    assert refused is None or refused["status"] == "not_run"
