"""The user's own words reach every calculator run, whatever path called it.

Order 0435Z / LANE2 root cause 3 ("21,528 sq ft" read as m2, 783 d): the
construction container forwarded the model's params only, and a model's
paraphrase in ``text`` dropped the unit the user wrote. The turn's user
words are set once where the turn starts and every run_calculation reads its
units from them too. Invented quantities.
"""
import asyncio

from app.lib import construction_formulas as cf

from tests.example_project_rates import (  # noqa: F401
    CONCRETE_RATES, ELECTRICAL_STAGE_RATES, FORMWORK_RATES, MOBILIZATION_RATES, REBAR_RATES,
)

NAME = "electrical_installation_sequence"
PARAPHRASE = {"floor_area_m2": 21528, "text": "electrical sequence for a 21528 floor", **ELECTRICAL_STAGE_RATES}
ASK = "Electrical installation sequence for a 21,528 sq ft office floor."


def _converted(result):
    conv = result.get("conversions") or (result.get("result") or {}).get("conversions") or []
    return {c.get("parameter"): round(c.get("value_out") or 0, 1) for c in conv}


def test_a_square_foot_area_is_converted_from_the_users_words():
    token = cf.set_current_ask(ASK)
    try:
        out = cf.run_calculation(NAME, dict(PARAPHRASE))
    finally:
        cf._CURRENT_ASK.reset(token)
    assert out["status"] == "success"
    assert _converted(out).get("floor_area_m2") == 2000.0


def test_without_the_users_words_nothing_is_assumed():
    out = cf.run_calculation(NAME, dict(PARAPHRASE))
    assert "floor_area_m2" not in _converted(out)


def test_the_users_words_follow_the_run_into_a_worker_thread():
    from app.core.offload import off_loop

    async def run():
        cf.set_current_ask(ASK)
        return await off_loop(cf.run_calculation, NAME, dict(PARAPHRASE))

    assert _converted(asyncio.run(run())).get("floor_area_m2") == 2000.0


def test_a_figure_the_user_did_not_write_gets_no_unit_from_the_words():
    token = cf.set_current_ask(ASK)
    try:
        out = cf.run_calculation(NAME, {"floor_area_m2": 2000, "text": "electrical sequence for 2000 m2"})
    finally:
        cf._CURRENT_ASK.reset(token)
    assert "floor_area_m2" not in _converted(out)
