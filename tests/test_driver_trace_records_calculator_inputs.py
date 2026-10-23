"""The driver's turn trace records the inputs the model passed a calculator.

Lane 4 on 8a87bf4: bolt shear switched to the bolt's gross shank area in 1
run of 3, and the trace could not show what the model passed (it named only
missing and defaulted inputs). Each construction_calc call now records the
numbers it was given; a text value only as "text", so the trace still
carries none of the user's words (at most ten).
"""
from app.agents.driver.loop import _given_inputs


def test_numbers_are_recorded_and_text_only_as_text():
    got = _given_inputs({"calculation": "x", "params": {"bolt_area_mm2": "M20", "n_shear_planes": 2,
                                                         "shear_strength_mpa": 400.0, "text": "the ask"}})
    assert got == {"bolt_area_mm2": "text", "n_shear_planes": 2, "shear_strength_mpa": 400.0}


def test_the_users_words_never_enter_the_record_and_it_is_bounded():
    got = _given_inputs({"params": {"note": "the qxsecret widgets", "flag": True}})
    assert got == {"note": "text"} and "qxsecret" not in str(got)
    assert len(_given_inputs({"params": {f"k{i}": i for i in range(15)}})) == 10


def test_no_params_records_nothing():
    assert _given_inputs({"calculation": "x"}) == {}
