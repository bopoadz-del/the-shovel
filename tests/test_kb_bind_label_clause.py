"""Same-unit knowledge inputs: a label binds within its own clause, and a
variable's declared description words bind it when its name words are not
written.

Each test builds its own entry and ask; no shipped entry or exam text is
read and no expected value is copied from a product answer.
"""
from __future__ import annotations

import pytest

from app.blocks import _knowledge as K


def _entry(variables: dict) -> dict:
    return {"id": "t.entry", "type": "formula",
            "variables": {n: {"unit": u, "desc": d} for n, (u, d) in variables.items()}}


_PUSH_PULL = _entry({"push_force": ("kN", ""), "pull_force": ("kN", "")})


@pytest.mark.parametrize("text,expected", [
    # the label after the clause mark belongs to the next figure
    ("pull of 40 kN, push from the jack of 25 kN",
     {"push_force": 25.0, "pull_force": 40.0}),
    ("pull of 40 kN; push from the jack of 25 kN",
     {"push_force": 25.0, "pull_force": 40.0}),
    ("pull of 40 kN. Push from the jack of 25 kN",
     {"push_force": 25.0, "pull_force": 40.0}),
    ("pull of 40 kN\npush from the jack of 25 kN",
     {"push_force": 25.0, "pull_force": 40.0}),
    # no clause mark: "push" is nearer the 40 kN "pull" already names
    ("pull of 40 kN and push from the jack of 25 kN", None),
])
def test_a_label_binds_within_its_own_clause(text, expected):
    assert K.bind_entry_variables(_PUSH_PULL, text) == expected


@pytest.mark.parametrize("text,expected", [
    # the label after its figure ends at the comma, before the next figure
    ("40 kN of push, 25 kN of pull", {"push_force": 40.0, "pull_force": 25.0}),
    ("40 kN of pull, 25 kN of push", {"push_force": 25.0, "pull_force": 40.0}),
])
def test_a_label_after_its_figure_stays_in_that_figures_clause(text, expected):
    assert K.bind_entry_variables(_PUSH_PULL, text) == expected


@pytest.mark.parametrize("text,expected", [
    ("push: 40 kN, pull: 25 kN", {"push_force": 40.0, "pull_force": 25.0}),
    ("pull: 40 kN, push: 25 kN", {"push_force": 25.0, "pull_force": 40.0}),
])
def test_a_colon_does_not_end_a_label_clause(text, expected):
    assert K.bind_entry_variables(_PUSH_PULL, text) == expected


_TARE = {
    "gross_mass": ("kg", "Bucket plus wet sand"),
    "tare_mass": ("kg", "The empty bucket"),
}


@pytest.mark.parametrize("text,expected", [
    ("bucket plus wet sand 50 kg, empty bucket 20 kg",
     {"gross_mass": 50.0, "tare_mass": 20.0}),
    ("empty bucket 18 kg, bucket plus wet sand 61 kg",
     {"gross_mass": 61.0, "tare_mass": 18.0}),
])
def test_description_words_bind_when_name_words_are_not_written(text, expected):
    assert K.bind_entry_variables(_entry(_TARE), text) == expected


@pytest.mark.parametrize("descs,expected", [
    (("Bucket plus wet sand", "The empty bucket"),
     {"gross_mass": 50.0, "tare_mass": 20.0}),
    # nothing declared: two different figures, no basis to choose
    (("", ""), None),
])
def test_the_declared_description_is_what_decides(descs, expected):
    entry = _entry({"gross_mass": ("kg", descs[0]), "tare_mass": ("kg", descs[1])})
    assert K.bind_entry_variables(
        entry, "bucket plus wet sand 50 kg, empty bucket 20 kg") == expected


_STOCK = _entry({
    "total_received": ("t", "Quantity received since the start"),
    "total_consumption": ("t", "Quantity actually consumed since the start"),
})


@pytest.mark.parametrize("text,expected", [
    # "consumed" is written nearer 80 t, which "received" already names
    ("Steel received so far is 80 t and consumed so far is 60 t",
     {"total_received": 80.0, "total_consumption": 60.0}),
    ("Steel received so far is 30 t and consumed so far is 90 t",
     {"total_received": 30.0, "total_consumption": 90.0}),
])
def test_description_words_choose_only_among_figures_no_name_took(text, expected):
    assert K.bind_entry_variables(_STOCK, text) == expected


@pytest.mark.parametrize("descs,expected", [
    # "soil" is in both descriptions: it names neither
    (("Mould plus wet soil", "Empty mould, soil weighed net"),
     [{"gross_mass": {"gross"}, "tare_mass": {"tare"}},
      {"gross_mass": {"plus", "wet"}, "tare_mass": {"empty", "weighed", "net"}}]),
    (("Mould plus wet soil", "Empty mould"),
     [{"gross_mass": {"gross"}, "tare_mass": {"tare"}},
      {"gross_mass": {"plus", "wet", "soil"}, "tare_mass": {"empty"}}]),
])
def test_a_description_word_both_inputs_declare_names_neither(descs, expected):
    names = ["gross_mass", "tare_mass"]
    got = K._own_words(names, dict(zip(names, descs)))
    assert got == expected


@pytest.mark.parametrize("text,expected", [
    # "barge" names the formula in the ask; it labels no figure
    ("the barge trip count for 950 t and 300 t", None),
    ("the barge trip count: total 950 t, one 300 t capacity",
     {"cargo": 950.0, "hold": 300.0}),
])
def test_a_title_word_is_no_description_word(text, expected):
    entry = _entry({"cargo": ("t", "Total cargo to move"),
                    "hold": ("t", "Capacity of one barge hold")})
    entry["title"] = "Barge trip count"
    assert K.bind_entry_variables(entry, text) == expected
