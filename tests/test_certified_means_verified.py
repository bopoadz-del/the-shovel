"""KS-1: 'certified' means verified.

A formula is certified only when it declares its standard AND its kernel
reproduces at least one hand-value case; the label is derived, never set by
hand. The registry check fails a formula that declares neither.

Every formula and every hand case here is synthetic and built by the test
itself; no real calculator, project or document is named. Each decision runs
on two inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.lib.construction_formulas as cf
from app.lib import formula_registry as fr

_PROBES = ("syn_ks1_area", "syn_ks1_rate")
_BASELINE = Path(__file__).parent / "fixtures" / "certification_gaps.json"


def _area(width_m: float, length_m: float):
    return {"area_m2": width_m * length_m}


def _rate(amount: float, days: float):
    return {"per_day": amount / days}


_KERNELS = {"syn_ks1_area": _area, "syn_ks1_rate": _rate}


def _register(name: str, standard: str = "") -> None:
    fn = _KERNELS[name]
    fn.__name__ = name
    fr._REGISTRY.pop(name, None)
    fr.formula(owner="base", description="Synthetic KS-1 probe.",
               display_name=f"Synthetic {name}", inputs={}, outputs={},
               standard=standard)(fn)


def _hand_dir(tmp_path: Path, monkeypatch, rows: list, name: str = "cases.json") -> Path:
    d = tmp_path / "hand_values"
    d.mkdir(exist_ok=True)
    (d / name).write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setenv("FORMULA_HAND_VALUES_DIR", str(d))
    fr.clear_hand_value_cache()
    return d


# Hand cases worked by hand: 4 m x 2.5 m = 10 m2; 900 / 6 days = 150 per day.
_GOOD = {
    "syn_ks1_area": {"formula": "syn_ks1_area", "inputs": {"width_m": 4, "length_m": 2.5},
                     "expect": {"area_m2": 10.0}, "working": "4 x 2.5 = 10"},
    "syn_ks1_rate": {"formula": "syn_ks1_rate", "inputs": {"amount": 900, "days": 6},
                     "expect": {"per_day": 150.0}, "working": "900 / 6 = 150"},
}
# The same inputs with a hand value the kernel does not reproduce.
_WRONG = {
    "syn_ks1_area": {**_GOOD["syn_ks1_area"], "expect": {"area_m2": 11.0}},
    "syn_ks1_rate": {**_GOOD["syn_ks1_rate"], "expect": {"per_day": 140.0}},
}
_STANDARD = {"syn_ks1_area": "Synthetic Standard A clause 1", "syn_ks1_rate": "Synthetic Standard B 2.3"}


@pytest.fixture(autouse=True)
def _cleanup(monkeypatch):
    monkeypatch.delenv("FORMULA_HAND_VALUES_DIR", raising=False)
    fr.clear_hand_value_cache()
    yield
    for name in _PROBES:
        fr._REGISTRY.pop(name, None)
    fr.clear_hand_value_cache()


@pytest.mark.parametrize("name", _PROBES)
def test_standard_and_reproduced_hand_value_is_certified(name, tmp_path, monkeypatch):
    _register(name, standard=_STANDARD[name])
    _hand_dir(tmp_path, monkeypatch, [_GOOD[name]])
    assert fr.verified_by_hand(name)
    assert fr.certified(name)
    out = cf.envelope_provenance(name, {})
    assert out["certified"] is True
    assert out["standard"] == _STANDARD[name]  # the declared standard is carried


@pytest.mark.parametrize("name", _PROBES)
def test_standard_without_hand_value_is_not_certified(name, tmp_path, monkeypatch):
    _register(name, standard=_STANDARD[name])
    _hand_dir(tmp_path, monkeypatch, [])
    assert not fr.verified_by_hand(name)
    assert not fr.certified(name)
    assert cf.envelope_provenance(name, {})["certified"] is False


@pytest.mark.parametrize("name", _PROBES)
def test_hand_value_without_standard_is_not_certified(name, tmp_path, monkeypatch):
    _register(name)
    _hand_dir(tmp_path, monkeypatch, [_GOOD[name]])
    assert fr.verified_by_hand(name)
    assert not fr.certified(name)
    out = cf.envelope_provenance(name, {})
    assert out["certified"] is False
    assert out["standard"] == "none declared"


@pytest.mark.parametrize("name", _PROBES)
def test_a_kernel_that_misses_its_hand_value_loses_the_label(name, tmp_path, monkeypatch):
    _register(name, standard=_STANDARD[name])
    _hand_dir(tmp_path, monkeypatch, [_GOOD[name], _WRONG[name]])
    assert not fr.certified(name)
    assert fr.hand_value_failures(name)
    assert f"formula:{name}:hand_value_mismatch" in fr.certification_gaps()


@pytest.mark.parametrize("name", _PROBES)
def test_a_raising_kernel_is_a_failed_case_not_a_pass(name, tmp_path, monkeypatch):
    _register(name, standard=_STANDARD[name])
    bad = {**_GOOD[name], "inputs": {"no_such_input": 1}}
    _hand_dir(tmp_path, monkeypatch, [bad])
    assert not fr.certified(name)
    assert any("kernel raised" in f for f in fr.hand_value_failures(name))


@pytest.mark.parametrize("name", _PROBES)
def test_registry_check_fails_a_formula_that_declares_neither(name, tmp_path, monkeypatch):
    _register(name)
    _hand_dir(tmp_path, monkeypatch, [])
    gap = f"formula:{name}:standard_or_hand_value"
    assert gap in fr.certification_gaps()
    assert gap not in _baseline()  # a new formula is never covered by the baseline


@pytest.mark.parametrize("name,declares", [
    ("syn_ks1_area", "standard"), ("syn_ks1_rate", "standard"),
    ("syn_ks1_area", "hand_value"), ("syn_ks1_rate", "hand_value"),
])
def test_declaring_either_half_passes_the_registry_check(name, declares, tmp_path, monkeypatch):
    _register(name, standard=_STANDARD[name] if declares == "standard" else "")
    _hand_dir(tmp_path, monkeypatch, [_GOOD[name]] if declares == "hand_value" else [])
    assert not [g for g in fr.certification_gaps() if g.startswith(f"formula:{name}:")]


@pytest.mark.parametrize("row,field", [
    ({"formula": "syn_ks1_area", "inputs": {"width_m": 1, "length_m": 1}, "expect": {"area_m2": 1.0}}, "working"),
    ({"formula": "syn_ks1_rate", "inputs": {"amount": 1, "days": 1}, "expect": {}, "working": "1 / 1"}, "expect"),
])
def test_a_malformed_hand_case_is_a_gap_and_certifies_nothing(row, field, tmp_path, monkeypatch):
    name = row["formula"]
    _register(name, standard=_STANDARD[name])
    _hand_dir(tmp_path, monkeypatch, [row])
    assert f"hand_value:cases.json[0]:{field}" in fr.certification_gaps()
    assert not fr.certified(name)


@pytest.mark.parametrize("missing", ["no_such_dir", "also_missing"])
def test_no_hand_value_directory_certifies_nothing(missing, tmp_path, monkeypatch):
    for name in _PROBES:
        _register(name, standard=_STANDARD[name])
    monkeypatch.setenv("FORMULA_HAND_VALUES_DIR", str(tmp_path / missing))
    fr.clear_hand_value_cache()
    assert not any(fr.certified(name) for name in _PROBES)
    assert "hand_value_dir:missing" in fr.certification_gaps()


@pytest.mark.parametrize("name", _PROBES)
def test_a_hand_case_naming_no_registered_formula_is_a_gap(name, tmp_path, monkeypatch):
    fr._REGISTRY.pop(name, None)
    _hand_dir(tmp_path, monkeypatch, [_GOOD[name]])
    assert "hand_value:cases.json[0]:unknown_formula" in fr.certification_gaps()


# --- the edges of the check ------------------------------------------------

@pytest.mark.parametrize("row,field", [
    (["not", "an", "object"], "not_an_object"),
    ({"formula": "", "inputs": {}, "expect": {"x": 1}, "working": "w"}, "formula"),
    ({"formula": "syn_ks1_area", "inputs": [4, 2.5], "expect": {"area_m2": 10.0}, "working": "w"}, "inputs"),
    ({"formula": "syn_ks1_rate", "inputs": {"amount": 1, "days": 1}, "expect": [], "working": "w"}, "expect"),
    ({"formula": "syn_ks1_rate", "inputs": {"amount": 1, "days": 1}, "expect": {"per_day": 1.0}, "working": " "},
     "working"),
])
def test_each_missing_field_of_a_hand_case_is_named(row, field, tmp_path, monkeypatch):
    _hand_dir(tmp_path, monkeypatch, [row])
    assert f"hand_value:cases.json[0]:{field}" in fr.certification_gaps()
    assert fr.hand_value_cases() == []


@pytest.mark.parametrize("text,problem", [
    ("{not json", "unreadable:JSONDecodeError"),
    ('{"formula": "syn_ks1_area"}', "not_a_list"),
])
def test_an_unreadable_or_non_list_file_is_a_gap(text, problem, tmp_path, monkeypatch):
    d = _hand_dir(tmp_path, monkeypatch, [_GOOD["syn_ks1_area"]], name="good.json")
    (d / "bad.json").write_text(text, encoding="utf-8")
    fr.clear_hand_value_cache()
    assert f"hand_value:bad.json:{problem}" in fr.certification_gaps()
    assert len(fr.hand_value_cases()) == 1  # the good file is still read


@pytest.mark.parametrize("name", _PROBES)
def test_clearing_the_cache_rereads_the_directory(name, tmp_path, monkeypatch):
    _register(name, standard=_STANDARD[name])
    d = _hand_dir(tmp_path, monkeypatch, [_WRONG[name]])
    assert not fr.certified(name)
    (d / "cases.json").write_text(json.dumps([_GOOD[name]]), encoding="utf-8")
    fr.clear_hand_value_cache()
    assert fr.certified(name)


@pytest.mark.parametrize("name,tol,within", [
    ("syn_ks1_area", 0.05, True),    # 10 vs hand 10.4: 3.8 % off, inside 5 %
    ("syn_ks1_rate", 0.05, True),    # 150 vs hand 156: 3.8 % off, inside 5 %
    ("syn_ks1_area", None, False),   # default 0.1 %: outside
    ("syn_ks1_rate", -1, False),     # a non-positive tolerance falls back to the default
    ("syn_ks1_area", True, False),   # a boolean is not a tolerance
])
def test_a_case_states_its_own_tolerance(name, tol, within, tmp_path, monkeypatch):
    _register(name, standard=_STANDARD[name])
    key = next(iter(_GOOD[name]["expect"]))
    near = {"syn_ks1_area": 10.4, "syn_ks1_rate": 156.0}[name]
    row = {**_GOOD[name], "expect": {key: near}}
    if tol is not None:
        row["rel_tol"] = tol
    _hand_dir(tmp_path, monkeypatch, [row])
    assert fr.certified(name) is within


def _labelled(width_m: float, length_m: float):
    return {"result": {"shape": "rectangle", "area_m2": width_m * length_m}}


@pytest.mark.parametrize("expect,ok", [
    ({"shape": "rectangle", "area_m2": 6.0}, True),   # 2 x 3 = 6; text compared exactly
    ({"shape": "square", "area_m2": 6.0}, False),     # wrong text
    ({"area_m2": "6"}, False),                         # text hand value against a number
    ({"shape": 6.0}, False),                           # number hand value against text
])
def test_outputs_nested_under_result_and_text_values_are_compared(expect, ok, tmp_path, monkeypatch):
    _labelled.__name__ = "syn_ks1_area"
    fr._REGISTRY.pop("syn_ks1_area", None)
    fr.formula(owner="base", description="Synthetic KS-1 probe.", display_name="Synthetic labelled",
               inputs={}, outputs={}, standard="Synthetic Standard C")(_labelled)
    _hand_dir(tmp_path, monkeypatch, [{"formula": "syn_ks1_area", "inputs": {"width_m": 2, "length_m": 3},
                                       "expect": expect, "working": "2 x 3 = 6"}])
    assert fr.certified("syn_ks1_area") is ok
    assert (fr.hand_value_failures("syn_ks1_area") == []) is ok


@pytest.mark.parametrize("name", _PROBES)
def test_failures_for_an_unregistered_name_say_so(name, tmp_path, monkeypatch):
    fr._REGISTRY.pop(name, None)
    _hand_dir(tmp_path, monkeypatch, [_GOOD[name], _WRONG[name]])
    failures = fr.hand_value_failures(name)
    assert len(failures) == 2
    assert all("no registered formula" in f for f in failures)


# --- the real registry -------------------------------------------------------

def _baseline() -> set:
    return set(json.loads(_BASELINE.read_text(encoding="utf-8")))


def _formula_of(gap: str) -> str:
    return gap.split(":")[1] if gap.startswith("formula:") else ""


def _partition(gaps: set, owner: str) -> set:
    """The gaps of one owner's formulas; "unregistered" is every gap naming no
    registered formula (so a stale baseline entry is still caught)."""
    owners = {spec.name: spec.owner for spec in fr.all_specs()}
    if owner == "unregistered":
        return {g for g in gaps if _formula_of(g) not in owners}
    return {g for g in gaps if owners.get(_formula_of(g)) == owner}


_PARTS = list(fr.OWNERS) + ["unregistered"]


@pytest.mark.parametrize("owner", _PARTS)
def test_no_registered_formula_is_outside_the_certification_baseline(owner):
    new = sorted(_partition(set(fr.certification_gaps()), owner) - _baseline())
    assert new == [], f"declare a standard or add a hand case under tests/hand_values: {new[:20]}"


@pytest.mark.parametrize("owner", _PARTS)
def test_the_certification_baseline_only_shrinks(owner):
    filled = sorted(_partition(_baseline(), owner) - set(fr.certification_gaps()))
    assert filled == [], f"remove these from tests/fixtures/certification_gaps.json: {filled[:20]}"


def test_the_baseline_never_excuses_a_wrong_hand_value_or_a_malformed_case():
    assert not [g for g in _baseline() if g.endswith(":hand_value_mismatch") or g.startswith("hand_value:")]
