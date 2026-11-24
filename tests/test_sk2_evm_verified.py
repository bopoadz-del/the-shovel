"""S-K2 item 4: earned value measures verified against their definitions.

Every expected value below is worked out here, from the definitions, on
inputs this file builds itself -- no import of the engine's arithmetic:

    SV = EV - PV        CV = EV - AC
    SPI = EV / PV       CPI = EV / AC
    EAC = BAC / CPI     ETC = EAC - AC      VAC = BAC - EAC

Two rules the definitions add on top of the arithmetic:
  * a derived figure (EAC, the CPI health band) is read from the unrounded
    CPI; rounding is for display only;
  * CPI is undefined when no actual cost has been spent, so its health band
    is undefined too -- not CRITICAL.
"""
from __future__ import annotations

import random

import pytest

from app.agents.hats.commercial.formulas.construction_knowledge import calculate_evm

_SEEDS = (11, 23, 37, 59)


def _cases(seed: int) -> list[tuple[float, float, float, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(25):
        bac = float(rng.randint(10_000, 5_000_000))
        pv = round(bac * rng.uniform(0.05, 0.95), 2)
        ev = round(pv * rng.uniform(0.6, 1.3), 2)
        ac = round(ev * rng.uniform(0.7, 1.4), 2)
        out.append((bac, pv, ev, ac))
    return out


@pytest.mark.parametrize("seed", _SEEDS)
def test_measures_match_their_definitions(seed):
    for bac, pv, ev, ac in _cases(seed):
        r = calculate_evm(bac=bac, pv=pv, ev=ev, ac=ac)
        cpi = ev / ac
        eac = bac / cpi
        assert r["SV"] == pytest.approx(ev - pv, abs=0.01)
        assert r["CV"] == pytest.approx(ev - ac, abs=0.01)
        assert r["SPI"] == pytest.approx(ev / pv, abs=0.0005)
        assert r["CPI"] == pytest.approx(cpi, abs=0.0005)
        assert r["EAC"] == pytest.approx(eac, abs=0.01)
        assert r["ETC"] == pytest.approx(eac - ac, abs=0.02)
        assert r["VAC"] == pytest.approx(bac - eac, abs=0.02)
        assert r["status"]["cost"] == ("UNDER BUDGET" if ev >= ac else "OVER BUDGET")
        assert r["status"]["schedule"] == ("AHEAD" if ev >= pv else "BEHIND")


@pytest.mark.parametrize("seed", _SEEDS)
def test_pmi_names_and_aliases_give_the_same_measures(seed):
    for bac, pv, ev, ac in _cases(seed):
        a = calculate_evm(bac=bac, pv=pv, ev=ev, ac=ac)
        b = calculate_evm(bac, ev, pv, ac)  # positional: bac, bcwp, bcws, acwp
        for key in ("SV", "CV", "SPI", "CPI", "EAC", "ETC", "VAC"):
            assert a[key] == b[key], key


@pytest.mark.parametrize("seed", _SEEDS)
def test_without_bac_no_forecast_is_invented(seed):
    for _bac, pv, ev, ac in _cases(seed):
        r = calculate_evm(pv=pv, ev=ev, ac=ac)
        assert r["EAC"] is None and r["ETC"] is None and r["VAC"] is None
        assert r["CV"] == pytest.approx(ev - ac, abs=0.01)


def _band(cpi: float) -> str:
    return "GOOD" if cpi >= 1 else ("WARNING" if cpi >= 0.9 else "CRITICAL")


@pytest.mark.parametrize(
    ("ev", "ac"),
    [
        (8_996.0, 10_000.0),   # CPI 0.8996 -> 0.9 at 3dp, below the 0.9 band
        (9_999.6, 10_000.0),   # CPI 0.99996 -> 1.0 at 3dp, below the 1.0 band
        (9_000.0, 10_000.0),   # exactly on the band edge
        (12_000.0, 10_000.0),
    ],
)
def test_health_band_reads_the_unrounded_cpi(ev, ac):
    r = calculate_evm(pv=ev, ev=ev, ac=ac)
    assert r["status"]["cpi_health"] == _band(ev / ac)


@pytest.mark.parametrize(("pv", "ev"), [(5_000.0, 0.0), (5_000.0, 4_000.0)])
def test_no_actual_cost_leaves_cpi_and_its_band_undefined(pv, ev):
    r = calculate_evm(bac=50_000.0, pv=pv, ev=ev, ac=0.0)
    assert r["CPI"] is None
    assert r["EAC"] is None
    assert r["status"]["cpi_health"] == "UNDEFINED"


@pytest.mark.parametrize(
    ("given", "absent"),
    [
        ({"pv": 5_000.0, "ev": 4_000.0}, "AC/ACWP"),
        ({"pv": 5_000.0, "ac": 4_500.0}, "EV/BCWP"),
        ({"ev": 4_000.0}, "PV/BCWS"),
    ],
)
def test_a_missing_measure_is_named_never_invented(given, absent):
    r = calculate_evm(bac=50_000.0, **given)
    assert "error" in r
    assert absent in r["error"]
    for key in ("CPI", "SPI", "CV", "SV", "EAC"):
        assert key not in r
