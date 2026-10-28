"""S-1b (ruling ORDERS_28OCT_1045Z): the productivity-norm table is sourced
kernel DATA. A row stays only with a REAL citation + unit; no verifiable
published figures are shipped in code, so the table is EMPTY and the schedule
path yields UNKNOWN until sourced rows exist.

These pin the ruling and the mechanism (the mechanism is exercised with an
injected, clearly-labelled TEST row — never a shipped one):
  * every shipped row (if any) is cited + united; the table may be empty;
  * no matching cited row -> norm_rate_for None -> duration UNKNOWN;
  * an uncited row never applies even if present;
  * a cited, unit-matching row -> kernel duration; two quantities -> two
    durations; mutating the rate moves the duration (kernel-detection);
  * a unit mismatch -> UNKNOWN; a stated rate wins; word-boundary matching.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.core import llm_client
from app.lib import productivity_norms
from app.lib.productivity_norms import norm_rate_for

# A cited, united row injected ONLY to exercise the mechanism. It is explicitly
# marked not-shipped; it never lives in app/lib/productivity_norms.py._NORMS.
_TEST_ROW = {
    "keywords": ["excavation", "excavate"],
    "rate": "200 m3 per gang-day",
    "unit": "m3",
    "citation": "TEST ONLY — Example Norms, 1st ed., tbl 1 (not shipped)",
}


def _patch_llm(monkeypatch, mapping):
    async def fake(system, user, **kwargs):
        low = (user or "").lower()
        for key, acts in mapping.items():
            if key in low:
                return {"activities": acts}
        return {"activities": []}
    monkeypatch.setattr(llm_client, "complete_json", fake)


async def _wbs(brief):
    return await ConstructionContainer().generate_wbs({"brief": brief}, {"target_count": 40})


def _act(result, name):
    for a in result.get("activities") or []:
        if isinstance(a, dict) and a.get("name") == name:
            return a
    return None


# ---- the ruling -----------------------------------------------------------

def test_every_shipped_row_is_cited_and_united():
    for row in productivity_norms._NORMS:
        assert str(row.get("citation") or "").strip(), "a shipped norm row must cite a real source"
        assert str(row.get("unit") or "").strip(), "a shipped norm row must carry its unit"


def test_empty_shipped_table_resolves_to_none():
    assert norm_rate_for("Bulk excavation", "m3") is None
    assert norm_rate_for("General site clearance", "m2") is None
    assert norm_rate_for("") is None


def test_uncited_row_never_applies(monkeypatch):
    monkeypatch.setattr(productivity_norms, "_NORMS", [
        {"keywords": ["excavation"], "rate": "200 m3 per gang-day", "unit": "m3", "citation": ""},
    ])
    assert norm_rate_for("excavation", "m3") is None


# ---- the mechanism, exercised with an injected cited row ------------------

def test_cited_row_matches_word_boundary_and_unit(monkeypatch):
    monkeypatch.setattr(productivity_norms, "_NORMS", [_TEST_ROW])
    got = norm_rate_for("Bulk excavation to reduced level", "m3")
    assert got and got["rate"] == "200 m3 per gang-day" and got["unit"] == "m3"
    assert "TEST ONLY" in got["source"]
    assert norm_rate_for("Bulk excavation", "m2") is None          # unit mismatch
    assert norm_rate_for("preexcavateway survey", "m3") is None    # word boundary


@pytest.mark.asyncio
async def test_no_matching_norm_quantity_without_rate_is_unknown(monkeypatch):
    _patch_llm(monkeypatch, {
        "job": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 400, "unit": "m3"}],
    })
    r = await _wbs("The job: 400 m3 of excavation.")  # empty shipped table
    a = _act(r, "Bulk excavation")
    assert a is not None and a["duration_days"] is None
    assert r.get("schedule_incomplete") is True


@pytest.mark.asyncio
async def test_injected_cited_norm_two_quantities_two_durations(monkeypatch):
    monkeypatch.setattr(productivity_norms, "_NORMS", [_TEST_ROW])
    _patch_llm(monkeypatch, {
        "small": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 200, "unit": "m3"}],
        "large": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 800, "unit": "m3"}],
    })
    small = await _wbs("A small job: 200 m3 of excavation.")
    large = await _wbs("A large job: 800 m3 of excavation.")
    assert _act(small, "Bulk excavation")["duration_days"] == 1   # 200/200
    assert _act(large, "Bulk excavation")["duration_days"] == 4   # 800/200


@pytest.mark.asyncio
async def test_mutating_the_cited_norm_moves_the_duration(monkeypatch):
    _patch_llm(monkeypatch, {
        "job": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 400, "unit": "m3"}],
    })
    monkeypatch.setattr(productivity_norms, "_NORMS", [{**_TEST_ROW, "rate": "100 m3 per gang-day"}])
    slow = await _wbs("The job: 400 m3 of excavation.")
    monkeypatch.setattr(productivity_norms, "_NORMS", [{**_TEST_ROW, "rate": "200 m3 per gang-day"}])
    fast = await _wbs("The job: 400 m3 of excavation.")
    assert _act(slow, "Bulk excavation")["duration_days"] == 4    # 400/100
    assert _act(fast, "Bulk excavation")["duration_days"] == 2    # 400/200


@pytest.mark.asyncio
async def test_unit_mismatch_is_unknown(monkeypatch):
    monkeypatch.setattr(productivity_norms, "_NORMS", [_TEST_ROW])  # unit m3
    _patch_llm(monkeypatch, {
        "job": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 100, "unit": "m2"}],
    })
    r = await _wbs("The job: 100 m2 of excavation.")
    assert _act(r, "Bulk excavation")["duration_days"] is None


@pytest.mark.asyncio
async def test_stated_rate_wins_over_injected_norm(monkeypatch):
    monkeypatch.setattr(productivity_norms, "_NORMS", [_TEST_ROW])  # 200 m3/gd
    _patch_llm(monkeypatch, {
        "job": [{
            "name": "Bulk excavation", "source_span": "excavation",
            "quantity": 200, "unit": "m3", "productivity_rate": "100 m3 per gang-day",
        }],
    })
    r = await _wbs("The job: 200 m3 of excavation at 100 m3 per gang-day.")
    a = _act(r, "Bulk excavation")
    assert a["duration_days"] == 2                       # stated 100/gd, not the norm's 1
    assert "basis" in a["duration_basis"] and "norm" not in a["duration_basis"]
