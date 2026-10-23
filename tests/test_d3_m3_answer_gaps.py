"""Gaps left after the D3 look-ahead date and M3 contract-price fixes.

Synthetic contract only. Two percentages, two amounts:

* Milestone / section delay damages: 0.015% of the Contract Price.
* Whole of the Works: 0.1% per calendar day.
* Clause 1.1.1 Contract Price: SAR 8,000,000.00 excluding VAT.
* Another document's Accepted Contract Amount: SAR 3,000,000.00.

A section or Milestone ask must not keep a "whole of the Works" lead,
must not quote that phrase back, and a day-count must not multiply by
the other document's amount. A stated today beats a model as_of that
is only the server clock.
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import date
from pathlib import Path

import pytest

import app.containers.construction.schedule as schedule_mod
import app.lib.pm_computations as pm
from app.agents.runtime import _graft_composed_delay_damages_daily
from app.lib import construction_formulas_commercial as cc

SECTION_ASK = "What are the delay damages in SAR per calendar day for a section?"
MILESTONE_ASK = (
    "What are the delay damages in SAR per calendar day for a Milestone?"
)
PERIOD_ASK = "If a section is 10 days late, what are the delay damages?"
CONTRACT_PRICE = 8_000_000.00
OTHER_AMOUNT = 3_000_000.00
DAILY = 1_200.00  # 0.015% of 8,000,000
PERIOD = 12_000.00  # 0.015% × 8,000,000 × 10

MILESTONE_CLAUSE = (
    "8.8.1 Delay Damages 0.015% of the Contract Price per calendar day "
    "per Milestone"
)
WHOLE_CLAUSE = (
    "Delay Damages for the whole of the Works are 0.1% of the Contract "
    "Price per calendar day"
)
PRICE_ROW = (
    "1.1.1 Accepted Contract Amount SAR 8,000,000.00 excluding VAT"
)
LIVE_BUNDLE = (
    "[doc_id=cover-note chunk=0] Kickoff note | "
    f"Accepted Contract Amount excluding VAT SAR {OTHER_AMOUNT:,.2f} | "
    "[doc_id=conditions chunk=2] CONTRACT DATA "
    f"{WHOLE_CLAUSE} and {MILESTONE_CLAUSE} {PRICE_ROW}"
)
# No row separator between the two particulars, so a 96-character
# window around 0.015% still contains the whole-of-Works wording.
BLENDED = (
    "[doc_id=conditions chunk=2] CONTRACT DATA "
    f"{WHOLE_CLAUSE} and {MILESTONE_CLAUSE}. {PRICE_ROW}"
)

CONTAINER = next(
    obj for obj in vars(schedule_mod).values()
    if inspect.isclass(obj) and hasattr(obj, "look_ahead")
)
STATED = (
    "Today is 21 September 2026. Give me a two-week look-ahead from these "
    "activities: Blinding 15 Sep to 25 Sep 2026; Rebar fixing 28 Sep to "
    "2 Oct 2026."
)


def _freeze(monkeypatch, day: date) -> None:
    monkeypatch.setattr(pm, "_clock_today", lambda: day)


def _look_ahead(params):
    return asyncio.run(CONTAINER().look_ahead({}, params))


def _minimal_xer(path: Path) -> Path:
    path.write_text(
        "\n".join([
            "%T\tPROJECT",
            "%F\tproj_id\tproj_short_name\tplan_start_date\tlast_recalc_date",
            "%R\t1\tFixture\t2026-03-01 00:00\t2026-03-08 00:00",
            "%T\tTASK",
            "%F\ttask_id\ttask_code\ttask_name\ttarget_drtn_hr_cnt"
            "\tearly_start_date\tearly_end_date\ttotal_float_hr_cnt"
            "\tremain_drtn_hr_cnt\twbs_id\tstatus_code\tphys_complete_pct",
            "%R\t1001\tA\tMobilise\t40\t2026-03-01 00:00\t2026-03-05 00:00"
            "\t80\t0\tW1\tTK_Complete\t100",
            "%E",
            "",
        ]),
        encoding="cp1252",
    )
    return path


@pytest.mark.parametrize("ask", [SECTION_ASK, MILESTONE_ASK])
def test_whole_of_the_works_lead_does_not_survive_a_matching_figure(ask):
    """Gap (a): the right daily figure must not keep the wrong label."""
    wrong = (
        f"Delay damages for the whole of the Works are SAR {DAILY:,.2f} "
        "per calendar day."
    )
    out = _graft_composed_delay_damages_daily(
        wrong, {"role": "system", "content": LIVE_BUNDLE},
        [{"role": "user", "content": ask}],
    )
    assert "whole of the Works" not in out.split("\n", 1)[0]
    assert "per Milestone" in out
    assert f"{DAILY:,.2f}" in out


def test_a_milestone_clause_quote_does_not_carry_whole_of_the_works():
    """Gap (b): the appended Contract Data quote must not wear the other label."""
    out = cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, BLENDED)
    assert out is not None
    assert out["basis"] == "milestone"
    line = cc.format_delay_damages_daily_line(out)
    assert "per Milestone" in line
    assert "whole of the Works" not in line
    assert "0.015%" in line


def test_a_section_day_count_uses_the_rate_documents_contract_price():
    """Gap (c): do not multiply by another document's Accepted Contract Amount."""
    out = cc.compose_delay_damages_over_period_from_excerpts(PERIOD_ASK, LIVE_BUNDLE)
    assert out is not None
    assert out["rates"] == [pytest.approx(0.015)]
    assert out["contract_amount"] == pytest.approx(CONTRACT_PRICE)
    assert out["amount"] == pytest.approx(PERIOD)
    assert OTHER_AMOUNT not in (out["contract_amount"],)


def test_stated_today_beats_a_model_as_of_equal_to_the_clock(monkeypatch):
    """Gap (d): as_of that equals the frozen clock is not an explicit date."""
    _freeze(monkeypatch, date(2026, 9, 27))
    out = _look_ahead({"user_message": STATED, "as_of": "2026-09-27"})
    assert out["status"] == "success"
    assert out["as_of"] == "2026-09-21"


def test_xer_stated_today_beats_a_model_as_of_equal_to_the_clock(monkeypatch, tmp_path):
    """Gap (d), the .xer as_of block, same rule as the inline window.

    The parser is stubbed so this asserts the date choice in that block,
    not Primavera availability.
    """
    _freeze(monkeypatch, date(2026, 9, 27))

    async def _parsed(self, path):
        return {
            "status": "success",
            "data_date": "2026-03-08",
            "activities": [{
                "id": "A",
                "name": "Mobilise",
                "start": "2026-03-01",
                "finish": "2026-03-05",
            }],
        }

    monkeypatch.setattr(CONTAINER, "_parse_xer_file", _parsed)
    path = _minimal_xer(tmp_path / "fixture.xer")
    out = _look_ahead({
        "schedule_file": str(path),
        "user_message": "Today is 21 September 2026. Give me a two-week look-ahead.",
        "as_of": "2026-09-27",
        "days": 14,
    })
    assert out["status"] == "success", out
    assert out["source"] == "primavera_xer_dates"
    assert out["as_of"] == "2026-09-21"


def test_a_genuine_as_of_that_is_not_the_clock_still_wins(monkeypatch):
    """An as_of the operator or tool set, different from the clock, stays."""
    _freeze(monkeypatch, date(2026, 9, 27))
    out = _look_ahead({"user_message": STATED, "as_of": "2026-08-01"})
    assert out["as_of"] == "2026-08-01"


def test_xer_genuine_as_of_that_is_not_the_clock_still_wins(monkeypatch, tmp_path):
    """The .xer block keeps a real as_of that is not the server clock."""
    _freeze(monkeypatch, date(2026, 9, 27))

    async def _parsed(self, path):
        return {
            "status": "success",
            "data_date": "2026-03-08",
            "activities": [{
                "id": "A",
                "name": "Mobilise",
                "start": "2026-03-01",
                "finish": "2026-03-05",
            }],
        }

    monkeypatch.setattr(CONTAINER, "_parse_xer_file", _parsed)
    path = _minimal_xer(tmp_path / "fixture.xer")
    out = _look_ahead({
        "schedule_file": str(path),
        "user_message": "Today is 21 September 2026. Give me a two-week look-ahead.",
        "as_of": "2026-08-01",
        "days": 14,
    })
    assert out["status"] == "success", out
    assert out["as_of"] == "2026-08-01"


def test_generate_wbs_uses_a_stated_today_when_start_date_is_absent(monkeypatch):
    """Gap (e): no start_date must not fall through to the server clock."""
    from app.containers.construction import ConstructionContainer

    _freeze(monkeypatch, date(2026, 9, 27))
    out = asyncio.run(ConstructionContainer().generate_wbs({}, {
        "brief": "Small office fit-out.",
        "user_message": "Today is 21 September 2026. Build a WBS for this office.",
        "target_count": 20,
    }))
    assert out["status"] == "success", out
    assert out["start_date"] == "2026-09-21"


def test_generate_wbs_keeps_an_explicit_start_date():
    """A caller-supplied start_date is not replaced by a stated today."""
    from app.containers.construction import ConstructionContainer

    out = asyncio.run(ConstructionContainer().generate_wbs({}, {
        "brief": "Small office fit-out.",
        "user_message": "Today is 21 September 2026. Build a WBS for this office.",
        "start_date": "2026-01-15",
        "target_count": 20,
    }))
    assert out["status"] == "success", out
    assert out["start_date"] == "2026-01-15"
