"""forensic_delay_analysis must compute, not just return a shape.

Audit bar 3, 2026-08-12. Replacing the whole body with

    return {"status": "success", "delay_events": [], "critical_delays": [],
            "total_delay_days": 0}

left 16 tests green. Every reference to the action in tests/ was a string in an
alias list, an entry in the eval-battery harness, or a routing docstring; the
one test with "delay_analysis" in its name calls parse_primavera_schedule and
asserts `isinstance(..., list)` — its own name says "shape".

That matters more here than in most places. This is the EOT / prolongation
path: its output feeds a claim value and an expert report. A version of it that
silently returned zeros would have produced a confidently empty claim.

The anchor assertion is metamorphic rather than a golden number: analysing a
programme against ITSELF must yield zero net delay. That holds no matter which
baseline is used, so it does not rot when the fixture changes, and it cannot be
satisfied by a stub that returns an arbitrary constant — the stub also has to
classify the events it was handed and populate the report sections.
"""
from __future__ import annotations

import pytest

from tests._synthetic_fixtures import build_baseline_xer


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """A synthetic P6 baseline programme (``build_baseline_xer``)."""
    path = tmp_path_factory.mktemp("forensic_delay") / "sample_works_baseline.xer"
    build_baseline_xer(path)
    return path


def test_the_fixture_this_file_depends_on_is_present(baseline):
    """Without this, a broken generator would return forensic_delay_analysis
    to zero coverage SILENTLY. The programme is generated, never committed,
    so it must exist and carry a TASK table on every run, in every profile.
    """
    assert baseline.is_file(), f"{baseline} was not generated"
    tables = {
        line.split("\t", 1)[1]
        for line in baseline.read_text(encoding="cp1252").splitlines()
        if line.startswith("%T\t")
    }
    assert {"PROJECT", "TASK", "TASKPRED"} <= tables, tables


@pytest.fixture
def container():
    from app.blocks.primavera_parser import PrimaveraParserBlock
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    # primavera_parser registers via CEREBRUM_DOMAIN_KITS=construction; wire it
    # directly so this runs under the virgin profile too.
    c.wire("primavera_parser", PrimaveraParserBlock())
    return c


DELAY_EVENTS = [
    {"description": "late site access", "days": 14, "type": "employer",
     "start_date": "2025-03-10"},
    {"description": "exceptional weather", "days": 7, "type": "neutral",
     "start_date": "2025-04-15"},
]


@pytest.mark.asyncio
async def test_identical_baseline_and_as_built_yields_no_net_delay(container, baseline):
    """A programme compared with itself has not slipped. Anything else is wrong.

    This is the property a stub cannot fake in the right direction: it must
    parse BOTH files and difference them to arrive at zero honestly.
    """
    result = await container.forensic_delay_analysis(
        {"baseline_file": str(baseline), "updated_file": str(baseline),
         "delay_events": DELAY_EVENTS}, {})

    assert result["status"] == "success", result
    duration = result["project_duration"]
    assert duration["net_delay"] == 0, (
        f"a programme analysed against itself slipped by {duration['net_delay']} "
        "days — the differencing is wrong"
    )
    assert duration["baseline"] == duration["as_built"], duration
    assert result["critical_path_analysis"]["critical_path_unchanged"] is True


@pytest.mark.asyncio
async def test_every_supplied_delay_event_is_classified(container, baseline):
    """Events handed in must be accounted for, not dropped.

    A claim that silently loses an event understates entitlement, which is the
    expensive direction to be wrong in.
    """
    result = await container.forensic_delay_analysis(
        {"baseline_file": str(baseline), "updated_file": str(baseline),
         "delay_events": DELAY_EVENTS}, {})

    events = result["delay_events"]
    assert events["total_identified"] == len(DELAY_EVENTS), events
    # Each event lands in exactly one compensability bucket and one
    # excusability bucket — the totals must reconcile, not merely exist.
    assert events["compensable"] + events["non_compensable"] == len(DELAY_EVENTS), events
    assert events["excusable"] + events["non_excusable"] == len(DELAY_EVENTS), events


@pytest.mark.asyncio
async def test_expert_report_sections_are_produced(container, baseline):
    result = await container.forensic_delay_analysis(
        {"baseline_file": str(baseline), "updated_file": str(baseline),
         "delay_events": DELAY_EVENTS}, {})

    sections = result["expert_report_sections"]
    assert isinstance(sections, list) and len(sections) >= 5, sections
    joined = " ".join(sections).lower()
    assert "opinion" in joined or "summary" in joined, sections


@pytest.mark.asyncio
async def test_analysis_method_is_honoured(container, baseline):
    """`method` selects the analysis; a stub ignoring it would report the default."""
    result = await container.forensic_delay_analysis(
        {"baseline_file": str(baseline), "updated_file": str(baseline),
         "delay_events": DELAY_EVENTS}, {"method": "windows"})

    assert result["status"] == "success", result
    assert result["analysis_method"] == "windows", result["analysis_method"]


@pytest.mark.asyncio
async def test_refuses_honestly_without_both_programmes(container, baseline):
    """Delay analysis needs two schedules. One, or none, must be an explicit
    error — never a zeroed-out claim that reads like a finding of no delay."""
    result = await container.forensic_delay_analysis(
        {"baseline_file": str(baseline)}, {})

    assert result["status"] == "error"
    assert "baseline" in result["error"].lower() or "updated" in result["error"].lower()
