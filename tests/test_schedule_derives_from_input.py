"""S-1 (PR#3, ORDERS_28OCT_0815Z): generate_wbs derives the schedule from the
input, never a template.

The LLM only STRUCTURES the brief/document text into activities (names +
stated durations/quantities/rates + stated dependencies, each with a verbatim
source_span); durations/sequence/CPM run in the deterministic construction
kernel (pm_computations / productivity_manpower_duration). These tests pin:
two different briefs -> two different, input-derived schedules; a span not in
the input is rejected; no scope -> no-scope (422 at the route); the output
declares its derivation source + kernel; and the 0845Z duration ruling
(stated / kernel-estimate-from-rate / UNKNOWN, never 0; kernel-detection).

complete_json (the single LLM egress) is monkeypatched so the tests are
deterministic and never hit a provider; the kernel is live.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.core import llm_client


def _patch_llm(monkeypatch, mapping):
    """Monkeypatch the single egress: return the structured activities for the
    brief whose lower-case text contains the key."""
    async def fake(system, user, **kwargs):
        low = (user or "").lower()
        for key, acts in mapping.items():
            if key in low:
                return {"activities": acts}
        return {"activities": []}
    monkeypatch.setattr(llm_client, "complete_json", fake)


async def _wbs(brief):
    return await ConstructionContainer().generate_wbs({"brief": brief}, {"target_count": 40})


@pytest.mark.asyncio
async def test_two_different_briefs_give_two_disjoint_input_derived_schedules(monkeypatch):
    _patch_llm(monkeypatch, {
        "tower": [
            {"name": "Excavate basement", "source_span": "excavate the basement", "duration_days": 10, "predecessors": []},
            {"name": "Pour raft", "source_span": "pour the raft", "duration_days": 6, "predecessors": ["Excavate basement"]},
        ],
        "road": [
            {"name": "Clear the route", "source_span": "clear the route", "duration_days": 4, "predecessors": []},
            {"name": "Lay subbase", "source_span": "lay the subbase", "duration_days": 8, "predecessors": ["Clear the route"]},
        ],
    })
    r1 = await _wbs("A tower: excavate the basement then pour the raft.")
    r2 = await _wbs("A road: clear the route then lay the subbase.")
    n1 = {a["name"] for a in r1["activities"]}
    n2 = {a["name"] for a in r2["activities"]}
    assert r1["status"] == "success" and r2["status"] == "success"
    assert n1 == {"Excavate basement", "Pour raft"}
    assert n2 == {"Clear the route", "Lay subbase"}
    assert n1.isdisjoint(n2)


@pytest.mark.asyncio
async def test_a_span_not_in_the_brief_is_rejected(monkeypatch):
    _patch_llm(monkeypatch, {"office": [
        {"name": "Real activity", "source_span": "build the office", "duration_days": 5, "predecessors": []},
        {"name": "Fabricated", "source_span": "this text is nowhere in the brief", "duration_days": 3, "predecessors": []},
    ]})
    r = await _wbs("Build the office.")
    names = {a["name"] for a in r["activities"]}
    assert "Real activity" in names
    assert "Fabricated" not in names, "a source_span not present in the brief must be dropped"


@pytest.mark.asyncio
async def test_empty_or_scopeless_brief_is_no_scope(monkeypatch):
    _patch_llm(monkeypatch, {})  # nothing structured
    r = await _wbs("")
    assert r["status"] == "error"
    assert not r["activities"]
    assert "no schedulable scope" in (r.get("error") or "").lower()


@pytest.mark.asyncio
async def test_output_declares_its_derivation_source_and_kernel(monkeypatch):
    _patch_llm(monkeypatch, {"warehouse": [
        {"name": "Erect steel", "source_span": "erect the steel", "duration_days": 12, "predecessors": []},
    ]})
    r = await _wbs("A warehouse: erect the steel frame.")
    sc = r["scaffold"]
    assert sc["source"] == "brief"
    assert sc["kernel"] == "pm_computations.compute_cpm"
    assert sc["derived_from_boq"] is False


@pytest.mark.asyncio
async def test_duration_ruling_stated_estimated_or_unknown(monkeypatch):
    _patch_llm(monkeypatch, {"fitout": [
        {"name": "Demolish", "source_span": "demolish", "duration_days": 7, "predecessors": []},
        {"name": "Plaster", "source_span": "plaster", "duration_days": None,
         "quantity": 100, "unit": "m2", "productivity_rate": "50 m2 per gang-day", "predecessors": ["Demolish"]},
        {"name": "Snag", "source_span": "snag", "duration_days": None, "predecessors": ["Plaster"]},
    ]})
    r = await _wbs("A fitout: demolish, plaster, snag.")
    by = {a["name"]: a for a in r["activities"]}
    assert by["Demolish"]["duration_days"] == 7 and by["Demolish"]["duration_basis"] == "stated"
    # quantity + stated rate -> kernel estimate, labelled; never 0.
    assert isinstance(by["Plaster"]["duration_days"], int) and by["Plaster"]["duration_days"] > 0
    assert by["Plaster"]["duration_basis"].startswith("estimated")
    # no rate, no stated duration -> UNKNOWN (None), never 0; chain incomplete.
    assert by["Snag"]["duration_days"] is None
    assert r["schedule_incomplete"] is True
    assert "Snag" in r["unscheduled_activities"]


@pytest.mark.asyncio
async def test_kernel_detection_two_quantities_two_durations(monkeypatch):
    """Same activity + rate, two quantities -> two kernel-computed durations
    (the duration comes from the kernel, not a fixed value)."""
    def brief_with(qty):
        _patch_llm(monkeypatch, {"plaster": [
            {"name": "Plaster", "source_span": "plaster", "duration_days": None,
             "quantity": qty, "unit": "m2", "productivity_rate": "50 m2 per gang-day", "predecessors": []},
        ]})
    brief_with(100)
    r1 = await _wbs("plaster 100 m2")
    brief_with(300)
    r2 = await _wbs("plaster 300 m2")
    d1 = r1["activities"][0]["duration_days"]
    d2 = r2["activities"][0]["duration_days"]
    assert d1 and d2 and d2 > d1, "a larger quantity must yield a larger kernel duration"


@pytest.mark.asyncio
async def test_typed_programme_becomes_rows_without_the_structurer(monkeypatch):
    """S-1 rework-1 #3: a programme the operator TYPED (named rows + date
    pairs) becomes the WBS rows directly — no LLM structuring call — with the
    duration taken from the typed dates and a 'typed_programme' source."""
    async def boom(*a, **k):
        raise AssertionError("a typed programme must not call the structurer")
    monkeypatch.setattr(llm_client, "complete_json", boom)
    r = await ConstructionContainer().generate_wbs(
        {"brief": "Excavation 05 Jan 2026 to 15 Jan 2026; "
                  "Foundations 16 Jan 2026 to 30 Jan 2026"},
        {"target_count": 40},
    )
    assert r["status"] == "success"
    names = {a["name"] for a in r["activities"]}
    assert "Excavation" in names and "Foundations" in names
    assert r["scaffold"]["source"] == "typed_programme"
    exc = next(a for a in r["activities"] if a["name"] == "Excavation")
    assert exc["duration_days"] == 10          # 05 -> 15 Jan
    assert "typed" in exc["duration_basis"]


@pytest.mark.asyncio
async def test_histogram_from_structured_schedule_needs_no_xer(monkeypatch):
    """S-1 rework-1 #3: a manpower/activity histogram is built from the
    STRUCTURED schedule (prose brief -> generate_wbs activities) with no .xer
    upload — the capability restored on the structured path."""
    _patch_llm(monkeypatch, {
        "job": [
            {"name": "Excavate", "source_span": "excavate", "duration_days": 5, "predecessors": []},
            {"name": "Pour", "source_span": "pour", "duration_days": 7, "predecessors": ["Excavate"]},
        ],
    })
    wbs = await ConstructionContainer().generate_wbs(
        {"brief": "The job: excavate then pour."}, {"target_count": 40},
    )
    assert wbs["status"] == "success"
    hist = await ConstructionContainer().resource_histogram(
        {"activities": wbs["activities"]}, {"period_unit": "week"},
    )
    assert hist["status"] == "success", hist
    assert hist.get("periods")


@pytest.mark.asyncio
async def test_two_briefs_give_two_different_derived_wbs_trees(monkeypatch):
    """ORDERS_28OCT_1415Z ruling (a): the brief path DERIVES its wbs_tree from
    its own structured activities (grouped by the trade each resolves to), so a
    brief WBS carries a real hierarchy and two different briefs give two
    different, non-empty trees — never a template. Each activity's stamped code
    is a key of its own tree (the chat-outline / xlsx-export contract)."""
    _patch_llm(monkeypatch, {
        "tower": [
            {"name": "Excavate basement", "source_span": "excavate the basement", "duration_days": 10, "predecessors": []},
            {"name": "Pour raft", "source_span": "pour the raft", "duration_days": 6, "predecessors": ["Excavate basement"]},
        ],
        "road": [
            {"name": "Clear the route", "source_span": "clear the route", "duration_days": 4, "predecessors": []},
            {"name": "Lay subbase", "source_span": "lay the subbase", "duration_days": 8, "predecessors": ["Clear the route"]},
        ],
    })
    r1 = await _wbs("A tower: excavate the basement then pour the raft.")
    r2 = await _wbs("A road: clear the route then lay the subbase.")
    t1, t2 = r1["wbs_tree"], r2["wbs_tree"]
    assert isinstance(t1, dict) and t1, "a brief WBS must carry a derived tree"
    assert isinstance(t2, dict) and t2
    assert t1 != t2, "two different briefs must give two different trees"
    for r in (r1, r2):
        tree = r["wbs_tree"]
        for a in r["activities"]:
            code = str(a.get("wbs_code") or a.get("wbs") or "")
            assert code in tree, f"activity code {code!r} is not a wbs_tree key"
