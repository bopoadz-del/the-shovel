"""A template WBS must declare itself at the glass.

THE INCIDENT (A-H gate battery, live ``13b2bf7``, 2026-08-31).

F1 asked for a WBS reflecting the actual bill structure. ``generate_wbs``
returned a template scaffold, and the answer showed only headline metrics --
204 activities, 688 working days, 44 critical -- with nothing saying the
structure was a standard template rather than this project's. F2 then went
further and attributed the scaffold to a BOQ that was never read.

Verified literally on that SHA, not inferred: ``generate_wbs``
(``app/containers/construction/schedule.py``) documents itself "Deterministic
template-based: no LLM", and its body contains no ``boq``/``bill``/
``quantit``/``contract`` token at all. It has no BOQ input to read.

THE RULING (owner, R3, 2026-09-01). The capability that exists is a template
scheduler, so that is what F1's ground truth becomes -- openly, with a dated
note. What the capability owes the reader is three things, and they are what
this file fences:

1. it DECLARES itself at the glass -- "template scaffold, project_type
   inferred: building, not derived from this project's BOQ";
2. it APPLIES overrides;
3. it never invents project-specific names -- "'Hall A/B/D' are fabrications
   in template clothing; placeholders must look like placeholders."

A BOQ-derived WBS is F1b, a separate capability, not this one.
"""

import inspect
import re

import pytest

from app.agents.runtime import _format_wbs_result
from app.containers.construction import ConstructionContainer


@pytest.fixture
def container():
    return ConstructionContainer()


async def _wbs(container, data=None, params=None):
    return await container.generate_wbs(data or {}, params or {"target_count": 60})


# --------------------------------------------------------------------------
# 3. Placeholders must look like placeholders
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("project_type", ["building", "infrastructure", "data_center"])
async def test_no_activity_is_named_after_an_invented_hall(container, project_type):
    """"Hall A" reads as a real, named part of the project. It never was one:
    "Hall" is a data-centre word that the generator applied to every project
    type, so a road package grew halls it does not have."""
    r = await _wbs(container, {}, {"project_type": project_type, "target_count": 200})
    offenders = [a["name"] for a in r["activities"] if "hall" in a["name"].lower()]
    assert offenders == [], offenders


@pytest.mark.asyncio
async def test_zone_labels_are_bracketed_ordinals(container):
    r = await _wbs(container, {}, {"project_type": "building", "target_count": 200})
    zoned = [a["name"] for a in r["activities"] if "Zone" in a["name"]]
    assert zoned, "expected the zone multiplier to fire at target_count=200"
    for name in zoned:
        assert re.search(r"\[Zone \d+\]$", name), name


@pytest.mark.asyncio
async def test_zone_labels_carry_no_letters_that_could_read_as_a_designation(container):
    """"Zone A" is still a designation a reader can mistake for the project's
    own. A bare ordinal in brackets cannot be."""
    r = await _wbs(container, {}, {"project_type": "building", "target_count": 200})
    for a in r["activities"]:
        assert not re.search(r"\b(?:Hall|Block|Building|Wing|Tower)\s+[A-Z]\b", a["name"]), a["name"]


def test_the_generator_no_longer_contains_the_hall_label_form():
    """Literal source check, so the label cannot come back by accident."""
    from app.containers.construction import schedule
    src = inspect.getsource(schedule)
    assert 'f"Hall ' not in src
    assert '[Zone {i + 1}]' in src


# --------------------------------------------------------------------------
# 1. It declares itself
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_carries_a_scaffold_record(container):
    r = await _wbs(container, {"brief": "office building, 12 storeys"})
    sc = r["scaffold"]
    assert sc["source"] == "template"
    assert sc["derived_from_boq"] is False
    assert sc["zone_labels"] == "placeholder"
    assert sc["declaration"]


@pytest.mark.asyncio
async def test_declaration_says_inferred_when_the_type_was_sniffed(container):
    r = await _wbs(container, {"brief": "Boulevard roads and utilities package"})
    assert r["scaffold"]["project_type_inferred"] is True
    assert "project_type inferred" in r["scaffold"]["declaration"]
    assert r["project_type"] in r["scaffold"]["declaration"]


@pytest.mark.asyncio
async def test_declaration_says_as_given_when_the_caller_stated_the_type(container):
    """Honest in both directions: a stated type is a fact, not a guess, and
    calling it inferred would be its own small fabrication."""
    r = await _wbs(container, {}, {"project_type": "building", "target_count": 40})
    assert r["scaffold"]["project_type_inferred"] is False
    assert "project_type as given: building" in r["scaffold"]["declaration"]
    assert "inferred" not in r["scaffold"]["declaration"]


@pytest.mark.asyncio
async def test_unknown_type_falls_back_to_building_and_says_so(container):
    """The fallback to `building` is itself an inference -- F39 made the
    fallback generic, R3 makes it visible."""
    r = await _wbs(container, {}, {"project_type": "submarine_base", "target_count": 40})
    assert r["project_type"] == "building"
    assert r["scaffold"]["project_type_inferred"] is True
    assert "project_type inferred: building" in r["scaffold"]["declaration"]


@pytest.mark.asyncio
async def test_declaration_denies_the_boq_in_words(container):
    """F2 attributed the scaffold to a bill. The declaration has to say the
    opposite in the answer's own words, not only in a structured field."""
    r = await _wbs(container, {"brief": "office building"})
    d = r["scaffold"]["declaration"].lower()
    assert "not derived from this project's boq" in d
    assert "placeholder" in d


@pytest.mark.asyncio
async def test_assumptions_name_the_zone_labels_as_placeholders(container):
    r = await _wbs(container, {"brief": "office building"})
    assert any("placeholder" in a.lower() for a in r["assumptions"])


# --------------------------------------------------------------------------
# ... at the GLASS, not only in the payload
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_declaration_is_rendered_before_any_number(container):
    """An assumptions list at the bottom is not a declaration. The reader must
    meet it before the activity count they might take for a project fact."""
    r = await _wbs(container, {"brief": "office building"})
    out = _format_wbs_result(r)
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[0] == "**Work Breakdown Structure**"
    assert "Template scaffold" in lines[1]
    decl_at = out.index("Template scaffold")
    count_at = out.index("Template activities:")
    assert decl_at < count_at


@pytest.mark.asyncio
async def test_glass_has_no_doubled_blank_lines(container):
    out = _format_wbs_result(await _wbs(container, {"brief": "office building"}))
    assert "\n\n\n" not in out


def test_formatter_without_a_scaffold_record_still_works():
    """Old payloads (and other callers) must not crash the glass."""
    out = _format_wbs_result({"actual_count": 12, "project_type": "building",
                              "summary": {"activity_count": 12}})
    assert "Work Breakdown Structure" in out
    assert "Template scaffold" not in out


# --------------------------------------------------------------------------
# 2. It applies overrides -- against the new placeholder names
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duration_override_still_matches_a_zoned_placeholder_row(container):
    """The override matcher used to be tuned against 'Site clearance - Hall
    A/B'. Renaming the zones must not break it."""
    r = await _wbs(container, {"brief": "building works"},
                   {"target_count": 60, "user_message": "use 6 days per tree removal and re-run"})
    applied = r.get("duration_overrides_applied") or []
    assert applied, r.get("duration_overrides_unmatched")
    assert applied[0]["days"] == 6
    assert applied[0]["activities_updated"] >= 1
    touched = {a["id"] for a in r["activities"] if a["duration_days"] == 6}
    assert touched, "override reported applied but no activity carries the new duration"


@pytest.mark.asyncio
async def test_override_reaches_every_zone_of_the_matched_activity(container):
    r = await _wbs(container, {"brief": "building works"},
                   {"target_count": 200, "user_message": "use 6 days per tree removal and re-run"})
    applied = (r.get("duration_overrides_applied") or [{}])[0]
    assert applied.get("activities_updated", 0) >= 2, applied


# --------------------------------------------------------------------------
# The capability boundary F1b exists to cross
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_passing_a_boq_changes_nothing_which_is_why_the_denial_is_true(container):
    """generate_wbs has no BOQ input. Handing it one must not quietly appear
    to work -- that would make the declaration a lie in the other direction.
    A BOQ-derived WBS is F1b, a different capability."""
    plain = await _wbs(container, {"brief": "office building"}, {"target_count": 60})
    with_boq = await _wbs(
        container,
        {"brief": "office building",
         "boq": [{"code": "D599.5", "description": "Hall A slab", "qty": 340904}]},
        {"target_count": 60},
    )
    assert [a["name"] for a in plain["activities"]] == [a["name"] for a in with_boq["activities"]]
    assert with_boq["scaffold"]["derived_from_boq"] is False
    assert not any("Hall A" in a["name"] for a in with_boq["activities"])
