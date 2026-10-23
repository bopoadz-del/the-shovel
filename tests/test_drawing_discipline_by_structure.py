"""Discipline is read by structure, or named by the open project -- never by a code table.

The drawing reader once carried one project's drawing-number discipline codes
as a table in code. Codes mean something only inside one numbering scheme, so
the reader now:

  1. reads the title block's DISCIPLINE label (inline after a separator, or
     the cell to its right), as written;
  2. names a code through the OPEN PROJECT'S profile -- the project fact
     ``drawing_discipline_codes``, a JSON object {"<code>": "<discipline>"};
  3. otherwise reports the discipline as unknown.

Every sheet here uses an invented numbering scheme ("MVP.FR.ELX.0427": an
invented consultant, an invented site, an invented discipline code) and
invented place names in the drawing area, so nothing can pass by matching a
real project's convention.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import app.core.projects as projects_store
from app.blocks import drawing_qto as dq
from app.blocks.drawing_qto import DrawingQTOBlock
from tests.conftest import requires_construction_kit
from tests._dev_key import DEV_KEY

DN = "MVP.FR.ELX.0427"
TITLE = "GROUND FLOOR LIGHTING LAYOUT"
PLACES = ("KESTREL QUAY", "HOLLOWMERE CROSSING", "ASHVALE YARD")


def _sheet(path: Path, *, discipline_cell: str | None, inline: bool) -> None:
    """An A3 sheet: invented place labels in the drawing area, a labelled
    title block in the bottom band (at least five lines, the richness gate)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=1190.0, height=842.0)
    for i, place in enumerate(PLACES):
        page.insert_text(fitz.Point(120 + i * 260, 300), place, fontsize=18)
    page.draw_rect(fitz.Rect(100, 100, 500, 400))
    x = 700.0
    page.insert_text(fitz.Point(x, 730), "PROJECT: FENWICK REACH LIBRARY", fontsize=7)
    page.insert_text(fitz.Point(x, 745), f"DRAWING TITLE: {TITLE}", fontsize=9)
    page.insert_text(fitz.Point(x, 762), "CONSULTANT: MORROW VARGA PARTNERS", fontsize=7)
    page.insert_text(fitz.Point(x, 776), "SCALE 1:100", fontsize=7)
    page.insert_text(fitz.Point(x, 790), "DATE 04/11/25", fontsize=7)
    page.insert_text(fitz.Point(x, 818), f"DRAWING NO: {DN}", fontsize=7)
    if discipline_cell is not None:
        if inline:
            page.insert_text(fitz.Point(x, 804), f"DISCIPLINE: {discipline_cell}", fontsize=7)
        else:
            # Label cell, value in the cell to its right (no separator).
            page.insert_text(fitz.Point(x, 804), "DISCIPLINE", fontsize=7)
            page.insert_text(fitz.Point(x + 70, 804), discipline_cell, fontsize=7)
    doc.save(str(path))
    doc.close()


def _read(path: Path, project_id: str | None = None) -> dict:
    params = {"project_id": project_id} if project_id else {}
    r = asyncio.run(DrawingQTOBlock().process({"file_path": str(path)}, params))
    assert r["status"] == "success", r
    return r["drawing"]


@pytest.fixture
def profile(monkeypatch):
    """Stand-in for the open project's profile: {project_id: {key: value}}."""
    facts: dict[str, dict[str, str]] = {}

    def get_fact(project_id, key):
        v = facts.get(project_id, {}).get(key)
        return {"key": key, "value": v} if v is not None else None

    monkeypatch.setattr(projects_store, "get_fact", get_fact)
    return facts


def test_no_discipline_table_is_left_in_code():
    assert not hasattr(dq, "DISCIPLINE_FULL")
    assert not hasattr(DrawingQTOBlock, "_discipline_from_number")


def test_title_fields_are_read_by_label_on_an_invented_scheme(tmp_path, profile):
    p = tmp_path / "sheet.pdf"
    _sheet(p, discipline_cell=None, inline=True)
    d = _read(p)
    assert d["drawing_number"] == DN
    assert d["drawing_title"] == TITLE
    assert d["project_name"] == "FENWICK REACH LIBRARY"
    assert d["scale"] == "1:100"
    # No label and no project: unknown, not guessed from "ELX".
    assert d["discipline"] is None and d["discipline_full"] is None
    assert d["discipline_source"] == "unknown"
    # Place labels in the drawing area are never title-block values.
    for place in PLACES:
        assert place not in (d["drawing_title"] or "")
        assert place != d["discipline"]


@pytest.mark.parametrize("inline", [True, False])
def test_discipline_label_is_read_inline_or_from_the_next_cell(tmp_path, profile, inline):
    p = tmp_path / "sheet.pdf"
    _sheet(p, discipline_cell="LIGHTING", inline=inline)
    d = _read(p)
    assert d["discipline"] == "LIGHTING"
    assert d["discipline_full"] == "LIGHTING"
    assert d["discipline_source"] == "title_block"


def test_project_profile_names_a_drawing_number_code(tmp_path, profile):
    profile["proj-fr"] = {
        dq.DISCIPLINE_CODES_FACT: json.dumps({"ELX": "Electrical Lighting", "qsv": "Quantity Surveying"}),
    }
    p = tmp_path / "sheet.pdf"
    _sheet(p, discipline_cell=None, inline=True)
    d = _read(p, project_id="proj-fr")
    assert d["discipline"] == "ELX"
    assert d["discipline_full"] == "Electrical Lighting"
    assert d["discipline_source"] == "project_profile"
    # Another project's mapping does not leak into this one.
    d2 = _read(p, project_id="proj-other")
    assert d2["discipline"] is None and d2["discipline_source"] == "unknown"


def test_project_profile_names_a_labelled_code(tmp_path, profile):
    profile["proj-fr"] = {dq.DISCIPLINE_CODES_FACT: json.dumps({"ELX": "Electrical Lighting"})}
    p = tmp_path / "sheet.pdf"
    _sheet(p, discipline_cell="ELX", inline=False)
    d = _read(p, project_id="proj-fr")
    assert (d["discipline"], d["discipline_full"], d["discipline_source"]) == (
        "ELX", "Electrical Lighting", "title_block",
    )


@pytest.mark.parametrize("stored", ["not json", json.dumps(["ELX"]), json.dumps({"": "x"})])
def test_an_unusable_profile_value_means_unknown(tmp_path, profile, stored):
    profile["proj-fr"] = {dq.DISCIPLINE_CODES_FACT: stored}
    p = tmp_path / "sheet.pdf"
    _sheet(p, discipline_cell=None, inline=True)
    d = _read(p, project_id="proj-fr")
    assert d["discipline"] is None and d["discipline_source"] == "unknown"


@requires_construction_kit
def test_the_mapping_is_read_from_the_stored_project_profile(tmp_path):
    """End to end through the real project store and POST /v1/execute."""
    from fastapi.testclient import TestClient

    from app.main import app

    h = {"Authorization": "Bearer " + DEV_KEY}
    with TestClient(app) as c:
        pid = c.post("/v1/projects", json={"name": "Fenwick Reach Library"}, headers=h).json()["id"]
        projects_store.set_fact(pid, dq.DISCIPLINE_CODES_FACT,
                                json.dumps({"ELX": "Electrical Lighting"}))
        p = tmp_path / "sheet.pdf"
        _sheet(p, discipline_cell=None, inline=True)
        r = c.post("/v1/execute", json={
            "block": "drawing_qto", "input": {"file_path": str(p)},
            "params": {"project_id": pid},
        }, headers=h)
    assert r.status_code == 200, r.text
    d = r.json()["result"]["drawing"]
    assert (d["discipline"], d["discipline_full"], d["discipline_source"]) == (
        "ELX", "Electrical Lighting", "project_profile",
    )
