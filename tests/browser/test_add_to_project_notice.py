""""+ Add to project" in the real React UI (stubbed model, fake embedder),
signed in as the admin account the fixture seeds.

An admin's project upload is accepted and its indexing is scheduled, so the
workspace must show the new row and NO alert: the server's own card state for
that row is "waiting to be read" or "searchable", never "stored, not added".
A rejected upload must still raise the alert, so the alert check below is not
vacuous. The label's title promises no search.

Requires ``frontend/dist`` and Playwright browsers; regular CI ignores this
directory (``--ignore=tests/browser``). Runs in browser-nightly.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.browser.test_ui_phys_nightly import _create_project, _login

pytestmark = pytest.mark.browser

# Two accepted uploads per run, different names, types and bodies, so a
# constant outcome cannot pass both.
_ACCEPTED = [
    {
        "kerb_line_setting_out.txt": "Kerb line set out from station 0+120 to 0+180.\n",
        "pump_station_handover.md": "# Handover\n\nPump 2 commissioned on day 9.\n",
    },
    {
        "rebar_cover_checks.md": "## Cover\n\nSlab cover measured 40 mm at grid C.\n",
        "crane_lift_plan_notes.txt": "Lift 7 uses the 50 t crane at radius 18 m.\n",
    },
]
_STATE_PREFIXES = ("Searchable", "Waiting to be read")


def _label(page: Page):
    return page.locator("label[for=doc-project-file-input]")


def _add(page: Page, path) -> None:
    page.locator("#doc-project-file-input").set_input_files(str(path))
    # The control comes back from its busy state when the request has ended.
    expect(_label(page)).to_have_attribute("aria-busy", "false", timeout=30_000)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("set_index", [0, 1])
def test_admin_add_to_project_shows_no_false_alert(page: Page, live_app, tmp_path, set_index):
    base = live_app["base"]
    _login(page, base)
    _create_project(page, f"Add To Project Notice {set_index}")

    label = _label(page)
    expect(label).to_have_text("+ Add to project")
    assert "search" not in (label.get_attribute("title") or "").lower()

    alert = page.locator(".docs-upload__error[role=alert]")
    for name, body in _ACCEPTED[set_index].items():
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        _add(page, path)
        stem = path.stem
        row = page.locator("li.doc-row").filter(
            has=page.locator(".doc-row__name", has_text=stem))
        expect(row).to_have_count(1, timeout=30_000)
        state = row.locator(".doc-row__state").inner_text()
        assert state.startswith(_STATE_PREFIXES), state
        # The admin's accepted upload: no alert at all, so in particular not
        # the "not added to the project knowledge base" sentence.
        expect(alert).to_have_count(0)

    # Must fire: a file the server refuses still raises the alert, and adds
    # no row. Two refused shapes, each a compressed container by its bytes.
    for refused in (f"site_photos_{set_index}.zip", f"drawings_{set_index}.7z"):
        bad = tmp_path / refused
        head = b"PK\x03\x04" if refused.endswith(".zip") else b"7z\xbc\xaf\x27\x1c"
        bad.write_bytes(head + b"\x00" * 64)
        _add(page, bad)
        expect(alert).to_have_count(1)
        expect(alert).to_contain_text("not supported")
        expect(page.locator("li.doc-row").filter(
            has=page.locator(".doc-row__name", has_text=bad.stem))).to_have_count(0)

    # The next accepted upload clears the alert again.
    again = tmp_path / f"after_refusal_{set_index}.txt"
    again.write_text("Joint 4 sealed on day 11.\n", encoding="utf-8")
    _add(page, again)
    expect(page.locator("li.doc-row").filter(
        has=page.locator(".doc-row__name", has_text=again.stem))).to_have_count(1, timeout=30_000)
    expect(alert).to_have_count(0)
