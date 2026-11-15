"""S1 in the real React UI (stubbed model, fake embedder): "Ask about this
document" from the sidebar list.

Two documents the test writes itself. A card's Ask action and a click on
its name each pin that document to the next question; the composer shows
the pin and can drop it; the send carries exactly the pinned id as
``document_ids`` and clears the pin. Each card shows a plain status line,
the x says Delete, and the chat pane states the server's totals.

Requires ``frontend/dist`` and Playwright browsers; regular CI ignores this
directory (``--ignore=tests/browser``).
"""
from __future__ import annotations

import json

import pytest
from playwright.sync_api import Page, expect

from tests.browser.test_ui_phys_nightly import _create_project, _login

pytestmark = pytest.mark.browser

_DOCS = {
    "slab_pour_sequence_notes.txt": "Pour bay 3 after bay 1 has cured for 48 hours.\n",
    "scaffold_tag_register.txt": "Tag 41 inspected and passed on day 2.\n",
}
_STATE_PREFIXES = ("Searchable", "Stored, not searchable", "Waiting to be read", "Failed")


def _upload_own(page: Page, tmp_path) -> dict[str, str]:
    """Upload the test's own two files; return name -> document id."""
    picker = page.locator("#doc-file-input")
    ids: dict[str, str] = {}
    for name, body in _DOCS.items():
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        picker.set_input_files(str(path))
        row = page.locator("li.doc-row").filter(
            has=page.locator(".doc-row__name", has_text=path.stem))
        expect(row).to_have_count(1, timeout=30_000)
        ids[name] = str(row.get_attribute("data-doc-id"))
    return ids


def _row(page: Page, name: str):
    stem = name.rsplit(".", 1)[0]
    return page.locator("li.doc-row").filter(
        has=page.locator(".doc-row__name", has_text=stem))


def _send_and_capture(page: Page, text: str) -> dict:
    box = page.locator(".chat-composer__textarea")
    box.fill(text)
    with page.expect_request(
        lambda r: r.url.endswith("/v1/chat/stream") and r.method == "POST"
    ) as req:
        page.locator(".chat-composer__send").click()
    body = json.loads(req.value.post_data or "{}")
    expect(box).to_be_enabled(timeout=90_000)
    return body


@pytest.mark.timeout(240)
def test_ask_about_a_document_from_the_sidebar(page: Page, live_app, tmp_path):
    base = live_app["base"]
    _login(page, base)
    _create_project(page, "Ask About Document UI")
    ids = _upload_own(page, tmp_path)
    first, second = list(_DOCS)

    # Each card: a plain status line, an Ask action, and a delete labelled so.
    for name in _DOCS:
        row = _row(page, name)
        line = row.locator(".doc-row__state")
        expect(line).to_be_visible()
        assert line.inner_text().startswith(_STATE_PREFIXES), line.inner_text()
        expect(row.locator(".doc-row__delete")).to_contain_text("Delete")
        expect(row.get_by_role("button", name=f"Ask about {name.rsplit('.', 1)[0]}",
                               exact=False).first).to_be_visible()

    # The chat pane states the server's totals, not the length of a page.
    expect(page.locator(".chat-list__empty-hint")).to_contain_text(
        "This project holds 2 documents")

    # The Ask action pins; the composer shows it; unpin drops it.
    _row(page, first).locator(".doc-row__ask").click()
    chips = page.locator("[data-testid=composer-pinned-doc]")
    expect(chips).to_have_count(1)
    expect(chips.first).to_contain_text(first.rsplit(".", 1)[0])
    expect(_row(page, first).locator(".doc-row__ask")).to_have_text("Pinned")
    chips.first.get_by_role("button").click()
    expect(chips).to_have_count(0)

    # Clicking the name pins too, and the send carries exactly that id.
    _row(page, second).locator(".doc-row__name").click()
    expect(chips).to_have_count(1)
    expect(chips.first).to_contain_text(second.rsplit(".", 1)[0])
    body = _send_and_capture(page, "What does this document say?")
    assert body.get("document_ids") == [ids[second]], body
    expect(chips).to_have_count(0)

    # A question with nothing pinned sends no document ids.
    body = _send_and_capture(page, "And in general?")
    assert not body.get("document_ids"), body
