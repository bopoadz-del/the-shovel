"""W1 in the real React UI (stubbed model, fake embedder): chat sessions persist.

The sidebar lists SESSIONS above DOCUMENTS. A question asked, then a reload
or leaving the project and coming back, returns to the same session with its
history. "New chat" starts an empty one and the earlier session stays listed;
clicking it reopens its history. Delete asks first; declining keeps the
session, accepting removes it from the list.

Requires ``frontend/dist`` and Playwright browsers; regular CI ignores this
directory (``--ignore=tests/browser``).
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.browser.test_ui_phys_nightly import _ask, _create_project, _login

pytestmark = pytest.mark.browser

_QUESTIONS = [
    ("What is the curing time for the east deck pour?", "List the open crane lift permits."),
    ("Summarise the dewatering log for week 3.", "Which scaffold tags failed inspection?"),
]


def _user_texts(page: Page) -> list[str]:
    return [t.strip() for t in page.locator(".chat-bubble--user").all_inner_texts()]


def _session_rows(page: Page):
    return page.locator(".left-panel__history-item")


@pytest.mark.timeout(240)
@pytest.mark.parametrize("set_index", [0, 1])
def test_a_session_survives_reload_and_navigation(page: Page, live_app, set_index):
    base = live_app["base"]
    first_q, second_q = _QUESTIONS[set_index]
    _login(page, base)
    _create_project(page, f"Sessions Persist UI {set_index}")
    project_url = page.url

    heads = [h.strip() for h in page.locator(".left-panel__section-head").all_text_contents()]
    assert heads.index("Sessions") < heads.index("Documents"), heads

    _ask(page, first_q)
    assert first_q in " ".join(_user_texts(page))
    expect(_session_rows(page)).to_have_count(1, timeout=15_000)

    # Reload: the same session, with its history.
    page.reload(wait_until="networkidle")
    expect(page.locator(".chat-bubble--user")).to_have_count(1, timeout=15_000)
    assert first_q in " ".join(_user_texts(page))

    # Leave the project and come back: still the same session.
    page.goto(f"{base}/", wait_until="networkidle")
    page.goto(project_url, wait_until="networkidle")
    expect(page.locator(".chat-bubble--user")).to_have_count(1, timeout=15_000)

    # New chat: empty composer thread; the earlier session stays listed.
    page.get_by_role("button", name="New chat").click()
    expect(page.locator(".chat-bubble--user")).to_have_count(0)
    _ask(page, second_q)
    expect(_session_rows(page)).to_have_count(2, timeout=15_000)

    # Reopen the earlier session from the list.
    _session_rows(page).filter(has_text=first_q[:20]).first.click()
    expect(page.locator(".chat-bubble--user")).to_have_count(1, timeout=15_000)
    assert first_q in " ".join(_user_texts(page))
    assert second_q not in " ".join(_user_texts(page))

    # Delete asks first: declining keeps it.
    target = first_q[:20]
    delete_btn = page.get_by_role("button", name=f"Delete session {target}", exact=False)
    page.once("dialog", lambda d: d.dismiss())
    delete_btn.first.click()
    expect(_session_rows(page)).to_have_count(2)

    # Accepting removes it, and the open thread becomes a fresh one.
    page.once("dialog", lambda d: d.accept())
    delete_btn.first.click()
    expect(_session_rows(page)).to_have_count(1, timeout=15_000)
    expect(_session_rows(page).filter(has_text=target)).to_have_count(0)
    expect(page.locator(".chat-bubble--user")).to_have_count(0)
