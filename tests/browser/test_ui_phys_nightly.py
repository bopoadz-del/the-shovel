"""Nightly UI regression for the sanitized UI-PHYS question set.

Drives the real React UI: login, upload fixture docs, pin hats, ask
A1 / B1 / E4 / G1 / G5, export H1. Assertions are the catalog's
``must`` / ``must_any`` / footer rules — same shapes as the confidential
Drive pack, fixture figures only.

Requires ``frontend/dist`` and Playwright browsers. Regular CI ignores
this directory (``--ignore=tests/browser``).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ui_phys"

pytestmark = pytest.mark.browser


def _register(page: Page, base: str) -> None:
    page.goto(f"{base}/login", wait_until="networkidle")
    expect(page.locator("#email")).to_be_visible(timeout=15_000)
    page.get_by_role("button", name="Create one").click()
    expect(page.get_by_role("heading", name="Create account")).to_be_visible()
    page.locator("#email").fill(f"ui-phys-nightly-{int(time.time())}@example.com")
    page.locator("#password").fill("UiPhysNightly9!")
    page.locator("#displayName").fill("UI PHYS Nightly")
    page.get_by_role("button", name="Create account").click()
    # Dev auto-verifies and navigates home. If the form stays put, surface it.
    try:
        expect(page.get_by_role("heading", name="Projects")).to_be_visible(timeout=20_000)
    except AssertionError:
        notice = page.locator(".auth-error, .auth-notice").inner_text() if page.locator(".auth-error, .auth-notice").count() else ""
        raise AssertionError(
            f"register did not reach Projects; url={page.url} notice={notice!r} "
            f"body={page.locator('body').inner_text()[:500]!r}"
        ) from None


def _create_project(page: Page, name: str) -> None:
    page.get_by_role("button", name="+ New project").click()
    expect(page.locator("#proj-name")).to_be_visible()
    page.locator("#proj-name").fill(name)
    page.locator("#proj-client").fill("North Spur Demo")
    page.get_by_role("button", name="Create project").click()
    expect(page.locator("#proj-name")).to_have_count(0, timeout=15_000)
    page.get_by_role("link", name=f"Open project {name}").click()
    page.wait_for_url("**/projects/**", timeout=20_000)


def _upload(page: Page, *rel_names: str) -> None:
    picker = page.locator("#doc-file-input")
    for name in rel_names:
        picker.set_input_files(str(FIXTURE_DIR / name))
        expect(page.locator(".doc-row__name").filter(has_text=name)).to_be_visible(
            timeout=30_000
        )
    _wait_docs_indexed(page, expected=len(rel_names))


def _wait_docs_indexed(page: Page, expected: int, timeout_ms: int = 20_000) -> None:
    """Poll GET /v1/projects/{id} until eager-index has chunked every upload."""
    page.wait_for_function(
        """async (expected) => {
          const token = localStorage.getItem('tf_token');
          const m = location.pathname.match(/\\/projects\\/([^/]+)/);
          if (!token || !m) return false;
          const r = await fetch('/v1/projects/' + m[1], {
            headers: { Authorization: 'Bearer ' + token },
          });
          if (!r.ok) return false;
          const proj = await r.json();
          const docs = proj.documents || [];
          const ready = docs.filter((d) => (d.chunk_count || 0) > 0);
          return ready.length >= expected;
        }""",
        arg=expected,
        timeout=timeout_ms,
    )


def _pin(page: Page, agent: str) -> None:
    box = page.locator(".chat-composer__textarea")
    box.click()
    box.fill(f"/{agent}")
    item = page.locator(".chat-composer__slash-item").filter(has_text=agent)
    expect(item.first).to_be_visible(timeout=10_000)
    item.first.click()
    expect(page.locator(".chat-composer__pin")).to_contain_text(agent)


def _ask(page: Page, text: str, timeout: float = 90_000) -> None:
    before = page.locator(".chat-bubble--assistant").count()
    box = page.locator(".chat-composer__textarea")
    box.fill(text)
    expect(box).to_have_value(text)
    page.locator(".chat-composer__send").click()
    # Stubbed replies can finish before the typing dots are observed.
    # Wait for a new assistant bubble (or an error), then for send to re-enable.
    expect(page.locator(".chat-bubble--assistant, .chat-bubble--error")).to_have_count(
        before + 1, timeout=int(timeout)
    )
    expect(box).to_be_enabled(timeout=timeout)
    if page.locator(".chat-bubble--error").count():
        raise AssertionError(
            f"assistant error bubble: {page.locator('.chat-bubble--error').last.inner_text()!r}"
        )


def _last_answer(page: Page) -> str:
    return page.locator(".chat-bubble--assistant").last.inner_text()


def _assert_case(page: Page, case: dict) -> None:
    body = _last_answer(page)
    for token in case.get("must") or []:
        assert token in body, f"missing {token!r} in:\n{body[:800]}"
    if case.get("must_any"):
        assert any(tok.lower() in body.lower() for tok in case["must_any"]), (
            f"none of {case['must_any']} in:\n{body[:800]}"
        )
    for banned in case.get("must_not") or []:
        assert banned.lower() not in body.lower(), f"invented {banned!r} in:\n{body[:800]}"
    # Answer rendered in the UI, not an empty/error bubble.
    expect(page.locator(".chat-bubble--error")).to_have_count(0)


@pytest.mark.timeout(240)
def test_ui_phys_nightly_login_upload_ask_export(page: Page, live_app, ui_phys_catalog):
    base = live_app["base"]
    cases = ui_phys_catalog["cases"]

    _register(page, base)
    expect(page.get_by_role("heading", name="Projects")).to_be_visible(timeout=15_000)

    _create_project(page, ui_phys_catalog["project_name"])
    expect(page.get_by_text("UI-PHYS Fixture").first).to_be_visible(timeout=15_000)

    _upload(page, "S1_contract_data.md", "S2_demolition_boq.md")

    # A1 — project-assistant contract QA
    _pin(page, cases["A1"]["agent"])
    _ask(page, cases["A1"]["ask"])
    _assert_case(page, cases["A1"])
    expect(page.locator(".sources-list")).to_contain_text("S1_contract_data", timeout=10_000)

    # B1 — quantity-surveyor BOQ numeric QA
    _pin(page, cases["B1"]["agent"])
    _ask(page, cases["B1"]["ask"])
    _assert_case(page, cases["B1"])

    # E4 — construction-pm named calculator (waste factor 1.05)
    _pin(page, cases["E4"]["agent"])
    _ask(page, cases["E4"]["ask"])
    _assert_case(page, cases["E4"])

    # G1 — honest refusal: Schedule 10 is Not Used
    _pin(page, cases["G1"]["agent"])
    _ask(page, cases["G1"]["ask"])
    _assert_case(page, cases["G1"])

    # G5 — honest refusal: drawing not in the corpus
    _pin(page, cases["G5"]["agent"])
    _ask(page, cases["G5"]["ask"])
    _assert_case(page, cases["G5"])

    # H1 — conversation export is last-message-only ("Excerpt"). Footer
    # checks run on the left-panel export; the ACA figure lives on A1, so
    # we also download that bubble's Word file.
    from docx import Document

    with page.expect_download() as pending:
        page.get_by_title("Export this conversation as a .docx file").click()
    convo = pending.value
    assert convo.suggested_filename.endswith(".docx")
    convo_path = Path(live_app["data_dir"]) / "h1_conversation.docx"
    convo.save_as(str(convo_path))
    convo_text = "\n".join(p.text for p in Document(str(convo_path)).paragraphs)
    _env_banned = [
        t.strip()
        for t in __import__("os").getenv("UI_PHYS_FORBIDDEN", "").splitlines()
        if t.strip()
    ]
    for banned in list(cases["H1"]["footer_must_not"]) + _env_banned:
        assert banned not in convo_text, f"export footer leaked {banned!r}"
    assert "theshovel.ai" in convo_text.lower() or "Generated by The Shovel" in convo_text

    with page.expect_download() as pending:
        page.locator(".chat-bubble--assistant").first.get_by_label(
            "Download as Word document"
        ).click()
    a1 = pending.value
    assert a1.suggested_filename.endswith(".docx")
    a1_path = Path(live_app["data_dir"]) / "h1_a1.docx"
    a1.save_as(str(a1_path))
    a1_text = "\n".join(p.text for p in Document(str(a1_path)).paragraphs)
    for token in cases["H1"].get("must_any") or []:
        assert token in a1_text, f"A1 export missing {token!r} in:\n{a1_text[:800]}"
    for banned in list(cases["H1"]["footer_must_not"]) + _env_banned:
        assert banned not in a1_text, f"A1 export footer leaked {banned!r}"
