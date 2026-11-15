"""EXAM 11: the chat workspace, as a person sees it (SPEC_EXAM_PLAN_15NOV).

Real React UI against the real backend, model stubbed (``_llm_stub``), fake
embedder. Runs in the browser nightly (regular CI ignores tests/browser).
The checkers are pure functions in ``tests/exam11_checks.py``; each one is
proven to pass a good input and fail a bad one by
``tests/test_exam11_checks.py`` (rule 3).

Items and the build step that must make them pass:
  11.1 a session survives leaving the project and a reload ........ W1
  11.2 sessions are listed above documents ......................... W1
  11.3 each session has its own documents .......................... W2
  11.4 a produced file is saved in the session and tagged Produced . W2
  11.5 citations only in the side panel, never in the answer text .. W3
  11.6 no process text in the answer ............................... W4
  11.7 the activity line matches the backend events of the turn .... W4

Items whose build step has not landed are ``xfail(strict=True)``: they are
written first and fail first; the PR that lands the step must turn them
green, and strict mode makes the nightly go red if one passes while still
marked, so the marker cannot outlive the fix. Every assertion message
starts with the STAGE that failed (rule 5).
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.browser.test_ui_phys_nightly import (
    FIXTURE_DIR,
    _ask,
    _create_project,
    _login,
)
from tests.exam11_checks import (
    activity_mismatch,
    citation_leaks,
    process_talk,
    ran_labels,
    sse_events,
)

pytestmark = pytest.mark.browser

_W2 = pytest.mark.xfail(strict=True, reason="W2 not landed: per-session documents")
_W3 = pytest.mark.xfail(strict=True, reason="W3 not landed: citations in the side panel only")

# Two invented questions per item; the stub answers any text.
_PAIRS = [
    ("What is the curing time for the east deck pour?", "List the open crane lift permits."),
    ("Summarise the dewatering log for week 3.", "Which scaffold tags failed inspection?"),
]

# Records every status line the bubble ever shows, and every SSE body.
_STATUS_RECORDER = """
window.__exam11Status = [];
new MutationObserver(() => {
  document.querySelectorAll('.chat-bubble__tool-status').forEach((el) => {
    const t = (el.innerText || '').trim();
    const seen = window.__exam11Status;
    if (t && seen[seen.length - 1] !== t) seen.push(t);
  });
}).observe(document, { subtree: true, childList: true, characterData: true });
"""


def _heads(page: Page) -> list[str]:
    return [h.strip() for h in page.locator(".left-panel__section-head").all_text_contents()]


def _last_answer(page: Page) -> str:
    return page.locator(".chat-bubble--assistant .chat-bubble__markdown").last.inner_text()


def _capture_stream(page: Page) -> list[str]:
    bodies: list[str] = []

    def handle(route):
        response = route.fetch()
        body = response.text()
        bodies.append(body)
        route.fulfill(response=response, body=body)

    page.route("**/v1/chat/stream**", handle)
    page.route("**/v1/agents/*/chat**", handle)
    return bodies


def _cited_asks(catalog: dict) -> list[str]:
    """Fixture questions whose stubbed answer carries a citation (test data)."""
    asks = [c["ask"] for c in catalog["cases"].values() if c.get("ask") and c.get("cite_any")]
    return asks[:2]


@pytest.mark.timeout(240)
@pytest.mark.parametrize("pair", [0, 1])
def test_11_1_a_session_survives_leaving_and_reload(page: Page, live_app, pair):
    base = live_app["base"]
    question = _PAIRS[pair][0]
    _login(page, base)
    _create_project(page, f"Exam11 Survive {pair}")
    project_url = page.url
    _ask(page, question)

    page.reload(wait_until="networkidle")
    expect(page.locator(".chat-bubble--user"), "stage=reload").to_have_count(1, timeout=15_000)
    page.goto(f"{base}/", wait_until="networkidle")
    page.goto(project_url, wait_until="networkidle")
    expect(page.locator(".chat-bubble--user"), "stage=leave-and-return").to_have_count(
        1, timeout=15_000
    )
    assert question in page.locator(".chat-bubble--user").first.inner_text(), "stage=history"


@pytest.mark.timeout(180)
@pytest.mark.parametrize("pair", [0, 1])
def test_11_2_sessions_are_listed_above_documents(page: Page, live_app, pair):
    _login(page, live_app["base"])
    _create_project(page, f"Exam11 Order {pair}")
    for q in _PAIRS[pair]:
        page.get_by_role("button", name="New chat").click()
        _ask(page, q)
    heads = _heads(page)
    assert "Sessions" in heads and "Documents" in heads, f"stage=sidebar {heads}"
    assert heads.index("Sessions") < heads.index("Documents"), f"stage=order {heads}"
    rows = page.locator(".left-panel__history-item")
    expect(rows, "stage=list").to_have_count(2, timeout=15_000)
    assert _PAIRS[pair][1][:20] in rows.first.inner_text(), "stage=newest-first"


@_W2
@pytest.mark.timeout(240)
@pytest.mark.parametrize("fixture_name", ["S1_contract_data.md", "S5_batch_plant_letter.md"])
def test_11_3_each_session_has_its_own_documents(page: Page, live_app, fixture_name):
    if not (FIXTURE_DIR / fixture_name).exists():
        pytest.fail(f"stage=fixture missing {fixture_name}")
    _login(page, live_app["base"])
    _create_project(page, f"Exam11 PerSession {fixture_name[:6]}")
    _ask(page, _PAIRS[0][0])
    page.locator("#doc-file-input").set_input_files(str(FIXTURE_DIR / fixture_name))

    this_session = page.locator(".left-panel__section", has_text="This session")
    expect(this_session, "stage=group-present").to_have_count(1, timeout=30_000)
    expect(this_session.filter(has_text=fixture_name), "stage=listed-in-session").to_have_count(1)
    expect(this_session.filter(has_text="Uploaded"), "stage=tagged-uploaded").to_have_count(1)

    page.get_by_role("button", name="New chat").click()
    _ask(page, _PAIRS[0][1])
    expect(
        page.locator(".left-panel__section", has_text="This session").filter(has_text=fixture_name),
        "stage=not-in-other-session",
    ).to_have_count(0)
    expect(
        page.locator(".left-panel__section", has_text="Project").filter(has_text=fixture_name),
        "stage=listed-in-project",
    ).to_have_count(1)


@_W2
@pytest.mark.timeout(240)
@pytest.mark.parametrize("pair", [0, 1])
def test_11_4_a_produced_file_is_saved_and_tagged(page: Page, live_app, pair):
    bodies = _capture_stream(page)
    _login(page, live_app["base"])
    _create_project(page, f"Exam11 Produced {pair}")
    asks = ("Generate a high-level WBS as a workbook.", "Export a manpower histogram as a workbook.")
    _ask(page, asks[pair])
    produced = [e for b in bodies for e in sse_events(b) if e.get("exports")]
    assert produced, "stage=produce: the turn produced no file"
    page.reload(wait_until="networkidle")
    this_session = page.locator(".left-panel__section", has_text="This session")
    expect(this_session.filter(has_text="Produced"), "stage=saved-and-tagged").to_have_count(
        1, timeout=15_000
    )


@_W3
@pytest.mark.timeout(240)
@pytest.mark.parametrize("which", [0, 1])
def test_11_5_citations_only_in_the_side_panel(page: Page, live_app, ui_phys_catalog, which):
    asks = _cited_asks(ui_phys_catalog)
    assert len(asks) == 2, "stage=fixture: need two cited questions"
    _login(page, live_app["base"])
    _create_project(page, f"Exam11 Cite {which}")
    _ask(page, asks[which])
    leaks = citation_leaks(_last_answer(page))
    assert leaks == [], f"stage=answer-text {leaks}"
    page.get_by_role("tab", name="Sources").click()
    expect(page.locator(".right-panel__section"), "stage=side-panel").not_to_be_empty()


@pytest.mark.timeout(240)
@pytest.mark.parametrize("pair", [0, 1])
def test_11_6_no_process_text_in_the_answer(page: Page, live_app, pair):
    _login(page, live_app["base"])
    _create_project(page, f"Exam11 Process {pair}")
    for q in _PAIRS[pair]:
        _ask(page, q)
        talk = process_talk(_last_answer(page))
        assert talk == [], f"stage=answer-text {talk}"


@pytest.mark.timeout(240)
@pytest.mark.parametrize("pair", [0, 1])
def test_11_7_the_activity_line_matches_backend_events(page: Page, live_app, pair):
    page.add_init_script(_STATUS_RECORDER)
    bodies = _capture_stream(page)
    _login(page, live_app["base"])
    _create_project(page, f"Exam11 Activity {pair}")
    _ask(page, _PAIRS[pair][0])
    assert bodies, "stage=capture: no stream was seen"
    shown = list(page.evaluate("window.__exam11Status || []"))
    if page.locator(".chat-bubble__tool-status").count():
        shown.append("<after answer>")
    mismatch = activity_mismatch(shown, ran_labels(sse_events(bodies[-1])))
    assert mismatch == [], f"stage=activity {mismatch}"
