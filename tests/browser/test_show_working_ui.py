"""The answer shows its working, in the real React UI at phone and desktop width.

The built UI talks to a real uvicorn (sign-in, project, sessions). Only the
model turn is replaced: ``/v1/chat/stream`` answers with a recorded SSE
``end`` event in the shape the driver sends on the catalogue path (``sheet``,
``ended``, ``provenance``). Two recorded sheets of different shape: a
three-step chain that finished, and a chain that stopped on a missing input.
A third turn carries no sheet and must show no panel.

Every expectation is read from the recorded sheet itself, so a renderer that
shows a constant cannot pass both.

Requires ``frontend/dist`` and Playwright browsers; regular CI ignores this
directory (``--ignore=tests/browser``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests.browser.test_ui_phys_nightly import _create_project, _login

pytestmark = pytest.mark.browser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "show_working"
SHEETS = ("three_step_chain.json", "stopped_on_missing_input.json")
WIDTHS = (390, 1440)


def _recorded(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _sse(turn: dict) -> str:
    """The recorded turn as the stream sends it: one token, then 'end'."""
    end = {"type": "end", "content": turn["content"], "sources": [],
           "provenance": turn.get("provenance", [])}
    if "sheet" in turn:
        end["sheet"] = turn["sheet"]
        end["ended"] = turn["ended"]
    events = [{"type": "token", "content": turn["content"]}, end]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def _replay(page: Page, turn: dict) -> None:
    page.unroute("**/v1/chat/stream")
    page.route("**/v1/chat/stream", lambda route: route.fulfill(
        status=200, headers={"content-type": "text/event-stream"}, body=_sse(turn)))


def _send(page: Page, text: str) -> None:
    before = page.locator(".chat-bubble--assistant").count()
    box = page.locator(".chat-composer__textarea")
    box.fill(text)
    page.locator(".chat-composer__send").click()
    expect(page.locator(".chat-bubble--assistant")).to_have_count(before + 1, timeout=30_000)
    expect(box).to_be_enabled(timeout=30_000)


def _number(figure: str):
    m = re.match(r"\s*(-?[\d,]*\.?\d+)", figure)
    return float(m.group(1).replace(",", "")) if m else None


def _expected(turn: dict) -> dict:
    sheet = turn["sheet"]
    by_id = {s["id"]: s for s in sheet["steps"]}
    statuses = [by_id.get(p["id"], {}).get("status", "not_run") for p in sheet["plan"]]
    sourced = {
        g["id"] for g in sheet["givens"]
        if any(_number(p["figure"]) == float(g["value"]) and p.get("source") not in ("", "unresolved")
               for p in turn.get("provenance", []))
    }
    return {
        "statuses": statuses,
        "results": [by_id[p["id"]]["result"] for p in sheet["plan"]
                    if by_id.get(p["id"], {}).get("status") == "done"],
        "needs": [n for s in sheet["steps"] for n in s.get("needs", [])],
        "missing": [m["label"] for m in sheet["missing"]],
        "givens": len(sheet["givens"]),
        "unsourced": len(sheet["givens"]) - len(sourced),
        "docs": [p["doc_name"] for p in turn.get("provenance", []) if p.get("doc_name")],
    }


def _shown(n: float) -> str:
    return str(float(f"{n:.12g}")).removesuffix(".0")


@pytest.mark.timeout(240)
@pytest.mark.parametrize("width", WIDTHS)
def test_the_working_sheet_folds_under_the_answer(page: Page, live_app, width):
    base = live_app["base"]
    page.set_viewport_size({"width": width, "height": 900})
    _login(page, base)
    _create_project(page, f"Show Working {width}")

    for i, name in enumerate(SHEETS):
        turn = _recorded(name)
        want = _expected(turn)
        _replay(page, turn)
        _send(page, f"recorded turn {i + 1}")

        bubble = page.locator(".chat-bubble--assistant").last
        # The answer text is the reply as sent: the panel never writes into it.
        expect(bubble.locator(".chat-bubble__markdown")).to_have_text(turn["content"])
        panel = bubble.locator("[data-testid=show-working]")
        expect(panel).to_have_count(1)
        # Folded until asked for.
        assert panel.evaluate("el => el.open") is False
        n_steps = len(want["statuses"])
        expect(panel.locator("summary")).to_contain_text(
            f"{n_steps} step{'s' if n_steps != 1 else ''}, {want['givens']} given")
        panel.locator("summary").click()
        assert panel.evaluate("el => el.open") is True

        steps = panel.locator("[data-testid=sw-step]")
        expect(steps).to_have_count(n_steps)
        assert [steps.nth(k).get_attribute("data-status") for k in range(n_steps)] == want["statuses"]
        results = panel.locator("[data-testid=sw-result]")
        expect(results).to_have_count(len(want["results"]))
        for k, result in enumerate(want["results"]):
            expect(results.nth(k)).to_contain_text(f"Result: {_shown(result['value'])} {result['unit']}")
        for need in want["needs"]:
            expect(panel.locator("[data-testid=sw-needs]")).to_contain_text(need)
        expect(panel.locator("[data-testid=sw-missing]")).to_have_count(len(want["missing"]))
        for label in want["missing"]:
            expect(panel.locator("[data-testid=sw-missing]").filter(has_text=label)).to_have_count(1)
        expect(panel.locator("[data-testid=sw-given]")).to_have_count(want["givens"])
        for doc in want["docs"]:
            expect(panel.locator("[data-testid=sw-given]").filter(has_text=doc)).to_have_count(1)
        ending = panel.locator("[data-testid=sw-ending]")
        if want["unsourced"]:
            expect(ending).to_contain_text(f"{want['unsourced']} given")
        else:
            expect(ending).to_contain_text("every given has a source")

        # Fits the screen: nothing in the open panel runs past the viewport.
        overflow = panel.evaluate(
            "el => Math.max(...[el, ...el.querySelectorAll('*')].map(n => n.getBoundingClientRect().right))")
        assert overflow <= width, (name, width, overflow)

    # A turn off the catalogue path carries no sheet: no panel under it.
    _replay(page, {"content": "A plain answer with no working sheet."})
    _send(page, "recorded turn 3")
    last = page.locator(".chat-bubble--assistant").last
    expect(last.locator(".chat-bubble__markdown")).to_have_text("A plain answer with no working sheet.")
    expect(last.locator("[data-testid=show-working]")).to_have_count(0)
