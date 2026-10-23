"""CORE tool failure: the answer names the failed items whole, and states
no invented figure."""
from __future__ import annotations

import pytest

from app.agents.core.tool_registry import _TOOLS, tool
from tests.lane5_core_support import answer_text, call, drive


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def test_a_failed_tool_is_narrated_with_its_items_whole(monkeypatch):
    name = "lane5_item_check"

    @tool(name, owner="base", display_name="Item check")
    async def handle(_call):
        return {
            "name": name,
            "ok": False,
            "error": "Could not use these items: red, green, blue",
            "result": {"items": ["red", "green", "blue"], "item_count": 0},
        }

    try:
        events, _seen = drive(
            monkeypatch,
            "Check the qxnet item set.",
            [
                {"content": "", "tool_calls": [call(name, {}, "c1")]},
                {"content": "0 items were found. The total is 9."},
            ],
        )
    finally:
        _TOOLS.pop(name, None)
    text = answer_text(events)
    assert "red" in text and "green" in text and "blue" in text
    assert "0 items" not in text
    assert "9" not in text
