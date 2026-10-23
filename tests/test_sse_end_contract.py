"""Every SSE ``end`` event in the codebase must carry a ``tools`` key.

#390 added the field to all eight ``end`` events in ``app/agents/runtime.py``
and stopped there. Four more live in ``app/routers/chat.py`` -- including the
predefined-dispatch path, which runs a container action directly rather than
through the agent tool loop. So a live ``payment_certificate`` turn still
arrived with ``tools`` ABSENT and rendered as ``tools=[]``: the exact
answer-versus-badge disagreement #390 set out to fix, on the one path it did
not touch.

A per-path test cannot catch that -- it only covers paths someone remembered
to write a test for, which is the same blind spot. This walks the source
instead, so a NEW ``end`` emitter added anywhere fails until it joins the
contract. An absent key and an empty list render identically in the UI, so
"ran nothing" must be stated, never implied by silence.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"


def _dict_keys(node: ast.Dict) -> set[str]:
    return {
        k.value for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }


def _is_end_event(node: ast.Dict) -> bool:
    """True for a dict literal with ``type`` set to the constant "end"."""
    for key, val in zip(node.keys, node.values):
        if (isinstance(key, ast.Constant) and key.value == "type"
                and isinstance(val, ast.Constant) and val.value == "end"):
            return True
    return False


def _end_events() -> list[tuple[Path, int, set[str]]]:
    """Every ``{"type": "end", ...}`` dict literal under app/.

    ast reaches into f-strings, so the ``json.dumps({...})`` emitters in
    chat.py are covered the same as the plain ``yield {...}`` ones in
    runtime.py.
    """
    found: list[tuple[Path, int, set[str]]] = []
    for py in sorted(APP.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - would fail the suite elsewhere
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and _is_end_event(node):
                found.append((py, node.lineno, _dict_keys(node)))
    return found


def test_end_events_are_actually_found():
    """Guard the guard: an ast change that stops matching would make every
    assertion below vacuously true."""
    events = _end_events()
    assert len(events) >= 12, (
        f"expected to find the known SSE end emitters, found {len(events)}"
    )


@pytest.mark.parametrize(
    "path,lineno,keys",
    [pytest.param(p, ln, k, id=f"{p.name}:{ln}") for p, ln, k in _end_events()],
)
def test_every_end_event_declares_tools(path: Path, lineno: int, keys: set[str]):
    assert "tools" in keys, (
        f"SSE end event at {path}:{lineno} has no 'tools' key. "
        f"The UI cannot tell an absent key from an empty list, so a turn that "
        f"ran a tool renders as tools=[]. Emit the tools that ran, or [] if "
        f"this path genuinely runs none. Keys present: {sorted(keys)}"
    )


# ── tool_call / tool_result key parity ─────────────────────────────────────

def _tool_events() -> list[tuple[str, int, set[str]]]:
    """Every ``{"type": "tool_call"|"tool_result", ...}`` dict under app/."""
    found = []
    for py in sorted(APP.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, val in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "type"
                        and isinstance(val, ast.Constant)
                        and val.value in ("tool_call", "tool_result")):
                    found.append((val.value, node.lineno, _dict_keys(node)))
    return found


@pytest.mark.parametrize(
    "kind,lineno,keys",
    [pytest.param(k, ln, ks, id=f"{k}:{ln}") for k, ln, ks in _tool_events()],
)
def test_tool_events_carry_the_key_the_browser_reads(kind, lineno, keys):
    """Both emitters must expose ``tool``, not just ``name``.

    The main tool loop emits ``tool`` (what the frontend reads) with ``name``
    as an alias. The deterministic file pre-dispatch emitted the reverse --
    ``name`` and no ``tool`` -- so a pre-dispatched fetch_document or
    bim_extractor rendered as a blank, unnamed entry. The turn narrated work
    the user could not see it do.
    """
    assert "tool" in keys, (
        f"{kind} at runtime line {lineno} has no 'tool' key; the browser reads "
        f"'tool' and would render this as blank. Keys: {sorted(keys)}"
    )
    assert "name" in keys, (
        f"{kind} at runtime line {lineno} dropped the 'name' alias, which "
        f"callback consumers read. Keys: {sorted(keys)}"
    )
