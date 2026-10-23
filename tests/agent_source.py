"""The agent runtime's source, wherever it lives.

F-DRIVER Phase A moves Agent methods out of app/agents/runtime.py into
app/agents/core/. Tests that read the runtime's source (to find a dispatch
branch, a call site or a tool schema) read it here: runtime.py and every
core module, so they keep checking the same code after it moves.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]


def files() -> List[Path]:
    core = sorted((_ROOT / "app" / "agents" / "core").glob("*.py"))
    return [_ROOT / "app" / "agents" / "runtime.py"] + core


def text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in files())
