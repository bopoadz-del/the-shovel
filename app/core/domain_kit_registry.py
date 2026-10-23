"""Persisted domain kit registry — written by Block Store install, read at boot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "domain_kit_registry.json"


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"kits": {}}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("kits", {})
    return data


def save_registry(data: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def register_kit(
    kit_id: str,
    *,
    container_class: str,
    blocks: list[str] | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Record a kit so boot can import its container and blocks."""
    state = load_registry()
    state["kits"][kit_id] = {
        "container_class": container_class,
        "blocks": blocks or [],
        "version": version,
    }
    save_registry(state)
    return state["kits"][kit_id]


def unregister_kit(kit_id: str) -> None:
    state = load_registry()
    state["kits"].pop(kit_id, None)
    save_registry(state)


def enabled_kit_ids() -> list[str]:
    return sorted(load_registry().get("kits", {}).keys())
