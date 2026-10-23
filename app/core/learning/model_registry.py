"""Model registry — maps logical names to current HF model IDs.

``scripts/finetune_router.py`` resolves its base model through
:func:`resolve_base_model` so deprecation swaps are one place.
"""

from __future__ import annotations

from typing import Dict


_REGISTRY: Dict[str, str] = {
    # Construction-domain chat model. Local default stays small —
    # Qwen2.5-3B-Instruct fits on a workstation card and runs at usable
    # speeds without quantization.
    "construction_v1": "Qwen/Qwen2.5-3B-Instruct",
}


def resolve_base_model(logical_name: str) -> str:
    """Return the concrete model ID for ``logical_name``.

    Raises ValueError on an unknown alias — silent fallbacks would hide
    real misconfigurations.
    """
    if logical_name not in _REGISTRY:
        raise ValueError(
            f"unknown model alias {logical_name!r}; "
            f"choices: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[logical_name]


def list_aliases() -> Dict[str, str]:
    """Return a copy of the registry for inspection / dashboards."""
    return dict(_REGISTRY)
