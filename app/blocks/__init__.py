"""Platform Blocks — Cerebrum runtime block registry.

Virgin Fork (default): ~17 generic blocks + ``DomainContainer`` host.
Domain kits (construction, etc.) register via ``CEREBRUM_DOMAIN_KITS`` or
Block Store install → ``data/domain_kit_registry.json``.

Set ``CEREBRUM_VIRGIN=false`` for legacy full-platform boot (production Fork).
"""

import importlib
import logging
import os
from typing import Dict, List, Tuple

from app.core.universal_base import UniversalBlock, UniversalContainer
from app.core.typed_block import TypedBlock

logger = logging.getLogger(__name__)

# Plug-and-play generic blocks — always loaded (virgin Fork outcome #1).
_GENERIC_BLOCK_SPECS: List[Tuple[str, str, str]] = [
    ("pdf",              "app.blocks.pdf",             "PDFBlock"),
    ("ocr",              "app.blocks.ocr",             "OCRBlock"),
    ("image",            "app.blocks.image",           "ImageBlock"),
    ("document_engine",  "app.blocks.document_engine", "DocumentEngineBlock"),
    ("chat",             "app.blocks.chat",            "ChatBlock"),
    ("translate",        "app.blocks.translate",       "TranslateBlock"),
    ("voice",            "app.blocks.voice",           "VoiceBlock"),
    ("web",              "app.blocks.web",             "WebBlock"),
    ("search",           "app.blocks.search",          "SearchBlock"),
    ("code",             "app.blocks.code",            "CodeBlock"),
    ("cache_manager",    "app.blocks.cache_manager",   "CacheManagerBlock"),
    ("file_hasher",      "app.blocks.file_hasher",     "FileHasherBlock"),
    ("orchestrator",     "app.blocks.orchestrator",    "OrchestratorBlock"),
    ("validation_pipeline", "app.blocks.validation_pipeline", "ValidationPipelineBlock"),
    ("async_processor",  "app.blocks.async_processor", "AsyncProcessorBlock"),
    # F35: every block an agent config references must LOAD on every
    # deployment -- gated behind CEREBRUM_VIRGIN=false they left shipped
    # agents listed but silently toolless (external-mcp had no mcp_consumer;
    # document-ingestion lost its three drive connectors; self-coding lost
    # its sandbox pre-flight). Blocks needing outbound access still refuse
    # under the on-prem (air-gap) profile at call time; drive connectors
    # without credentials error loudly naming the missing config.
    ("mcp_consumer",     "app.blocks.mcp_consumer",    "MCPConsumerBlock"),
    ("local_drive",      "app.blocks.local_drive",     "LocalDriveBlock"),
    ("google_drive",     "app.blocks.google_drive",    "GoogleDriveBlock"),
    ("onedrive",         "app.blocks.onedrive",        "OneDriveBlock"),
    ("sandbox",          "app.blocks.sandbox",         "SandboxBlock"),
    ("aconex",           "app.blocks.aconex",          "AconexBlock"),
]

# Extended platform blocks — loaded when CEREBRUM_VIRGIN=false (legacy production).
# Construction domain blocks are NEVER here; they load only via domain kits.
_EXTENDED_PLATFORM_SPECS: List[Tuple[str, str, str]] = [
    ("pdf_v2",              "app.blocks.pdf_v2",               "PDFBlockV2"),
    ("ocr_v2",              "app.blocks.ocr_v2",               "OCRBlockV2"),
    ("llm_enhancer",        "app.blocks.llm_enhancer",         "LLMEnhancerBlock"),
    ("android_drive",       "app.blocks.android_drive",        "AndroidDriveBlock"),
    ("mcp_adapter",         "app.blocks.mcp_adapter",          "MCPAdapterBlock"),
    ("webhook",             "app.blocks.webhook",              "WebhookBlock"),
]


def _legacy_boot() -> bool:
    return os.getenv("CEREBRUM_VIRGIN", "true").strip().lower() in ("0", "false", "no")


def _build_block_specs() -> List[Tuple[str, str, str]]:
    from app.core.domain_kit_loader import kit_block_specs, verify_installed_containers

    verify_installed_containers()
    specs = list(_GENERIC_BLOCK_SPECS)
    seen = {name for name, _, _ in specs}

    if _legacy_boot():
        for item in _EXTENDED_PLATFORM_SPECS:
            if item[0] not in seen:
                specs.append(item)
                seen.add(item[0])

    for item in kit_block_specs():
        if item[0] not in seen:
            specs.append(item)
            seen.add(item[0])

    return specs


def _load_blocks(
    specs: List[Tuple[str, str, str]]
) -> Tuple[Dict[str, type], Dict[str, str]]:
    """Import each ``(name, module, class)`` spec in isolation."""
    registry: Dict[str, type] = {}
    failed: Dict[str, str] = {}
    for name, module, class_name in specs:
        try:
            mod = importlib.import_module(module)
            registry[name] = getattr(mod, class_name)
        except Exception as exc:  # noqa: BLE001
            failed[name] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Block '%s' failed to load and was skipped: %s", name, exc
            )
    return registry, failed


_BLOCK_SPECS = _build_block_specs()
BLOCK_REGISTRY, FAILED_BLOCKS = _load_blocks(_BLOCK_SPECS)

for _cls in BLOCK_REGISTRY.values():
    globals()[_cls.__name__] = _cls

if FAILED_BLOCKS:
    logger.warning(
        "%d/%d blocks failed to load: %s",
        len(FAILED_BLOCKS), len(_BLOCK_SPECS), ", ".join(sorted(FAILED_BLOCKS)),
    )

from app.core.domain_kit_loader import active_kit_ids as _active_kit_ids

if _legacy_boot():
    logger.info("block registry: legacy boot (%d blocks)", len(BLOCK_REGISTRY))
else:
    logger.info(
        "block registry: virgin boot (%d generic + %d kit blocks)",
        len(_GENERIC_BLOCK_SPECS),
        len(BLOCK_REGISTRY) - len(_GENERIC_BLOCK_SPECS),
    )

_kits = _active_kit_ids()
if _kits:
    logger.info("active domain kits: %s", ", ".join(_kits))
else:
    logger.info("active domain kits: none")


def get_block(name: str):
    return BLOCK_REGISTRY.get(name)


def get_all_blocks():
    return BLOCK_REGISTRY


__all__ = [
    "UniversalBlock",
    "UniversalContainer",
    "TypedBlock",
    "BLOCK_REGISTRY",
    "FAILED_BLOCKS",
    "_load_blocks",
    "get_block",
    "get_all_blocks",
]
