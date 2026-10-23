from fastapi import APIRouter, HTTPException

from app.blocks import BLOCK_REGISTRY, get_all_blocks
from app.core.universal_base import UniversalContainer
from app.dependencies import block_instances, _create_block_instance

router = APIRouter()


@router.get("/blocks")
def list_blocks():
    """List all available blocks."""
    blocks = []
    for name, block_class in get_all_blocks().items():
        # Skip containers - they belong to Block Store
        if issubclass(block_class, UniversalContainer):
            continue
        try:
            if name not in block_instances:
                block_instances[name] = _create_block_instance(block_class)
            instance = block_instances[name]

            blocks.append({
                "name": name,
                "version": getattr(instance, "version", "1.0"),
                "description": getattr(instance, "description", ""),
                "layer": getattr(instance, "layer", 3),
                "tags": getattr(instance, "tags", []),
                "requires": getattr(instance, "requires", []),
                "ui_schema": getattr(block_class, "ui_schema", {}),
            })
        except Exception as e:
            blocks.append({
                "name": name,
                "error": str(e),
                "status": "failed",
            })

    return {
        "blocks": blocks,
        "total": len(blocks),
        "categories": {
            "ai": ["chat", "pdf", "ocr", "voice", "image", "translate", "code", "web", "search"],
            "storage": ["google_drive", "onedrive", "local_drive", "android_drive"],
        },
    }


@router.get("/blocks/{block_name}")
def get_block_info(block_name: str):
    """Get block details."""
    if block_name not in BLOCK_REGISTRY:
        raise HTTPException(404, f"Block '{block_name}' not found")

    if block_name not in block_instances:
        block_instances[block_name] = _create_block_instance(BLOCK_REGISTRY[block_name])

    instance = block_instances[block_name]
    block_class = BLOCK_REGISTRY[block_name]

    return {
        "name": block_name,
        "config": {
            "version": getattr(instance, "version", "1.0"),
            "description": getattr(instance, "description", ""),
            "layer": getattr(instance, "layer", 3),
            "tags": getattr(instance, "tags", []),
            "requires": getattr(instance, "requires", []),
        },
        "ui_schema": getattr(block_class, "ui_schema", {}),
        "stats": instance.get_stats(),
    }


@router.get("/v1/blocks")
def list_blocks_v1():
    """List all available blocks (v1 API)."""
    return list_blocks()


@router.get("/v1/blocks/{block_name}")
def get_block_v1(block_name: str):
    """Get block details (v1 API)."""
    return get_block_info(block_name)
